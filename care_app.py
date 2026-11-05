"""Purpose of this program: the Streamlit app for the Patient Intake &
Condition Guide, used by a caregiver together with the patient.

Pages, in a slim top bar (no sidebar):
  0. Home                - what the app does, stat cards, Start new intake.
  1. New Intake          - one question per screen: big question, big answer
                           buttons, a typed box, Back / Skip / Next, a thin
                           progress bar and every answer as a chip at the
                           bottom (tap one to go back to it). The urgent
                           signs check runs after every answer and stops the
                           intake with a big "Get emergency care now"
                           banner. Then it makes both drafts.
  2. Patient Info Sheet  - review and edit the plain-language sheet.
  3. Care Team Briefing  - review and edit the briefing.
  4. Patient History     - approved records, with PDF / SMS / Markdown export.

Human approval: the drafts are only saved (agents/care_store.py) after the
caregiver ticks "I reviewed it" on both and clicks Approve. Exports are only
offered for approved records.

Privacy: the patient's name (optional) is kept only in this browser session
to mask it in answers; it is never saved or sent to OpenAI or Langfuse.

Run it with:  streamlit run care_app.py
"""

import re
import uuid
from collections import Counter
from datetime import datetime, timedelta

import streamlit as st

from agents.care_coordinator import run_care_flow
from agents.care_export import care_team_markdown, care_team_pdf, intake_lines, sheet_pdf, sheet_sms
from agents.care_models import NO_DATA, Point
from agents.care_store import get_record, list_records, save_approved
from agents.intake_agent import (QUESTION_BY_FIELD, QUESTIONS, answer_question, new_session,
                                 next_question, progress)
from agents.patient_education import reading_grade
from guardrails.grounding import safety_problem
from triage.urgent_signs import BANNER_TEXT, BANNER_TITLE, check_urgent

st.set_page_config(page_title="Patient Intake & Condition Guide",
                   page_icon=":material/medical_services:", layout="centered",
                   initial_sidebar_state="collapsed")

# Styles, made for a phone or tablet held by the caregiver:
#   - no sidebar (pages are in the top bar), less empty space at the top
#   - big buttons; the small "Start over" button and the answer chips stay small
#   - a thin progress bar, a big question, a big red emergency banner
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; }
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
    min-height: 3.2rem; font-size: 1.15rem; font-weight: 600; }
.st-key-start_over button { min-height: 2.1rem; font-size: 0.9rem; font-weight: 500; }
.topbar { display: flex; justify-content: space-between; align-items: center;
    font-size: 0.95rem; color: #52606D; padding: 0.2rem 0; }
.topbar b { color: #1F2933; }
.thin-progress { height: 5px; background: #E4EBEF; border-radius: 3px; margin: 0.3rem 0 0.2rem; }
.thin-progress > div { height: 5px; background: #0B6E8A; border-radius: 3px; }
.step { font-size: 0.85rem; color: #52606D; margin-bottom: 1.2rem; }
.question { font-size: 1.7rem; font-weight: 700; line-height: 1.3; margin-bottom: 0.3rem; }
.hint { color: #52606D; margin-bottom: 1rem; }
.urgent-banner { background: #B00020; color: white; padding: 1.2rem 1.4rem;
    border-radius: 0.6rem; margin-bottom: 1rem; }
.urgent-banner h2 { color: white; margin: 0 0 0.4rem 0; font-size: 2rem; }
</style>""", unsafe_allow_html=True)

ss = st.session_state  # short name; everything that must survive a rerun lives here


# --- Small helpers ---
def points_to_text(points: list[Point]) -> str:
    """One point per line, sources in brackets: 'Fever is common. [M1, P2]'."""
    return "\n".join(f"{p.text} [{', '.join(p.sources)}]" if p.sources else p.text
                     for p in points)


def text_to_points(text: str) -> list[Point]:
    """The reverse. A line without [ids] was written by the caregiver."""
    points = []
    for line in filter(None, (l.strip() for l in text.splitlines())):
        m = re.match(r"^(.*?)\s*\[([^\]]*)\]\s*$", line)
        if m:
            points.append(Point(text=m.group(1), sources=[s.strip() for s in m.group(2).split(",")
                                                         if s.strip()]))
        else:
            points.append(Point(text=line, sources=[]))
    # Once the caregiver adds a real line, the "No current data found." line goes.
    real = [p for p in points if p.text != NO_DATA]
    return real or [Point(text=NO_DATA, sources=[])]


def edit_points(label: str, obj, field: str) -> None:
    """A text box for one section. Edits are copied into the draft at once
    (on_change), so they survive switching pages."""
    key = f"{ss.run_id}-{id(obj)}-{field}"

    def save():
        setattr(obj, field, text_to_points(ss[key]))

    st.text_area(label, value=points_to_text(getattr(obj, field)), key=key, on_change=save,
                 height=120)
    for p in getattr(obj, field):
        if problem := safety_problem(p.text):
            st.warning(f"This line {problem}: \"{p.text}\". Please change it.", icon=":material/warning:")
        elif not p.sources and p.text != NO_DATA:
            st.caption(f"Added by caregiver (no source): {p.text}")


def edit_lines(label: str, obj, field: str) -> None:
    """A text box for a list of plain lines (questions, notes)."""
    key = f"{ss.run_id}-{id(obj)}-{field}"

    def save():
        setattr(obj, field, [l.strip() for l in ss[key].splitlines() if l.strip()])

    st.text_area(label, value="\n".join(getattr(obj, field)), key=key, on_change=save, height=110)


def include_box(label: str, group: str, item_id: str) -> None:
    """A tick box to keep or remove one item (study, treatment, organisation)."""
    excluded = ss.draft["excluded"].setdefault(group, set())
    key = f"{ss.run_id}-{group}-{item_id}"

    def save():
        (excluded.discard if ss[key] else excluded.add)(item_id)

    st.checkbox(label, value=item_id not in excluded, key=key, on_change=save)


def urgent_banner(result) -> None:
    signs = "".join(f"<li><b>{s.name}</b>: {s.reason}</li>" for s in result.signs)
    st.markdown(f'<div class="urgent-banner"><h2>{BANNER_TITLE}</h2>'
                f"<p>{BANNER_TEXT}</p><ul>{signs}</ul></div>", unsafe_allow_html=True)
    st.subheader("While you get help")
    for i, step in enumerate(result.first_aid, 1):
        st.markdown(f"**{i}.** {step}")


def no_draft_message() -> bool:
    if "draft" not in ss:
        st.info("No draft yet. Finish a New Intake first.", icon=":material/info:")
        st.page_link(PAGES["intake"], label="Go to New Intake", icon=":material/arrow_forward:")
        return True
    return False


# --- Page 0: Home (the only page with stat cards) ---
def page_home() -> None:
    st.title("Patient Intake & Condition Guide")
    st.write("For caregivers where a doctor isn't readily available. Do a short intake with the "
             "patient, then get a plain-language info sheet and a care team briefing. "
             "It does not diagnose and never gives doses.")

    records = list_records()
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    conditions = Counter((r["condition"] or "").lower() for r in records if r["condition"])
    cards = [("Approved records", len(records)),
             ("Patients", len({r["patient_id"] for r in records})),
             ("Last 7 days", sum(r["approved_at"] >= week_ago for r in records)),
             ("Most common", conditions.most_common(1)[0][0].capitalize() if conditions else "-")]
    for col, (label, value) in zip(st.columns(4), cards):
        with col.container(border=True):
            st.metric(label, value)

    if st.button("Start new intake", type="primary", width="stretch", icon=":material/person_add:"):
        reset_intake()
        st.switch_page(PAGES["intake"])
    if st.button("Patient history", width="stretch", icon=":material/history:"):
        st.switch_page(PAGES["history"])


# --- Page 1: New Intake (one question per screen) ---
# Big answer buttons for common answers: (button label, the answer it sends).
# The typed box is always there too, for any other answer.
CHOICES = {
    "preferred_language": [(x, x) for x in ("English", "Swahili", "French", "Spanish",
                                            "Portuguese", "Arabic")],
    "sex": [("Female", "female"), ("Male", "male"), ("Other", "other")],
    "pregnant": [("Yes", "yes"), ("No", "no"), ("Not sure", "maybe")],  # not sure counts as yes
    "reported_condition": [("No, not told", "no")],
    "known_conditions": [("None", "none")],
    "current_meds": [("None", "none")],
    "allergies": [("None", "none")],
    "past_surgeries_hospital_stays": [("None", "none")],
    "family_history": [("None", "none")],
    "lifestyle": [("No smoking, no alcohol", "no smoking, no alcohol")],
    "caregiver_notes": [("Nothing else", "no")],
}
# Short names for the answer chips at the bottom of the screen.
SHORT = {"preferred_language": "Language", "age": "Age", "sex": "Sex", "pregnant": "Pregnant",
         "weight_kg": "Weight", "chief_complaint": "Main problem", "symptoms": "Symptoms",
         "reported_condition": "Told they have", "known_conditions": "Conditions",
         "current_meds": "Medicines", "allergies": "Allergies",
         "past_surgeries_hospital_stays": "Surgeries", "family_history": "Family",
         "lifestyle": "Smoking/alcohol", "vitals": "Vitals", "caregiver_notes": "Notes"}


def reset_intake() -> None:
    for k in ("session", "urgent", "draft", "approved", "cursor"):
        ss.pop(k, None)


def applicable(session) -> list:
    """The questions that apply to this patient, in order."""
    return [q for q in QUESTIONS if q.applies(session.record)]


def current_question(session):
    """The question on screen: one the caregiver went back to (ss.cursor), or
    else the first one not answered yet. None when every question is done."""
    cursor = ss.get("cursor")
    if cursor and QUESTION_BY_FIELD[cursor].applies(session.record):
        return QUESTION_BY_FIELD[cursor]
    return next_question(session)


def submit(session, field: str, answer: str) -> None:
    """Save one answer (masked first, inside answer_question), run the urgent
    check, then move to the next question."""
    with st.spinner("Reading the answer..."):
        answer_question(session, field, answer)
    ss.urgent = check_urgent(session.record, session.answers)
    # Next screen: if the following question was already answered (we are
    # going back through earlier answers), show it; otherwise the first
    # unanswered one. On an urgent sign, show the banner.
    qs = applicable(session)
    fields = [q.field for q in qs]
    after = fields[fields.index(field) + 1:] if field in fields else []
    if ss.urgent.urgent or not after or after[0] not in session.answers:
        ss.cursor = None
    else:
        ss.cursor = after[0]
    st.rerun()


def go_back(session, question) -> None:
    fields = [q.field for q in applicable(session)]
    if question is None:  # from the "done" screen: back to the last question
        ss.cursor = fields[-1]
    elif question.field in fields and fields.index(question.field) > 0:
        ss.cursor = fields[fields.index(question.field) - 1]
    st.rerun()


def top_bar(session) -> None:
    """Slim bar: app name and patient ID on the left, Start over on the right."""
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.markdown(f'<div class="topbar"><span><b>New intake</b> · Patient '
                  f'{session.record.patient_id}</span></div>', unsafe_allow_html=True)
    if right.button("Start over", key="start_over", width="stretch"):
        reset_intake()
        st.rerun()


def thin_progress(session, question) -> None:
    qs = applicable(session)
    done, total = progress(session)
    number = ([q.field for q in qs].index(question.field) + 1) if question in qs else total
    st.markdown(f'<div class="thin-progress"><div style="width:{100 * done / max(total, 1):.0f}%">'
                f'</div></div><div class="step">Question {number} of {total}</div>',
                unsafe_allow_html=True)


def answer_chips(session) -> None:
    """Every answer so far as a small chip. Tapping one goes back to it."""
    fields = [q.field for q in applicable(session) if q.field in session.answers]
    if not fields:
        return

    def label(f: str) -> str:
        text = session.answers[f].strip() or "skipped"
        text = "skipped" if text.lower() == "skip" else text
        return f"{SHORT[f]}: {text[:24]}{'...' if len(text) > 24 else ''}"

    def jump():
        ss.cursor = ss.chips
        ss.chips = None  # clear the selection so the same chip can be tapped again

    st.divider()
    st.pills("Answers so far (tap one to change it)", fields, format_func=label, key="chips",
             on_change=jump)
    if session.pii_found:
        st.caption(f":material/shield_lock: Hidden before anything was sent: "
                   f"{', '.join(session.pii_found)}")


def question_screen(session, question) -> None:
    """One question: big text, big answer buttons, a typed box, Back/Skip/Next."""
    st.markdown(f'<div class="question">{question.text}</div>', unsafe_allow_html=True)
    if question.hint:
        st.markdown(f'<div class="hint">Example: {question.hint}</div>', unsafe_allow_html=True)
    current = session.answers.get(question.field, "")

    # Big answer buttons, two per row. The current answer is highlighted.
    choices = CHOICES.get(question.field, [])
    for row in range(0, len(choices), 2):
        for col, (label, value) in zip(st.columns(2), choices[row:row + 2]):
            chosen = current.strip().lower() == value.lower()
            if col.button(label, key=f"choice-{question.field}-{label}", width="stretch",
                          type="primary" if chosen else "secondary"):
                submit(session, question.field, value)

    # Typed answer + footer. A form lets Enter submit, and keeps the box
    # from rerunning the app on every keystroke.
    with st.form(f"form-{question.field}", border=False):
        typed = st.text_input("Or type the answer" if choices else "Answer",
                              value="" if current.lower() == "skip" else current,
                              placeholder=question.hint,
                              label_visibility="visible" if choices else "collapsed")
        back, skip, nxt = st.columns([1, 1, 2])
        is_first = question.field == applicable(session)[0].field
        pressed_back = back.form_submit_button("Back", width="stretch", disabled=is_first,
                                               icon=":material/arrow_back:")
        pressed_skip = skip.form_submit_button("Skip", width="stretch")
        pressed_next = nxt.form_submit_button("Next", type="primary", width="stretch",
                                              icon=":material/arrow_forward:")
    if pressed_back:
        go_back(session, question)
    if pressed_skip:
        submit(session, question.field, "skip")
    if pressed_next:
        submit(session, question.field, typed)


def done_screen(session) -> None:
    st.markdown('<div class="question">Intake complete</div>', unsafe_allow_html=True)
    st.success("No urgent signs found.", icon=":material/check_circle:")
    with st.expander("Intake summary"):
        st.markdown("\n".join(f"- {line}" for line in intake_lines(session.record)))

    st.write("Which condition should the information be about? This app does not diagnose: "
             "use what a health worker told the patient.")
    condition = st.text_input("Condition", value=session.record.reported_condition or "",
                              placeholder="e.g. malaria, type 2 diabetes")
    back, make = st.columns([1, 3])
    if back.button("Back", width="stretch", icon=":material/arrow_back:"):
        go_back(session, None)
    if make.button("Make patient sheet and briefing", type="primary", width="stretch",
                   icon=":material/description:", disabled=not condition.strip()):
        make_drafts(condition)
    if "draft" in ss:
        st.page_link(PAGES["sheet"], label="Review the Patient Info Sheet",
                     icon=":material/arrow_forward:")
        st.page_link(PAGES["briefing"], label="Review the Care Team Briefing",
                     icon=":material/arrow_forward:")


def page_intake() -> None:
    if "session" not in ss:
        st.title("New Intake")
        st.write("Ask the patient each question in simple words. Tap an answer or type it. "
                 "You can skip any question and change answers later.")
        name = st.text_input("Patient's name (optional)",
                             help="Only used to hide the name in answers. Never saved or sent.")
        if st.button("Start", type="primary", width="stretch", icon=":material/play_arrow:"):
            reset_intake()
            ss.session = new_session(known_names=[name] if name else [])
            st.rerun()
        return

    session = ss.session
    top_bar(session)
    question = current_question(session)

    # The urgent check runs after every answer. If it fires, the intake stops
    # (unless the caregiver tapped a chip to correct a mistyped answer).
    if ss.get("urgent") and ss.urgent.urgent and not ss.get("cursor"):
        urgent_banner(ss.urgent)
        st.caption("If an answer was typed by mistake, tap it below to correct it. "
                   "The check runs again.")
        answer_chips(session)
        return

    thin_progress(session, question)
    if question:
        question_screen(session, question)
    else:
        done_screen(session)
    answer_chips(session)


def make_drafts(condition: str) -> None:
    labels = {"urgent_check": "Urgent signs check: none found",
              "lookup": "Looked up trials, articles, drug labels and patient info",
              "patient_sheet": "Patient Info Sheet drafted and checked",
              "briefing": "Care Team Briefing drafted and checked"}
    with st.status("Working on it (about 20-40 seconds)...", expanded=True) as status:
        result = run_care_flow(ss.session.record, ss.session.answers, condition,
                               on_step=lambda s: status.write(labels.get(s, s)))
        status.update(label=f"Done in {result['seconds']} s", state="complete", expanded=False)
    if result["status"] == "urgent":  # the check runs again inside the flow
        ss.urgent = result["urgent"]
        st.rerun()
    ss.run_id = uuid.uuid4().hex[:8]  # new widget keys for the new drafts
    ss.draft = {**result, "excluded": {}, "reviewed_sheet": False, "reviewed_briefing": False}
    ss.pop("approved", None)
    for note in result["notes"]:
        st.info(note, icon=":material/info:")


# --- Shared: approve block ---
def approve_block() -> None:
    st.divider()
    draft = ss.draft
    if "approved" in ss:
        st.success(f"Approved and saved (record #{ss.approved}).", icon=":material/verified:")
        st.page_link(PAGES["history"], label="Open in Patient History to export",
                     icon=":material/arrow_forward:")
        return
    st.subheader("Approve")
    c1, c2 = st.columns(2)
    for col, name, label in [(c1, "reviewed_sheet", "I reviewed the Patient Info Sheet"),
                             (c2, "reviewed_briefing", "I reviewed the Care Team Briefing")]:
        key = f"{ss.run_id}-{name}"
        # on_change copies the tick into the draft, so it survives page changes.
        col.checkbox(label, value=draft[name], key=key,
                     on_change=lambda n=name, k=key: draft.__setitem__(n, ss[k]))
    unsafe = [p.text for p in all_points() if safety_problem(p.text)]
    if unsafe:
        st.error("Some lines give a dose, prescribe or diagnose. Change them before approving.",
                 icon=":material/block:")
    ready = draft["reviewed_sheet"] and draft["reviewed_briefing"] and not unsafe
    if st.button("Approve and save", type="primary", width="stretch", disabled=not ready,
                 icon=":material/verified:"):
        sheet, briefing = final_outputs()
        ss.approved = save_approved(ss.session.record, draft["condition"], sheet, briefing)
        st.rerun()
    if not ready:
        st.caption("Tick both boxes after reading each draft. Nothing is saved until you approve.")


def all_points() -> list[Point]:
    s, b = ss.draft["sheet"], ss.draft["briefing"]
    pts = [p for f in ("what_it_is", "common_signs", "how_it_is_usually_treated",
                       "what_you_can_do_at_home", "warning_signs_get_help_now") for p in getattr(s, f)]
    pts += b.overview + [p for f in ("first_line", "second_line", "supportive_care", "when_to_refer")
                         for p in getattr(b.standard_of_care, f)]
    return pts


def final_outputs():
    """Copies of the drafts with the removed items taken out."""
    ex = ss.draft["excluded"]
    sheet = ss.draft["sheet"].model_copy(deep=True)
    sheet.research_studies_you_can_ask_about = [
        s for s in sheet.research_studies_you_can_ask_about if s.nct_id not in ex.get("studies", set())]
    briefing = ss.draft["briefing"].model_copy(deep=True)
    briefing.emerging_treatments = [t for t in briefing.emerging_treatments
                                    if t.nct_id + t.name not in ex.get("treatments", set())]
    for group in ("companies", "academic_institutions", "government_ngo"):
        setattr(briefing.key_players, group, [p for p in getattr(briefing.key_players, group)
                                              if p.name not in ex.get("players", set())])
    return sheet, briefing


# --- Page 2: Patient Info Sheet ---
def page_sheet() -> None:
    st.title("Patient Info Sheet")
    if no_draft_message():
        return
    sheet = ss.draft["sheet"]
    st.caption(f"About **{sheet.condition}** · written in {sheet.language} · "
               "one point per line; the [ids] show where each point came from.")
    if sheet.language.lower().startswith("en"):
        st.caption(f"Reading level: about grade {reading_grade(sheet):.0f} (goal: about 6)")
    if ss.draft["sheet_report"]:
        with st.expander("Safety and source check removed some text"):
            st.markdown("\n".join(f"- {r}" for r in ss.draft["sheet_report"]))

    edit_lines("Important for you (from the intake)", sheet, "personal_notes")
    for field, label in [("what_it_is", "What it is"), ("common_signs", "Common signs"),
                         ("how_it_is_usually_treated", "How it is usually treated"),
                         ("what_you_can_do_at_home", "What you can do at home"),
                         ("warning_signs_get_help_now", "Warning signs: get help now")]:
        edit_points(label, sheet, field)
    edit_lines("Questions to ask your health worker", sheet, "questions_to_ask_your_health_worker")

    st.markdown("**Research studies you can ask about** (recruiting now)")
    if not sheet.research_studies_you_can_ask_about:
        st.write(NO_DATA)
    for s in sheet.research_studies_you_can_ask_about:
        include_box(f"{s.plain_summary} [{s.nct_id}](<{s.url}>) · "
                    f"{', '.join(s.countries) or 'country not listed'}", "studies", s.nct_id)

    st.info(sheet.disclaimer, icon=":material/info:")
    with st.expander(f"Sources ({len(sheet.sources)})"):
        st.markdown("\n".join(f"- **{s.id}** [{s.title}](<{s.url}>) · {s.type}" for s in sheet.sources))
    approve_block()


# --- Page 3: Care Team Briefing ---
def page_briefing() -> None:
    st.title("Care Team Briefing")
    if no_draft_message():
        return
    b = ss.draft["briefing"]
    st.caption(f"**{b.condition}** · made {b.generated_at.replace('T', ' ')}")
    if ss.draft["briefing_report"]:
        with st.expander("Safety and source check removed some text"):
            st.markdown("\n".join(f"- {r}" for r in ss.draft["briefing_report"]))

    with st.expander("Intake summary"):
        st.markdown("\n".join(f"- {line}" for line in intake_lines(ss.session.record)))
    edit_points("Overview", b, "overview")
    st.subheader("Standard of care")
    for field, label in [("first_line", "First line"), ("second_line", "Second line"),
                         ("supportive_care", "Supportive care"), ("when_to_refer", "When to refer")]:
        edit_points(label, b.standard_of_care, field)

    st.subheader("Emerging treatments")
    st.caption("From ClinicalTrials.gov (phase 2-3, recruiting or active). Untick to remove.")
    if not b.emerging_treatments:
        st.write(NO_DATA)
    for t in b.emerging_treatments:
        include_box(f"**{t.name}** ({t.type}) · {t.phase} · {t.status} · {t.sponsor} · "
                    f"[{t.nct_id}](<{t.url}>)", "treatments", t.nct_id + t.name)

    st.subheader("Key players")
    for group, label in [("companies", "Companies"), ("academic_institutions", "Academic"),
                         ("government_ngo", "Government / NGO")]:
        st.markdown(f"**{label}**")
        players = getattr(b.key_players, group)
        if not players:
            st.write(NO_DATA)
        for p in players:
            include_box(f"{p.name}: {p.role}", "players", p.name)

    st.subheader("What this means in the field")
    key = f"{ss.run_id}-field"
    st.text_area("Field relevance", value=b.field_relevance, key=key, label_visibility="collapsed",
                 on_change=lambda: setattr(b, "field_relevance", ss[key]))
    st.info(b.disclaimer, icon=":material/info:")
    with st.expander(f"Sources ({len(b.sources)})"):
        st.markdown("\n".join(f"- **{s.id}** [{s.title}](<{s.url}>) · {s.type}" for s in b.sources))
    approve_block()


# --- Page 4: Patient History ---
def page_history() -> None:
    st.title("Patient History")
    st.caption("Approved records only. Drafts that were not approved are not saved.")
    pid = st.text_input("Search by patient ID", placeholder="e.g. P-1A2B3C")
    records = list_records(pid or None)
    if not records:
        st.info("No approved records yet." if not pid else "No records for that patient ID.")
        return
    options = {r["id"]: f"#{r['id']} · {r['patient_id']} · {r['condition']} · "
                        f"{r['approved_at'].replace('T', ' ')}" for r in records}
    default = list(options).index(ss.approved) if ss.get("approved") in options else 0
    chosen = st.selectbox("Record", list(options), index=default, format_func=options.get)
    rec = get_record(chosen)

    st.subheader(f"{rec['condition']} · {rec['patient_id']}")
    with st.expander("Intake summary"):
        st.markdown("\n".join(f"- {line}" for line in intake_lines(rec["record"])))

    st.markdown("**For the patient**")
    c1, c2 = st.columns(2)
    if rec["sheet"]:
        c1.download_button("Patient sheet (PDF)", sheet_pdf(rec["sheet"]), width="stretch",
                           file_name=f"{rec['patient_id']}_patient_sheet.pdf",
                           mime="application/pdf", icon=":material/print:")
        sms = sheet_sms(rec["sheet"])
        c2.download_button("SMS text (.txt)", sms, width="stretch",
                           file_name=f"{rec['patient_id']}_sms.txt", icon=":material/sms:")
        st.code(sms, language=None, wrap_lines=True)
        st.caption(f"{len(sms)} characters (up to 3 text messages). Copy it with the button "
                   "at the top right of the box.")

    if rec["briefing"]:
        st.markdown("**For the care team**")
        c3, c4 = st.columns(2)
        c3.download_button("Intake + briefing (PDF)", care_team_pdf(rec["record"], rec["briefing"]),
                           file_name=f"{rec['patient_id']}_care_team.pdf", mime="application/pdf",
                           width="stretch", icon=":material/picture_as_pdf:")
        c4.download_button("Intake + briefing (Markdown)",
                           care_team_markdown(rec["record"], rec["briefing"]),
                           file_name=f"{rec['patient_id']}_care_team.md", mime="text/markdown",
                           width="stretch", icon=":material/description:")


# --- Navigation: pages in a slim top bar; the sidebar is hidden ---
PAGES = {
    "home": st.Page(page_home, title="Home", icon=":material/home:", default=True),
    "intake": st.Page(page_intake, title="New Intake", icon=":material/person_add:",
                      url_path="intake"),
    "sheet": st.Page(page_sheet, title="Patient Info Sheet", icon=":material/article:",
                     url_path="patient-sheet"),
    "briefing": st.Page(page_briefing, title="Care Team Briefing", icon=":material/groups:",
                        url_path="care-team-briefing"),
    "history": st.Page(page_history, title="Patient History", icon=":material/history:",
                       url_path="history"),
}
st.navigation(list(PAGES.values()), position="top").run()
