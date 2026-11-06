"""Purpose of this program: export approved outputs to hand out or send.

  - sheet_pdf(sheet):      Patient Info Sheet as a PDF to print and give
                           to the patient (large text, simple layout).
  - sheet_sms(sheet):      a short text message version (at most 3 SMS,
                           480 characters), always with the disclaimer.
  - care_team_markdown():  intake summary + briefing as Markdown.
  - care_team_pdf():       the same as a PDF for the care team.

PDF fonts: a Unicode font is needed for languages with accents or other
scripts. We use Arial (Windows) or DejaVu Sans (Linux/Docker) if found;
otherwise the built-in Helvetica, with characters it can't show replaced.

Try it:  python -m agents.care_export   (writes sample files to the scratch folder you pass)
"""

import re
import sys
from pathlib import Path

from fpdf import FPDF

from agents.care_models import NO_DATA, Briefing, PatientSheet, Point
from agents.intake_models import IntakeRecord

SMS_LIMIT = 480  # 3 SMS messages of 160 characters

FONT_CANDIDATES = [  # (regular, bold)
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]

SHEET_HEADINGS = [  # (field, heading shown to the patient)
    ("what_it_is", "What it is"),
    ("common_signs", "Common signs"),
    ("how_it_is_usually_treated", "How it is usually treated"),
    ("what_you_can_do_at_home", "What you can do at home"),
    ("warning_signs_get_help_now", "Warning signs: get help now"),
]
SOC_HEADINGS = [("first_line", "First line"), ("second_line", "Second line"),
                ("supportive_care", "Supportive care"), ("when_to_refer", "When to refer")]


# --- A small PDF helper ---
class SimplePDF(FPDF):
    def __init__(self, title: str, base_size: int = 12):
        super().__init__()
        self.base = base_size
        self.unicode = False
        for regular, bold in FONT_CANDIDATES:
            if Path(regular).exists() and Path(bold).exists():
                self.add_font("Body", "", regular)
                self.add_font("Body", "B", bold)
                self.unicode = True
                break
        self.family_name = "Body" if self.unicode else "Helvetica"
        self.set_title(title)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def _t(self, text: str) -> str:
        """Without a Unicode font, swap characters Helvetica can't show."""
        if self.unicode:
            return text
        text = (text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
                .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-"))
        return text.encode("latin-1", "replace").decode("latin-1")

    def h1(self, text: str) -> None:
        self.set_font(self.family_name, "B", self.base + 8)
        self.multi_cell(0, self.base * 0.7, self._t(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_font(self.family_name, "B", self.base + 3)
        self.multi_cell(0, self.base * 0.6, self._t(text), new_x="LMARGIN", new_y="NEXT")

    def para(self, text: str, bold: bool = False, size: int | None = None) -> None:
        self.set_font(self.family_name, "B" if bold else "", size or self.base)
        self.multi_cell(0, (size or self.base) * 0.55, self._t(text), new_x="LMARGIN",
                        new_y="NEXT")

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.para(f"- {item}")

    def box(self, text: str) -> None:
        """A bordered note, for the disclaimer."""
        self.ln(3)
        self.set_font(self.family_name, "B", self.base)
        self.multi_cell(0, self.base * 0.6, self._t(text), border=1, padding=2,
                        new_x="LMARGIN", new_y="NEXT")

    def bytes(self) -> bytes:
        return bytes(self.output())


def _cite(p: Point) -> str:
    return f"{p.text} [{', '.join(p.sources)}]" if p.sources else p.text


def _points(points: list[Point], with_ids: bool) -> list[str]:
    return [_cite(p) if with_ids else p.text for p in points] or [NO_DATA]


# --- Patient Info Sheet ---
def sheet_pdf(sheet: PatientSheet) -> bytes:
    """Large-print sheet for the patient. Source ids are left out of the text
    to keep it easy to read; the sources are listed at the end."""
    pdf = SimplePDF(f"About {sheet.condition}", base_size=14)
    pdf.h1(f"About {sheet.condition}")
    if sheet.personal_notes:
        pdf.h2("Important for you")
        pdf.bullets(sheet.personal_notes)
    for field, heading in SHEET_HEADINGS:
        pdf.h2(heading)
        pdf.bullets(_points(getattr(sheet, field), with_ids=False))
    pdf.h2("Questions to ask your health worker")
    pdf.bullets(sheet.questions_to_ask_your_health_worker or [NO_DATA])
    pdf.h2("Research studies you can ask about")
    pdf.bullets([f"{s.plain_summary} ({s.nct_id}; {', '.join(s.countries) or 'country not listed'})"
                 for s in sheet.research_studies_you_can_ask_about] or [NO_DATA])
    pdf.box(sheet.disclaimer)
    pdf.h2("Sources")
    pdf.set_font(pdf.family_name, "", 9)
    for s in sheet.sources:
        pdf.para(f"[{s.id}] {s.title} - {s.url}", size=9)
    return pdf.bytes()


def sheet_sms(sheet: PatientSheet) -> str:
    """A short text message. The title and disclaimer always fit; then as
    many key lines as fit within 480 characters."""
    first = lambda pts: pts[0].text if pts and pts[0].text != NO_DATA else ""
    head = f"{sheet.condition.upper()}: "
    tail = f" {sheet.disclaimer}"
    optional = [first(sheet.what_it_is)]
    # "Seek help if you have a high fever" -> "a high fever" (the SMS already says GET HELP NOW).
    lead_in = re.compile(r"^(seek|get|go for|go to a clinic|go to the clinic|get medical)\s*"
                         r"(help|care|medical help)?\s*(now|right away)?\s*if\s+(you\s+(have|feel|are)\s+)?",
                         re.IGNORECASE)
    warnings = [lead_in.sub("", p.text).rstrip(".") for p in sheet.warning_signs_get_help_now
                if p.text != NO_DATA]
    if warnings:
        optional.append("GET HELP NOW if: " + "; ".join(warnings[:3]) + ".")
    optional += sheet.personal_notes[:2]
    if sheet.sources:
        optional.append(f"More: {sheet.sources[0].url}")

    body = ""
    for line in filter(None, optional):
        if len(head + body + line + tail) + 1 <= SMS_LIMIT:
            body += line + " "
    return (head + body.strip() + tail).strip()


# --- Care team: intake summary + briefing ---
def intake_lines(r: IntakeRecord) -> list[str]:
    """The intake as short 'Label: value' lines (patient_id only, no name)."""
    v = r.vitals
    vitals = ", ".join(x for x in [
        f"temp {v.temp_c} C" if v.temp_c is not None else "",
        f"pulse {v.pulse}" if v.pulse is not None else "",
        f"BP {v.bp_systolic}/{v.bp_diastolic}" if v.bp_systolic is not None else "",
        f"breathing {v.breathing_rate}/min" if v.breathing_rate is not None else ""] if x)
    rows = [
        ("Patient ID", r.patient_id), ("Age", r.age), ("Sex", r.sex),
        ("Pregnant", {True: "yes / possibly", False: "no"}.get(r.pregnant)),
        ("Weight (lb)", round(r.weight_kg * 2.205) if r.weight_kg else None), ("Preferred language", r.preferred_language),
        ("Main problem", r.chief_complaint),
        ("Symptoms", "; ".join(f"{s.name}" + (f" ({s.duration})" if s.duration else "")
                               for s in r.symptoms)),
        ("Condition reported by patient", r.reported_condition),
        ("Known conditions", ", ".join(r.known_conditions)),
        ("Current medicines", ", ".join(r.current_meds)),
        ("Allergies", ", ".join(r.allergies)),
        ("Past surgeries / hospital stays", ", ".join(r.past_surgeries_hospital_stays)),
        ("Family history", ", ".join(r.family_history)),
        ("Smoking", r.lifestyle.smoking), ("Alcohol", r.lifestyle.alcohol),
        ("Vitals", vitals), ("Caregiver notes", r.caregiver_notes),
    ]
    return [f"{label}: {value}" for label, value in rows if value not in (None, "", [])]


def care_team_markdown(record: IntakeRecord, b: Briefing) -> str:
    md = [f"# Care Team Briefing: {b.condition}", f"_Generated {b.generated_at}_", "",
          "## Intake summary", *[f"- {line}" for line in intake_lines(record)], "",
          "## Overview", *[f"- {_cite(p)}" for p in b.overview], "", "## Standard of care"]
    for field, heading in SOC_HEADINGS:
        md += [f"**{heading}**", *[f"- {x}" for x in _points(getattr(b.standard_of_care, field),
                                                              with_ids=True)], ""]
    md += ["## Emerging treatments"]
    if b.emerging_treatments:
        md += ["| Treatment | Type | Phase | Status | Sponsor | Trial |", "|---|---|---|---|---|---|"]
        md += [f"| {t.name} | {t.type} | {t.phase} | {t.status} | {t.sponsor} | "
               f"[{t.nct_id}]({t.url}) |" for t in b.emerging_treatments]
    else:
        md += [NO_DATA]
    md += ["", "## Key players"]
    for group, label in [("companies", "Companies"), ("academic_institutions", "Academic"),
                         ("government_ngo", "Government / NGO")]:
        players = getattr(b.key_players, group)
        md += [f"**{label}**", *([f"- {p.name}: {p.role}" for p in players] or [f"- {NO_DATA}"]),
               ""]
    md += ["## What this means in the field", b.field_relevance, "", "## Sources",
           *[f"- [{s.id}] {s.title} ({s.type}) {s.url}" for s in b.sources], "",
           f"> {b.disclaimer}"]
    return "\n".join(md)


def care_team_pdf(record: IntakeRecord, b: Briefing) -> bytes:
    """The Markdown above, laid out as a PDF (headings, bullets, plain text)."""
    pdf = SimplePDF(f"Care Team Briefing: {b.condition}", base_size=10)
    for line in care_team_markdown(record, b).splitlines():
        if line.startswith("# "):
            pdf.h1(line[2:])
        elif line.startswith("## "):
            pdf.h2(line[3:])
        elif line.startswith("|---"):
            continue
        elif line.startswith("| Treatment"):
            pdf.para("Treatment | Type | Phase | Status | Sponsor | Trial", bold=True)
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells[-1] = re.sub(r"\[(NCT\d+)\]\((.+?)\)", r"\1 \2", cells[-1])
            pdf.para("- " + " | ".join(cells))
        elif line.startswith("**"):
            pdf.para(line.strip("*"), bold=True)
        elif line.startswith("> "):
            pdf.box(line[2:])
        elif line.strip():
            pdf.para(line.replace("_", ""))
    return pdf.bytes()


if __name__ == "__main__":
    from agents.care_models import KeyPlayers, SourceRef, StandardOfCare
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    src = [SourceRef(id="M1", title="MedlinePlus: Malaria", url="https://medlineplus.gov/malaria.html",
                     type="MedlinePlus")]
    sheet = PatientSheet(condition="Malaria", language="French",
                         what_it_is=[Point(text="Le paludisme vient des moustiques.", sources=["M1"])],
                         common_signs=[Point(text="Fièvre et frissons.", sources=["M1"])],
                         how_it_is_usually_treated=[], what_you_can_do_at_home=[],
                         warning_signs_get_help_now=[Point(text="Très fatigué ou confus.",
                                                           sources=["M1"])],
                         questions_to_ask_your_health_worker=["Quel type de paludisme ?"],
                         research_studies_you_can_ask_about=[],
                         personal_notes=["Dites que vous êtes enceinte."], sources=src)
    rec = IntakeRecord(patient_id="P-DEMO1", age=29, sex="female", reported_condition="malaria")
    b = Briefing(condition="Malaria", generated_at="2026-11-05T12:00",
                 overview=[Point(text="Parasite disease.", sources=["M1"])],
                 standard_of_care=StandardOfCare(first_line=[], second_line=[], supportive_care=[],
                                                 when_to_refer=[]),
                 emerging_treatments=[], key_players=KeyPlayers(companies=[],
                 academic_institutions=[], government_ngo=[]), field_relevance="Use rapid tests.",
                 sources=src)
    (out / "sheet.pdf").write_bytes(sheet_pdf(sheet))
    (out / "care_team.pdf").write_bytes(care_team_pdf(rec, b))
    (out / "care_team.md").write_text(care_team_markdown(rec, b), encoding="utf-8")
    print(sheet_sms(sheet), f"\n({len(sheet_sms(sheet))} chars) - files written to {out}")
