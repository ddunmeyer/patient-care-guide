"""Purpose of this program: the Streamlit chat UI for the Lakeshore Health
member assistant.

Sends each question to the coordinator (agents/coordinator.py) and shows the
answer, which agent answered and why, its sources, and any guardrail notices.
If the Prior Auth specialist drafts a request, it shows Approve and Reject
buttons: nothing is submitted until a person clicks Approve (human in the loop).

Run it with:  streamlit run app.py
"""

import json
import uuid  # makes a unique thread_id for each conversation
from datetime import datetime
from pathlib import Path

import streamlit as st

from agents.coordinator import ask
from guardrails.checks import redact_pii

MEMBERS_FILE = Path("data/members.json")
APPROVED_FILE = Path("data/approved_requests.json")  # approved prior auth requests

# What to show in the progress box as each step finishes.
STEP_LABELS = {
    "input_check": "Input check (PII redaction, prompt-injection scan)",
    "router": "Routing the question",
    "benefits": "Benefits specialist answered",
    "prior_auth": "Prior Auth specialist answered",
    "out_of_scope": "Out of scope: no specialist needed",
    "output_check": "Output check (medical advice, sources)",
}
AGENT_NAMES = {"benefits": "Benefits specialist", "prior_auth": "Prior Auth specialist",
               "none": "No specialist"}

st.set_page_config(page_title="Lakeshore Health Member Assistant",
                   page_icon=":material/health_and_safety:")

# --- Session state: Streamlit reruns this whole file on every click, so
# anything that must survive a rerun lives in st.session_state. ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # one id per conversation
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: {"role", "content", "result"}


# --- Helpers ---
def safe_md(text: str) -> str:
    """Escape $ so Streamlit shows "$50" as money, not as a math formula."""
    return text.replace("$", "\\$")


def nice_time(iso: str) -> str:
    """Turn "2026-11-04T21:34:12" into "Nov 04, 2026 at 09:34 PM"."""
    return datetime.fromisoformat(iso).strftime("%b %d, %Y at %I:%M %p")


def load_members() -> dict:
    return json.loads(MEMBERS_FILE.read_text(encoding="utf-8"))


def save_approved(draft: dict) -> None:
    """Add an approved request to data/approved_requests.json.
    In a real system, this is where the request would be sent to the plan."""
    saved = json.loads(APPROVED_FILE.read_text(encoding="utf-8")) if APPROVED_FILE.exists() else []
    saved.append(draft)
    APPROVED_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def render_draft(draft: dict) -> None:
    """Show a prior auth draft as a card with Approve / Reject buttons.
    The draft dict lives in session_state, so changing it here is remembered."""
    with st.container(border=True):
        st.markdown("**:material/assignment: Prior authorization request**")
        st.markdown(
            f"**Member:** {draft['member_name']} ({draft['member_id']})  \n"
            f"**Service:** {draft['service']}  \n"
            f"**Reason:** {draft['reason']}  \n"
            f"**Draft ID:** `{draft['draft_id']}`"
        )

        if draft["status"] == "pending_approval":
            st.badge("Pending approval", icon=":material/schedule:", color="orange")
            st.caption("Not submitted. Review the details, then approve or reject.")
            approve_col, reject_col = st.columns(2)
            # Each button needs a unique key, so use the draft id.
            if approve_col.button("Approve and submit", key=f"approve-{draft['draft_id']}",
                                  type="primary", width="stretch"):
                draft["status"] = "Approved - submitted"
                draft["decided_at"] = datetime.now().isoformat(timespec="seconds")
                draft["note"] = "Approved by a person in the member assistant, then submitted."
                save_approved(draft)
                st.rerun()  # redraw the page with the new status
            if reject_col.button("Reject", key=f"reject-{draft['draft_id']}",
                                 width="stretch"):
                draft["status"] = "Rejected"
                draft["decided_at"] = datetime.now().isoformat(timespec="seconds")
                draft["note"] = "Rejected by a person. Not submitted."
                st.rerun()
        elif draft["status"] == "Approved - submitted":
            st.success(f"Approved - submitted on {nice_time(draft['decided_at'])}",
                       icon=":material/check_circle:")
        else:
            st.error(f"Rejected on {nice_time(draft['decided_at'])}. Nothing was submitted.",
                     icon=":material/cancel:")


def render_notices(result: dict) -> None:
    """Show guardrail and confidence notices for one answer."""
    inp, out = result["input_guardrail"], result["output_guardrail"]

    if result["route"] == "blocked":
        st.error(f"Blocked by input guardrail: {inp['reason']}", icon=":material/block:")
    redacted = [f.replace("redacted ", "") for f in inp["flags"] if f.startswith("redacted")]
    if redacted:
        st.info(f"Personal information was removed before processing: {', '.join(redacted)}.",
                icon=":material/shield_lock:")
    if out and not out["allowed"]:
        st.warning(f"Answer replaced by output guardrail: {out['reason']}",
                   icon=":material/gpp_maybe:")
    elif out and "no sources" in out["flags"]:
        st.warning(out["reason"], icon=":material/help:")
    if result["low_confidence"]:
        st.badge("Low confidence: the best document match was weak, check the sources",
                 icon=":material/warning:", color="orange")


def render_assistant(result: dict) -> None:
    """Show one assistant turn: answer, notices, caption, steps, sources, draft."""
    st.markdown(safe_md(result["answer"]))
    render_notices(result)

    # Caption: who answered and why.
    agent = AGENT_NAMES.get(result["agent_used"], result["agent_used"])
    st.caption(f"**{agent}** · {result['route_reason']}")

    # The steps that ran and the tools that were called, folded away.
    with st.expander("Steps and tools"):
        for step in result["steps"]:
            st.markdown(f"- {STEP_LABELS.get(step, step)}")
        for call in result["tools_called"]:
            st.markdown(f"- Tool: `{call}`")

    # Sources, with duplicate chunks from the same section shown once.
    if result["sources"]:
        unique = {(s["source"], s["location"]): s.get("score") for s in result["sources"]}
        with st.expander(f"Sources ({len(unique)})"):
            for (source, location), score in unique.items():
                score_text = f" · relevance {score:.2f}" if score is not None else ""
                st.markdown(f"- **{source}** · {location}{score_text}")

    if result["pending_draft"]:
        render_draft(result["pending_draft"])


# --- Sidebar ---
with st.sidebar:
    st.header("Member")
    members = load_members()
    member_id = st.selectbox(
        "Member ID",
        options=[None, *members],  # None = no member selected
        format_func=lambda m: "No member selected" if m is None else f"{m} · {members[m]['name']}",
    )
    st.caption("An ID typed in a question overrides this.")

    if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())  # new thread = fresh memory
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Conversation id: `{st.session_state.thread_id[:8]}`")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Input check**: personal data is masked; prompt injection is blocked.\n"
        "2. **Router**: picks Benefits, Prior Auth, or out of scope.\n"
        "3. **Specialist**: answers using plan documents and member records only.\n"
        "4. **Output check**: blocks medical advice and flags unsourced answers.\n"
        "5. **You approve**: prior auth requests are drafts until you approve them."
    )

# --- Main area ---
st.title("Lakeshore Health Member Assistant")
st.caption("Answers about your Lakeshore Health PPO coverage, costs and prior authorization, "
           "with sources.")

if not st.session_state.messages:
    st.info("Try: *What's my copay for a specialist visit?* or "
            "*I need an MRI, does it need prior auth? Please start the request.*",
            icon=":material/lightbulb:")

# Redraw the whole conversation (Streamlit starts from a blank page each rerun).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(safe_md(msg["content"]))
        else:
            render_assistant(msg["result"])

# Handle a new question.
if prompt := st.chat_input("Ask about your coverage, costs or prior authorization"):
    shown = redact_pii(prompt)[0]  # never show or store raw personal data
    with st.chat_message("user"):
        st.markdown(safe_md(shown))

    with st.chat_message("assistant"):
        steps = []
        # st.status is a progress box; each finished step is written into it.
        with st.status("Working on it...", expanded=True) as status:
            def on_step(name: str) -> None:
                steps.append(name)
                status.write(STEP_LABELS.get(name, name))

            result = ask(prompt, thread_id=st.session_state.thread_id,
                         member_id=member_id, on_step=on_step)
            status.update(label="Done", state="complete", expanded=False)

    result["steps"] = steps
    st.session_state.messages.append({"role": "user", "content": shown})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"],
                                      "result": result})
    st.rerun()  # redraw from history so the answer and buttons render in one place
