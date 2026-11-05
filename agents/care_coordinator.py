"""Purpose of this program: the coordinator for the Patient Intake &
Condition Guide (a LangGraph graph).

The intake itself is interactive (the caregiver answers one question at a
time in the app, agents/intake_agent.py). When it is done, run_care_flow()
runs this graph:

    START -> urgent_check --(urgent sign)--> END   (emergency banner, no AI)
                          --(ok)----------> lookup -> patient_sheet --> END
                                                   -> briefing      --> END

  - urgent_check: the rule-based check in triage/urgent_signs.py. If it
    finds a danger sign the graph stops here; no model is ever called, so
    the AI can't override it.
  - lookup: runs the five lookup tools once (tools/evidence.py). Both agents
    then read the same evidence, so the sheet and the briefing agree.
  - patient_sheet and briefing run at the same time (two edges out of
    lookup), each followed by the grounding guardrail.

Nothing is saved here. The caregiver reviews both drafts in the app, can
edit them, and only an approved record is saved (agents/care_store.py).

Tracing: one Langfuse trace per run, via the kit's observability/tracing.py
(session = patient_id, tag route:care_flow or route:urgent_stop). Only the
patient_id and the condition are sent, never names.

Try it:  python -m agents.care_coordinator
"""

import operator
import time
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.briefing import make_briefing
from agents.care_evidence import build_catalog
from agents.intake_models import IntakeRecord
from agents.patient_education import make_patient_sheet
from observability import tracing
from tools.evidence import gather_evidence
from triage.urgent_signs import UrgentResult, check_urgent


class CareState(TypedDict, total=False):
    record: IntakeRecord
    answers: dict  # masked intake answers, for the urgent check
    condition: str
    urgent: UrgentResult
    evidence: dict  # raw tool results
    sheet: object  # PatientSheet
    briefing: object  # Briefing
    sheet_report: list[str]
    briefing_report: list[str]
    # Both agents run in parallel and may add tool errors at the same time;
    # operator.add joins their lists instead of one overwriting the other.
    notes: Annotated[list[str], operator.add]


# --- Nodes ---
def urgent_check(state: CareState) -> CareState:
    return {"urgent": check_urgent(state["record"], state.get("answers"))}


def lookup(state: CareState) -> CareState:
    evidence = gather_evidence(state["condition"])
    errors = [r["error"] for r in evidence.values() if r.get("error")]
    return {"evidence": evidence, "notes": errors}


def patient_sheet(state: CareState) -> CareState:
    catalog = build_catalog(state["evidence"])
    sheet, report = make_patient_sheet(state["condition"], state["record"], catalog)
    return {"sheet": sheet, "sheet_report": report}


def briefing(state: CareState) -> CareState:
    catalog = build_catalog(state["evidence"])
    result, report = make_briefing(state["condition"], catalog)
    return {"briefing": result, "briefing_report": report}


def after_urgent_check(state: CareState) -> str:
    """Stop at once on any urgent sign. This is a plain if, not a model call."""
    return END if state["urgent"].urgent else "lookup"


graph = StateGraph(CareState)
graph.add_node("urgent_check", urgent_check)
graph.add_node("lookup", lookup)
graph.add_node("patient_sheet", patient_sheet)
graph.add_node("briefing", briefing)
graph.add_edge(START, "urgent_check")
graph.add_conditional_edges("urgent_check", after_urgent_check, ["lookup", END])
graph.add_edge("lookup", "patient_sheet")  # two edges out of lookup:
graph.add_edge("lookup", "briefing")  # both agents run in parallel
graph.add_edge("patient_sheet", END)
graph.add_edge("briefing", END)
care_app = graph.compile()


def run_care_flow(record: IntakeRecord, answers: dict | None = None,
                  condition: str | None = None,
                  on_step: Callable[[str], None] | None = None) -> dict:
    """Run urgent check -> lookup -> sheet + briefing for a finished intake.
    condition: typed by the caregiver; defaults to what the patient reported.
    Returns a dict with status "urgent", "no_condition" or "done"."""
    report = on_step or (lambda step: None)
    condition = (condition or record.reported_condition or "").strip()
    start = time.perf_counter()

    trace_input = f"patient {record.patient_id} | condition: {condition or '(none)'}"
    with tracing.trace_run(trace_input, session_id=record.patient_id, user_id=None) as span:
        handler = tracing.get_handler()
        config = {"callbacks": [handler]} if handler else {}

        # The urgent check always runs, even if there is no condition yet.
        state: dict = {"record": record, "answers": answers or {}, "condition": condition}
        if not condition:
            state["urgent"] = check_urgent(record, answers)
            report("urgent_check")
        else:
            for update in care_app.stream(state, config, stream_mode="updates"):
                for node, values in update.items():
                    state.update(values or {})
                    report(node)

        if state["urgent"].urgent:
            status = "urgent"
        elif not condition:
            status = "no_condition"
        else:
            status = "done"
        tracing.finish_run(span, route="urgent_stop" if status == "urgent" else "care_flow",
                           answer=f"status: {status}")

    return {
        "status": status,
        "condition": condition,
        "urgent": state["urgent"],
        "sheet": state.get("sheet"),
        "briefing": state.get("briefing"),
        "sheet_report": state.get("sheet_report", []),
        "briefing_report": state.get("briefing_report", []),
        "notes": state.get("notes", []),
        "seconds": round(time.perf_counter() - start, 1),
        "trace_id": span.trace_id if span else None,
    }


if __name__ == "__main__":
    from agents.intake_models import Symptom
    calm = IntakeRecord(patient_id="P-TEST01", age=52, sex="male",
                        chief_complaint="headaches", reported_condition="hypertension",
                        current_meds=["amlodipine"])
    urgent = IntakeRecord(patient_id="P-TEST02", age=60, sex="male",
                          chief_complaint="chest pain", reported_condition="hypertension",
                          symptoms=[Symptom(name="sweating", duration="1 hour")])
    for rec in (urgent, calm):
        out = run_care_flow(rec, on_step=lambda s: print("  step:", s))
        print(rec.patient_id, out["status"], f"{out['seconds']}s",
              [s.name for s in out["urgent"].signs], out["notes"])
        if out["status"] == "done":
            print("  sheet sections ok, sources:", len(out["sheet"].sources),
                  "| briefing trials:", len(out["briefing"].emerging_treatments),
                  "| reports:", out["sheet_report"], out["briefing_report"])
    tracing.flush()
