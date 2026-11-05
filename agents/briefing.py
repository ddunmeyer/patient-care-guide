"""Purpose of this program: the BriefingAgent, which writes a structured
Care Team Briefing about a condition: current standard of care, emerging
treatments in development, the key companies and institutions, and what it
all means in a low-resource setting.

Steps in make_briefing():
  1. gpt-4o-mini reads the numbered evidence (agents/care_evidence.py) and
     fills the Briefing model. Every standard-of-care and overview point
     carries evidence ids; emerging treatments carry the trial's NCT id;
     organisations must be named exactly as in the evidence.
  2. Python sets generated_at, and the grounding guardrail
     (guardrails/grounding.py) checks the draft and fills in each trial's
     phase, status, sponsor and link from ClinicalTrials.gov, plus the
     source list.

No patient data is sent: the briefing is about the condition only.

Try it:  python -m agents.briefing tuberculosis
"""

import sys
from datetime import datetime

from agents.care_evidence import EvidenceCatalog, build_catalog
from agents.care_models import Briefing
from agents.specialists import llm  # the kit's gpt-4o-mini, temperature 0
from guardrails.grounding import ground_briefing

BRIEFING_PROMPT = """You prepare a briefing on {condition} for a care team
(nurses, community health workers, medics) working where a doctor is often
not available.

Rules:
- Use ONLY the numbered evidence below. Never use memory. After each point,
  list the evidence ids it comes from in "sources", e.g. ["P1", "W2"]. If
  the evidence does not cover a section, leave that section empty.
- overview: 2-3 short sentences on what the condition is and why it matters.
- standard_of_care: first_line, second_line, supportive_care and
  when_to_refer, 2-4 short points each. Name drug classes or drugs if the evidence
  does, but NEVER give doses, amounts, frequencies or durations, and never
  say what a specific patient should take.
- emerging_treatments: up to 8 interventions being tested in the trials in
  the evidence (ids that start with NCT). Copy the NCT id exactly. Use the
  intervention name from that trial. Skip placebo and comparator arms.
- key_players: organisations named in the evidence (trial lead sponsors and
  collaborators, FDA label labelers, organisations named in articles or web
  pages). Copy each name exactly as written. Sort them into companies,
  academic_institutions (universities, hospitals, research institutes) and
  government_ngo (ministries, public agencies, WHO, charities, foundations).
  The role says what they do here, e.g. "lead sponsor of NCT01234567".
- field_relevance: 2-3 lines on what this means in a clinic with few doctors,
  little equipment and limited medicines.
- Do not diagnose anyone."""


def make_briefing(condition: str, catalog: EvidenceCatalog) -> tuple[Briefing, list[str]]:
    """Write and ground the briefing. Returns (briefing, guardrail notes)."""
    messages = [("system", BRIEFING_PROMPT.format(condition=condition)),
                ("human", f"EVIDENCE:\n{catalog.prompt_text()}")]
    writer = llm.with_structured_output(Briefing, method="function_calling")
    draft = writer.invoke(messages, config={"run_name": "briefing_agent"})

    # The model sometimes stops early and leaves sections out. If there are
    # trials in the evidence but the draft has no treatments or organisations,
    # ask once more.
    players = draft.key_players
    empty = not (players.companies or players.academic_institutions or players.government_ngo)
    if catalog.trials and (not draft.emerging_treatments or empty):
        draft = writer.invoke(messages + [("human", "Your answer left out emerging_treatments "
                                           "or key_players. Fill in every section.")],
                              config={"run_name": "briefing_agent_retry"})
    draft.condition = condition
    draft.generated_at = datetime.now().isoformat(timespec="minutes")
    return ground_briefing(draft, catalog)


if __name__ == "__main__":
    from tools.evidence import gather_evidence
    condition = " ".join(sys.argv[1:]) or "tuberculosis"
    briefing, report = make_briefing(condition, build_catalog(gather_evidence(condition)))
    print(briefing.model_dump_json(indent=2))
    print("\nGuardrail:", report or "nothing removed")
