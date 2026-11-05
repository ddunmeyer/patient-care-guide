"""Purpose of this program: measure answer quality.

Runs each case in evals/eval_set.json through coordinator.ask() with a fresh
thread (so no memory carries over between cases), checks the result, prints
a results table, the overall pass rate and the median and p95 latency, and
sends each case's pass/fail to Langfuse as a score on that case's trace.

Four checks per case; a case passes only if all four pass:
  route      the coordinator picked the expected route
  keywords   every must_contain keyword is in the answer (case-insensitive)
  grounding  one of: the expected source was cited in the answer as
             [file.md] (expected_source; members.json only needs to have
             been looked up), the answer is a refusal (should_refuse), or a
             prior auth draft was created (expect_draft)
  guardrail  the expected input flag was raised (or, if none is expected,
             the question was not blocked); no SSN appears in the text sent
             to the model or in the answer; and a grounded answer is not
             flagged or replaced by the output check

Run it from the project folder:
    python -m evals.run_eval
"""

import json  # the eval set is a JSON file
import math  # ceil, for the p95 rank
import re  # to spot an unredacted SSN
import statistics  # median
import time  # to time each case
import uuid  # a fresh thread_id per case
from pathlib import Path

from agents.coordinator import ask
from observability import tracing  # Langfuse scores; does nothing without keys

EVAL_FILE = Path("evals/eval_set.json")

# Phrases that show the assistant declined or said it doesn't know. The eval
# keeps its own list instead of reusing the guardrail's, so it checks the
# behaviour independently.
REFUSAL_PHRASES = [
    "can only help", "can't help", "cannot help", "can't give medical advice",
    "don't have", "do not have", "couldn't find", "could not find",
    "unable to", "not able to", "not in the plan documents",
    "doesn't mention", "does not mention", "not mentioned", "no information",
]
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # same shape as the guardrail's


def is_refusal(answer: str) -> bool:
    """True if the answer contains any refusal phrase."""
    text = answer.lower().replace("’", "'")  # treat curly ' as plain '
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def check_case(case: dict, result: dict) -> dict:
    """Run the four checks. Returns {check name: (passed, note)}."""
    answer = result["answer"]
    checks = {}

    # 1. Route
    got = result["route"]
    checks["route"] = (got == case["expected_route"],
                       f"expected {case['expected_route']}, got {got}")

    # 2. Keywords: list any that are missing.
    missing = [k for k in case["must_contain"] if k.lower() not in answer.lower()]
    checks["keywords"] = (not missing, f"missing {missing}")

    # 3. Grounding: which kind depends on the fields the case has.
    retrieved = sorted({s["source"] for s in result["sources"]})
    if case.get("expected_source"):
        source = case["expected_source"]
        if source == "members.json":
            # Member data is never cited in brackets, so it is enough that the
            # member record was looked up (it is then in the sources list).
            ok = source in retrieved
        else:
            # A plan document must be cited in the answer itself, like
            # [summary_of_benefits.md]; merely being retrieved is not enough.
            ok = f"[{source}]" in answer
        checks["grounding"] = (ok, f"expected [{source}] cited, retrieved {retrieved}")
    elif case.get("should_refuse"):
        checks["grounding"] = (is_refusal(answer), "answer is not a refusal")
    elif case.get("expect_draft"):
        draft = result.get("pending_draft")
        ok = bool(draft) and draft.get("status") == "pending_approval"
        checks["grounding"] = (ok, "no pending_approval draft was created")

    # 4. Guardrail
    flags = result["input_guardrail"]["flags"]
    if case.get("expected_guardrail"):
        flag_ok = case["expected_guardrail"] in flags
    else:
        flag_ok = result["input_guardrail"]["allowed"]  # a normal question must not be blocked
    leaked = bool(SSN_RE.search(result["question"]) or SSN_RE.search(answer))
    # A grounded answer (a cited fact or a created draft) must pass the output
    # check cleanly: not replaced, and not flagged as having no sources.
    out = result.get("output_guardrail") or {}
    out_flags = out.get("flags", [])
    grounded = case.get("expected_source") or case.get("expect_draft")
    out_ok = not (grounded and out_flags)
    checks["guardrail"] = (flag_ok and not leaked and out_ok,
                           f"input flags {flags}, output flags {out_flags}"
                           + (", SSN leaked" if leaked else ""))
    return checks


def p95(values: list[float]) -> float:
    """95th percentile, nearest-rank method: sort, then take the value at
    rank ceil(0.95 * n). With 12 cases that is the 12th (slowest) value."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> None:
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["items"]
    rows, latencies = [], []

    for case in cases:
        # A fresh thread per case, so memory from one case can't help another.
        thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
        start = time.perf_counter()
        result = ask(case["question"], thread_id=thread_id, member_id=case["member_id"])
        seconds = time.perf_counter() - start
        latencies.append(seconds)

        checks = check_case(case, result)
        passed = all(ok for ok, _ in checks.values())
        # Notes for the checks that failed, e.g. "keywords: missing ['$50']".
        failures = [f"{name}: {note}" for name, (ok, note) in checks.items() if not ok]

        # Send pass/fail to Langfuse, with the failure notes as the comment.
        tracing.score(result.get("trace_id"), "eval_pass", passed, "; ".join(failures))

        rows.append((case, checks, passed, seconds, failures, result["answer"]))
        print(f"  ran {case['id']} ({seconds:.1f}s)")  # progress while it runs

    # --- Results table ---
    names = ["route", "keywords", "grounding", "guardrail"]
    print()
    print(f"{'case':<18}" + "".join(f"{n:<11}" for n in names) + f"{'secs':>6}  result")
    print("-" * 76)
    for case, checks, passed, seconds, _, _ in rows:
        marks = "".join(f"{'ok' if checks[n][0] else 'FAIL':<11}" for n in names)
        print(f"{case['id']:<18}{marks}{seconds:>6.1f}  {'PASS' if passed else 'FAIL'}")
    print("-" * 76)

    passed_count = sum(1 for row in rows if row[2])
    print(f"Pass rate: {passed_count}/{len(rows)} ({passed_count / len(rows):.0%})")
    print(f"Latency:   median {statistics.median(latencies):.1f}s, p95 {p95(latencies):.1f}s")

    # --- Why each failed case failed ---
    for case, _, passed, _, failures, answer in rows:
        if not passed:
            print(f"\nFAIL {case['id']}: {case['question']}")
            for note in failures:
                print(f"  - {note}")
            print(f"  answer: {answer[:300]}")

    tracing.flush()  # send the scores before the script exits


if __name__ == "__main__":
    main()
