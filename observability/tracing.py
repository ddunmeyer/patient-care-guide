"""Purpose of this program: Langfuse tracing helpers, written for langfuse 5.x.

This is the only file that talks to Langfuse. Everything else calls these
small helpers, so the rest of the app never imports langfuse directly.

How one question becomes one Langfuse trace:

    with trace_run(question, session_id, user_id) as run:   # opens the trace
        graph config callbacks = [get_handler()]             # graph steps nest inside
        ... run the graph ...
        finish_run(run, route, answer)                       # adds the route tag

  - trace_run(): opens a root span called "ask" and gives the trace its
    session_id and user_id.
  - get_handler(): a LangChain CallbackHandler. Passed into the graph config,
    it records every node, model call, tool call, token count, cost and
    latency as child steps of that root span.
  - finish_run(): saves the answer on the root span and tags the trace with
    the route. This happens at the end, because the route is only known after
    the router has run.
  - score(): attaches a pass/fail score to a trace (used by the evals).
  - flush(): sends any traces still waiting in the queue (call before exit).

If LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing, tracing is turned
off: trace_run() yields None, get_handler() returns None and the other
helpers do nothing, so the app still runs normally.

Privacy: callers must pass only the redacted text. ask() runs the input
guardrail before it calls anything here, so raw PII never reaches Langfuse.
"""

import os  # to read the keys from the environment
from contextlib import contextmanager  # lets trace_run() be used with "with"

from dotenv import load_dotenv  # reads the Langfuse keys from the .env file

load_dotenv()  # make the keys available before we check them

# Tracing is on only when both keys are present. LANGFUSE_HOST (or
# LANGFUSE_BASE_URL) is read by the langfuse library itself.
ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    # Import only when needed, so a missing key never triggers langfuse setup.
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler


@contextmanager
def trace_run(question: str, session_id: str, user_id: str | None):
    """Open one Langfuse trace for one question. Yields the root span, or
    None if tracing is off. question must already be redacted."""
    if not ENABLED:
        yield None  # nothing to trace; the "with" body still runs
        return
    client = get_client()  # the shared Langfuse client, set up from the keys
    # The root span: everything recorded inside this block becomes its child.
    with client.start_as_current_observation(name="ask", input=question) as span:
        # propagate_attributes puts these trace fields on the root span and on
        # every step created inside it, so Langfuse can filter by them.
        with propagate_attributes(session_id=session_id,  # one conversation
                                  user_id=user_id,  # the member id (may be None)
                                  trace_name="ask"):
            yield span


def get_handler():
    """Return a new CallbackHandler, or None if tracing is off.

    Created inside trace_run(), it attaches the graph's steps to the open
    root span. A new handler per run keeps runs from mixing when several
    people use the Streamlit app at once.
    """
    return CallbackHandler() if ENABLED else None


def finish_run(span, route: str, answer: str) -> None:
    """Save the answer and tag the trace with the route ("route:benefits").
    Must be called inside the trace_run() block. Does nothing if span is None."""
    if span is None:
        return
    span.update(output=answer)  # shown as the trace's output in Langfuse
    # Entering propagate_attributes sets the tag on the span that is active
    # right now, which is our root span, so the whole trace gets the tag.
    # There is nothing else to run inside the block, so its body is "pass".
    with propagate_attributes(tags=[f"route:{route}"]):
        pass


def score(trace_id: str | None, name: str, passed: bool, comment: str = "") -> None:
    """Attach a pass (1) or fail (0) score to a trace. Does nothing if
    tracing is off or the run has no trace id."""
    if not ENABLED or not trace_id:
        return
    get_client().create_score(
        trace_id=trace_id,
        name=name,
        value=1 if passed else 0,
        data_type="BOOLEAN",  # Langfuse shows it as true/false
        comment=comment,
    )


def flush() -> None:
    """Send everything still queued. Short scripts must call this before they
    exit, because Langfuse sends traces in the background."""
    if ENABLED:
        get_client().flush()
