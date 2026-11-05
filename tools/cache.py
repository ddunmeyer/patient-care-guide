"""Purpose of this program: a 24-hour SQLite cache for the lookup tools.

The medical APIs are slow and some have rate limits, and a caregiver with a
weak connection shouldn't wait for the same search twice in one day. Each
tool wraps its work in @cached("tool_name"): the first call fetches and saves
the result in data/tool_cache.db; calls with the same arguments in the next
24 hours read it back from the file.

Results with an error, or with no results, are not saved, so a failed or
empty lookup is tried again next time.
The cache holds only public search results (condition names, trials,
articles), never patient data.
"""

import functools
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

CACHE_FILE = Path("data/tool_cache.db")
MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


@contextmanager
def _connect():
    """Open the cache file, commit when the block ends, then close it.
    Each call opens its own connection, so tools running in threads are safe."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_FILE, timeout=10)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS cache "
                     "(key TEXT PRIMARY KEY, value TEXT, saved_at REAL)")
        yield conn
        conn.commit()
    finally:
        conn.close()


def cached(tool_name: str):
    """Decorator: cache a tool's result by tool name + arguments for 24 hours."""
    def wrap(func):
        @functools.wraps(func)
        def inner(*args):
            # "pubmed_search|malaria": lower case, so "Malaria" hits the same entry.
            key = tool_name + "|" + "|".join(str(a).strip().lower() for a in args)
            with _connect() as conn:
                row = conn.execute("SELECT value, saved_at FROM cache WHERE key = ?",
                                   (key,)).fetchone()
            if row and time.time() - row[1] < MAX_AGE_SECONDS:
                return {**json.loads(row[0]), "cached": True}

            result = func(*args)
            # Save only good, non-empty results: an empty list may just mean
            # the API was busy, so it is worth asking again next time.
            if not result.get("error") and result.get("results"):
                with _connect() as conn:
                    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                                 (key, json.dumps(result), time.time()))
            return {**result, "cached": False}
        return inner
    return wrap
