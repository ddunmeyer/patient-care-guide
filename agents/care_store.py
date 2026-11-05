"""Purpose of this program: save and load APPROVED patient records in SQLite.

Only records a caregiver has reviewed and approved are saved; drafts live in
the app's memory and disappear if not approved. Each saved row holds the
intake record (patient_id, no names), the final Patient Info Sheet and the
final Care Team Briefing as JSON, plus when it was approved.

The file is data/care_records.db. It holds patient health data, so it is
listed in .gitignore and .dockerignore and must never be committed.

Try it:  python -m agents.care_store   (lists saved records)
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from agents.care_models import Briefing, PatientSheet
from agents.intake_models import IntakeRecord

DB_FILE = Path("data/care_records.db")


@contextmanager
def _connect():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # rows act like dicts: row["patient_id"]
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS approved_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            condition TEXT,
            approved_at TEXT NOT NULL,
            caregiver_note TEXT,
            intake_json TEXT NOT NULL,
            sheet_json TEXT,
            briefing_json TEXT)""")
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_approved(record: IntakeRecord, condition: str, sheet: PatientSheet | None,
                  briefing: Briefing | None, caregiver_note: str = "") -> int:
    """Save one approved record. Returns its id."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO approved_records (patient_id, condition, approved_at, caregiver_note, "
            "intake_json, sheet_json, briefing_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.patient_id, condition, datetime.now().isoformat(timespec="seconds"),
             caregiver_note, record.model_dump_json(),
             sheet.model_dump_json() if sheet else None,
             briefing.model_dump_json() if briefing else None))
        return cursor.lastrowid


def list_records(patient_id: str | None = None) -> list[dict]:
    """Short summaries, newest first, optionally for one patient."""
    sql = "SELECT id, patient_id, condition, approved_at FROM approved_records"
    args = ()
    if patient_id:
        sql += " WHERE patient_id = ?"
        args = (patient_id.strip().upper(),)
    with _connect() as conn:
        return [dict(row) for row in conn.execute(sql + " ORDER BY id DESC", args)]


def get_record(record_id: int) -> dict | None:
    """One saved record, with the JSON turned back into Pydantic objects."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM approved_records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"], "patient_id": row["patient_id"], "condition": row["condition"],
        "approved_at": row["approved_at"], "caregiver_note": row["caregiver_note"],
        "record": IntakeRecord.model_validate_json(row["intake_json"]),
        "sheet": PatientSheet.model_validate_json(row["sheet_json"]) if row["sheet_json"] else None,
        "briefing": Briefing.model_validate_json(row["briefing_json"]) if row["briefing_json"] else None,
    }


if __name__ == "__main__":
    for r in list_records():
        print(json.dumps(r))
    print("(no approved records yet)" if not list_records() else "")
