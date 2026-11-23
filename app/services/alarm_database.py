
from pathlib import Path
from contextlib import contextmanager
import json
import sqlite3


# =========================================================
# DATABASE LOCATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_PATH = PROJECT_ROOT / "data" / "alarms.db"


# =========================================================
# DATABASE CONNECTION
# =========================================================


@contextmanager
def get_connection(db_path=None):
    """
    Open an SQLite connection and always close it
    when the operation is finished.
    """

    path = Path(db_path) if db_path else DATABASE_PATH

    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:
        yield connection

    finally:
        connection.close()



# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_database(db_path=None):
    """
    Create the alarms table if it does not already exist.
    """

    with get_connection(db_path) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alarms (

                alarm_id TEXT PRIMARY KEY,

                detected_at TEXT,

                scenario TEXT,

                fault_number INTEGER,

                simulation_run INTEGER,

                sample INTEGER,

                detecting_models TEXT,

                models_flagged INTEGER,

                alarm_type TEXT,

                severity TEXT,

                priority INTEGER,

                scores TEXT,

                status TEXT DEFAULT 'Active',

                acknowledged_at TEXT,

                acknowledged_by TEXT,

                operator_notes TEXT

            )
            """
        )

        connection.commit()


# =========================================================
# SAVE AUTOMATICALLY GENERATED ALARMS
# =========================================================

def save_alarms(alarms, db_path=None):
    """
    Save new alarms into SQLite.

    alarm_id is the primary key, so an alarm cannot
    accidentally be stored twice.

    Returns the number of newly inserted alarms.
    """

    if not alarms:
        return 0

    initialize_database(db_path)

    inserted_count = 0

    with get_connection(db_path) as connection:

        for alarm in alarms:

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO alarms (

                    alarm_id,
                    detected_at,
                    scenario,
                    fault_number,
                    simulation_run,
                    sample,
                    detecting_models,
                    models_flagged,
                    alarm_type,
                    severity,
                    priority,
                    scores,
                    status,
                    acknowledged_at,
                    acknowledged_by,
                    operator_notes

                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alarm["alarm_id"],
                    alarm.get("detected_at"),
                    alarm.get("scenario"),
                    alarm.get("fault_number"),
                    alarm.get("simulation_run"),
                    alarm.get("sample"),
                    json.dumps(alarm.get("detecting_models", [])),
                    alarm.get("models_flagged", 0),
                    alarm.get("alarm_type"),
                    alarm.get("severity"),
                    alarm.get("priority"),
                    json.dumps(alarm.get("scores", {})),
                    alarm.get("status", "Active"),
                    alarm.get("acknowledged_at"),
                    alarm.get("acknowledged_by"),
                    alarm.get("operator_notes", "")
                )
            )

            inserted_count += cursor.rowcount

        connection.commit()

    return inserted_count


# =========================================================
# RETRIEVE ALARM HISTORY
# =========================================================

def get_all_alarms(db_path=None):
    """
    Retrieve all saved alarms.

    Newest alarms appear first.
    JSON fields are converted back into Python objects.
    """

    initialize_database(db_path)

    with get_connection(db_path) as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM alarms
            ORDER BY detected_at DESC, rowid DESC
            """
        ).fetchall()

    alarms = []

    for row in rows:

        alarm = dict(row)

        alarm["detecting_models"] = json.loads(
            alarm["detecting_models"] or "[]"
        )

        alarm["scores"] = json.loads(
            alarm["scores"] or "{}"
        )

        alarms.append(alarm)

    return alarms


# =========================================================
# RETRIEVE KNOWN ALARM IDS
# =========================================================

def get_known_alarm_ids(db_path=None):
    """
    Retrieve alarm IDs already stored in the database.

    This helps prevent the same observation from
    generating duplicate alarm records.
    """

    initialize_database(db_path)

    with get_connection(db_path) as connection:

        rows = connection.execute(
            "SELECT alarm_id FROM alarms"
        ).fetchall()

    return {row["alarm_id"] for row in rows}


# =========================================================
# ACKNOWLEDGE AN ALARM
# =========================================================

def acknowledge_alarm(
    alarm_id,
    acknowledged_by="Operator",
    db_path=None
):
    """
    Mark an active alarm as acknowledged.

    Acknowledgement does not delete the alarm.
    """

    initialize_database(db_path)

    from datetime import datetime, timezone

    acknowledged_at = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection(db_path) as connection:

        cursor = connection.execute(
            """
            UPDATE alarms

            SET
                status = 'Acknowledged',
                acknowledged_at = ?,
                acknowledged_by = ?

            WHERE
                alarm_id = ?
                AND status = 'Active'
            """,
            (
                acknowledged_at,
                acknowledged_by,
                alarm_id
            )
        )

        connection.commit()

    return cursor.rowcount > 0
