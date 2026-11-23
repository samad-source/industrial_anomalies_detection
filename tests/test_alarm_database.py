
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.alarm_database import (
    initialize_database,
    save_alarms,
    get_all_alarms,
    get_known_alarm_ids,
    acknowledge_alarm
)


def run_test():

    with tempfile.TemporaryDirectory() as temporary_folder:

        test_database = (
            Path(temporary_folder) / "test_alarms.db"
        )

        # 1. Initialize database
        initialize_database(test_database)

        print("1. Database initialized successfully.")

        # 2. Create a sample alarm
        sample_alarm = {
            "alarm_id": "TEST-ALARM-001",
            "detected_at": "2026-11-23T09:00:00",
            "scenario": "Test Scenario",
            "fault_number": 1,
            "simulation_run": 1,
            "sample": 100,
            "detecting_models": [
                "Isolation Forest",
                "Autoencoder"
            ],
            "models_flagged": 2,
            "alarm_type": "CONSENSUS",
            "severity": "HIGH",
            "priority": 1,
            "scores": {
                "Isolation Forest": -0.25,
                "Autoencoder": 0.85
            },
            "status": "Active",
            "acknowledged_at": None,
            "acknowledged_by": None,
            "operator_notes": ""
        }

        # 3. Save alarm
        inserted = save_alarms(
            [sample_alarm],
            test_database
        )

        print(f"2. Alarms inserted: {inserted}")

        # 4. Test duplicate prevention
        duplicate_inserted = save_alarms(
            [sample_alarm],
            test_database
        )

        print(
            f"3. Duplicate alarms inserted: "
            f"{duplicate_inserted}"
        )

        # 5. Retrieve alarms
        alarms = get_all_alarms(test_database)

        print(f"4. Retrieved alarms: {len(alarms)}")

        # 6. Retrieve known IDs
        known_ids = get_known_alarm_ids(test_database)

        print(
            "5. Alarm ID found:",
            "TEST-ALARM-001" in known_ids
        )

        # 7. Acknowledge alarm
        acknowledged = acknowledge_alarm(
            "TEST-ALARM-001",
            db_path=test_database
        )

        print("6. Acknowledgement successful:", acknowledged)

        # 8. Confirm updated status
        updated_alarms = get_all_alarms(test_database)

        print(
            "7. Final alarm status:",
            updated_alarms[0]["status"]
        )

        # 9. Confirm everything
        assert inserted == 1
        assert duplicate_inserted == 0
        assert len(alarms) == 1
        assert "TEST-ALARM-001" in known_ids
        assert acknowledged is True
        assert updated_alarms[0]["status"] == "Acknowledged"

        print("\nALL DATABASE TESTS PASSED!")


if __name__ == "__main__":
    run_test()
