
from datetime import datetime, timezone
import json
import math

import pandas as pd


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_CONFIG = {
    "Isolation Forest": {
        "prediction_column": "isolation_forest_prediction",
        "score_column": "isolation_forest_score"
    },
    "One-Class SVM": {
        "prediction_column": "one_class_svm_prediction",
        "score_column": "one_class_svm_score"
    },
    "Autoencoder": {
        "prediction_column": "autoencoder_prediction",
        "score_column": "autoencoder_score"
    }
}


# =========================================================
# VALUE NORMALISATION
# =========================================================

def _normalise_value(value):
    """Convert metadata values into stable, serialisable values."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


# =========================================================
# PREDICTION VALIDATION
# =========================================================

def _is_anomaly(value):
    """Return True when a model prediction identifies an anomaly."""

    if pd.isna(value):
        return False

    try:
        return int(value) == 1
    except (TypeError, ValueError, OverflowError):
        return False


# =========================================================
# MODEL SCORE COLLECTION
# =========================================================

def _collect_model_scores(row):
    """Collect available scores from all detection models."""

    scores = {}

    for model_name, config in MODEL_CONFIG.items():

        score_column = config["score_column"]

        if score_column not in row.index:
            continue

        value = row[score_column]

        if pd.isna(value):
            continue

        score = float(value)

        if math.isfinite(score):
            scores[model_name] = score

    return scores


# =========================================================
# ALARM IDENTITY
# =========================================================

def _build_alarm_identity(row, scenario_name, row_index):
    """Create a stable identity for each detected observation."""

    identity = {
        "scenario": str(scenario_name),
        "fault_number": _normalise_value(
            row.get("faultNumber")
        ),
        "simulation_run": _normalise_value(
            row.get("simulationRun")
        ),
        "sample": _normalise_value(
            row.get("sample")
        )
    }

    if (
        identity["fault_number"] is None
        or identity["simulation_run"] is None
        or identity["sample"] is None
    ):
        identity["row_index"] = _normalise_value(row_index)

    return identity


# =========================================================
# ALARM GENERATION
# =========================================================

def generate_alarm_events(
    results,
    scenario_name,
    known_alarm_ids
):
    """
    Generate one alarm event for each newly detected anomalous
    observation.

    One model detecting an anomaly produces a single-model alert.

    Two or more models detecting the same observation produce
    a consensus alert.

    known_alarm_ids prevents duplicate alarm events.
    """

    if results is None or results.empty:
        return []

    required_columns = [
        config["prediction_column"]
        for config in MODEL_CONFIG.values()
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    if known_alarm_ids is None:
        known_alarm_ids = set()

    alarms = []

    # -----------------------------------------------------
    # PROCESS EACH OBSERVATION
    # -----------------------------------------------------

    for row_index, row in results.iterrows():

        detecting_models = []

        for model_name, config in MODEL_CONFIG.items():

            prediction_column = config["prediction_column"]

            if _is_anomaly(row[prediction_column]):
                detecting_models.append(model_name)

        # Ignore normal observations.
        if not detecting_models:
            continue

        # -------------------------------------------------
        # CREATE UNIQUE OBSERVATION ID
        # -------------------------------------------------

        identity = _build_alarm_identity(
            row,
            scenario_name,
            row_index
        )

        alarm_id = json.dumps(
            identity,
            sort_keys=True,
            default=str
        )

        # Prevent duplicate alarm records.
        if alarm_id in known_alarm_ids:
            continue

        known_alarm_ids.add(alarm_id)

        # -------------------------------------------------
        # CLASSIFY ALARM
        # -------------------------------------------------

        consensus = len(detecting_models) >= 2

        if consensus:
            alarm_type = "CONSENSUS"
            severity = "HIGH"
            priority = 1
        else:
            alarm_type = "SINGLE-MODEL"
            severity = "MEDIUM"
            priority = 2

        # -------------------------------------------------
        # COLLECT MODEL SCORES
        # -------------------------------------------------

        scores = _collect_model_scores(row)

        # -------------------------------------------------
        # BUILD ALARM RECORD
        # -------------------------------------------------

        alarm = {
            "alarm_id": alarm_id,

            "detected_at": datetime.now(
                timezone.utc
            ).astimezone().isoformat(
                timespec="seconds"
            ),

            "scenario": str(scenario_name),

            "fault_number": identity["fault_number"],

            "simulation_run": identity["simulation_run"],

            "sample": identity["sample"],

            "detecting_models": detecting_models,

            "models_flagged": len(detecting_models),

            "alarm_type": alarm_type,

            "severity": severity,

            "priority": priority,

            "scores": scores,

            "status": "Active",

            "acknowledged_at": None,

            "acknowledged_by": None,

            "operator_notes": ""
        }

        alarms.append(alarm)

    return alarms
