
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed"

DATA_FILE = DATA_DIR / "X_train_normal.npy"
METADATA_FILE = DATA_DIR / "train_metadata.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Keep the original preliminary model untouched.
MODEL_FILE = MODEL_DIR / "isolation_forest_run_based.joblib"

THRESHOLD_FILE = (
    MODEL_DIR / "isolation_forest_run_based_threshold.json"
)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42

TRAINING_SAMPLES = 100_000
VALIDATION_SAMPLES = 20_000
CALIBRATION_SAMPLES = 20_000

CALIBRATION_PERCENTILE = 99.0

N_ESTIMATORS = 200


# --------------------------------------------------
# 3. HELPER FUNCTION
# --------------------------------------------------

def select_subset(indices, number_of_samples, rng):

    if number_of_samples > len(indices):
        raise ValueError(
            f"Requested {number_of_samples:,} observations, "
            f"but only {len(indices):,} are available."
        )

    selected = rng.choice(
        indices,
        size=number_of_samples,
        replace=False
    )

    return np.sort(selected)


# --------------------------------------------------
# 4. LOAD NORMAL DATA AND METADATA
# --------------------------------------------------

print("=" * 65)
print("RUN-BASED ISOLATION FOREST TRAINING")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

print("\nLoading normal-operation data...")

X = np.load(DATA_FILE, mmap_mode="r")

metadata = pd.read_csv(METADATA_FILE)

if X.ndim != 2:
    raise ValueError("Expected a two-dimensional feature array.")

if len(X) != len(metadata):
    raise ValueError(
        "Feature and metadata row counts do not match."
    )

if "simulationRun" not in metadata.columns:
    raise ValueError(
        "The metadata file must contain simulationRun."
    )

groups = metadata["simulationRun"].to_numpy()

if pd.isna(groups).any():
    raise ValueError("Missing simulationRun values detected.")

if not np.isfinite(X).all():
    raise ValueError(
        "Dataset contains non-finite feature values."
    )

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Total observations: {len(X):,}")
print(f"Unique simulation runs: {len(np.unique(groups)):,}")


# --------------------------------------------------
# 5. FIRST GROUP SPLIT: TRAINING AND REMAINING DATA
# --------------------------------------------------

print("\nCreating run-based training split...")

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

training_indices, remaining_indices = next(
    first_split.split(
        np.zeros(len(groups)),
        groups=groups
    )
)


# --------------------------------------------------
# 6. SECOND GROUP SPLIT: VALIDATION AND CALIBRATION
# --------------------------------------------------

print("Separating validation and calibration runs...")

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

validation_relative, calibration_relative = next(
    second_split.split(
        np.zeros(len(remaining_indices)),
        groups=groups[remaining_indices]
    )
)

validation_indices = remaining_indices[validation_relative]

calibration_indices = remaining_indices[calibration_relative]


# --------------------------------------------------
# 7. VERIFY RUN SEPARATION
# --------------------------------------------------

training_runs = set(groups[training_indices])
validation_runs = set(groups[validation_indices])
calibration_runs = set(groups[calibration_indices])

if training_runs & validation_runs:
    raise ValueError("Training and validation runs overlap.")

if training_runs & calibration_runs:
    raise ValueError("Training and calibration runs overlap.")

if validation_runs & calibration_runs:
    raise ValueError("Validation and calibration runs overlap.")

print("\nSimulation-run separation verified.")

print(f"Training runs: {len(training_runs):,}")
print(f"Validation runs: {len(validation_runs):,}")
print(f"Calibration runs: {len(calibration_runs):,}")


# --------------------------------------------------
# 8. SELECT OBSERVATIONS
# --------------------------------------------------

print("\nSelecting observations from each group...")

rng = np.random.default_rng(RANDOM_STATE)

selected_training_indices = select_subset(
    training_indices,
    TRAINING_SAMPLES,
    rng
)

selected_validation_indices = select_subset(
    validation_indices,
    VALIDATION_SAMPLES,
    rng
)

selected_calibration_indices = select_subset(
    calibration_indices,
    CALIBRATION_SAMPLES,
    rng
)

X_train = np.asarray(
    X[selected_training_indices],
    dtype=np.float32
)

X_validation = np.asarray(
    X[selected_validation_indices],
    dtype=np.float32
)

X_calibration = np.asarray(
    X[selected_calibration_indices],
    dtype=np.float32
)

print(f"Training observations: {len(X_train):,}")
print(f"Validation observations: {len(X_validation):,}")
print(f"Calibration observations: {len(X_calibration):,}")


# --------------------------------------------------
# 9. INITIALISE ISOLATION FOREST
# --------------------------------------------------

print("\nInitialising Isolation Forest...")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination="auto",
    max_samples="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# --------------------------------------------------
# 10. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")
print("Only normal-operation observations are used.")

model.fit(X_train)

print("Model training completed.")


# --------------------------------------------------
# 11. CALCULATE VALIDATION SCORES
# --------------------------------------------------

print("\nCalculating validation anomaly scores...")

validation_scores = -model.score_samples(
    X_validation
)


# --------------------------------------------------
# 12. CALIBRATE ANOMALY THRESHOLD
# --------------------------------------------------

print("\nCalculating calibration anomaly scores...")

calibration_scores = -model.score_samples(
    X_calibration
)

threshold = float(
    np.percentile(
        calibration_scores,
        CALIBRATION_PERCENTILE
    )
)

calibration_predictions = (
    calibration_scores >= threshold
).astype(np.int8)

validation_predictions = (
    validation_scores >= threshold
).astype(np.int8)

calibration_false_alarm_rate = float(
    calibration_predictions.mean()
)

validation_false_alarm_rate = float(
    validation_predictions.mean()
)

print(f"\nAnomaly threshold: {threshold:.6f}")

print(
    "Calibration false alarm rate: "
    f"{calibration_false_alarm_rate:.4%}"
)

print(
    "Validation false alarm rate: "
    f"{validation_false_alarm_rate:.4%}"
)


# --------------------------------------------------
# 13. SAVE TRAINED MODEL
# --------------------------------------------------

print("\nSaving run-based model...")

joblib.dump(model, MODEL_FILE)

threshold_information = {
    "algorithm": "Isolation Forest",
    "experiment": "run_based",
    "threshold": threshold,
    "calibration_percentile": CALIBRATION_PERCENTILE,
    "calibration_false_alarm_rate": calibration_false_alarm_rate,
    "validation_false_alarm_rate": validation_false_alarm_rate,
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_validation)),
    "calibration_samples": int(len(X_calibration)),
    "training_runs": int(len(training_runs)),
    "validation_runs": int(len(validation_runs)),
    "calibration_runs": int(len(calibration_runs)),
    "number_of_features": int(X.shape[1]),
    "n_estimators": N_ESTIMATORS,
    "random_state": RANDOM_STATE
}

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        threshold_information,
        file,
        indent=4
    )


# --------------------------------------------------
# 14. COMPLETION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("RUN-BASED ISOLATION FOREST TRAINING COMPLETED")
print("=" * 65)

print(f"Model saved: {MODEL_FILE}")
print(f"Threshold saved: {THRESHOLD_FILE}")

print(f"\nTraining observations: {len(X_train):,}")
print(f"Validation observations: {len(X_validation):,}")
print(f"Calibration observations: {len(X_calibration):,}")

print(
    "Calibration false alarm rate: "
    f"{calibration_false_alarm_rate:.4%}"
)

print(
    "Validation false alarm rate: "
    f"{validation_false_alarm_rate:.4%}"
)

print("\nNext: Evaluate the run-based model.")
