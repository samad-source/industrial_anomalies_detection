
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.model_selection import GroupShuffleSplit


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"

DATA_FILE = DATA_DIR / "X_train_normal.npy"
METADATA_FILE = DATA_DIR / "train_metadata.csv"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "one_class_svm_run_based.joblib"

THRESHOLD_FILE = (
    MODEL_DIR / "one_class_svm_run_based_threshold.json"
)


# ==================================================
# 2. CONFIGURATION
# ==================================================

TRAIN_SIZE = 5_000
VALIDATION_SIZE = 10_000
CALIBRATION_SIZE = 10_000

RANDOM_STATE = 42

NU = 0.01
GAMMA = "scale"

BATCH_SIZE = 2_000
CALIBRATION_PERCENTILE = 99.0


# ==================================================
# 3. HELPER FUNCTION
# ==================================================

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


# ==================================================
# 4. LOAD NORMAL DATA AND METADATA
# ==================================================

print("=" * 65)
print("RUN-BASED ONE-CLASS SVM TRAINING")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

if not METADATA_FILE.exists():
    raise FileNotFoundError(METADATA_FILE)

print("\nLoading normal training data...")

X = np.load(DATA_FILE, mmap_mode="r")

metadata = pd.read_csv(METADATA_FILE)

if X.ndim != 2:
    raise ValueError("Expected a two-dimensional feature array.")

if len(X) != len(metadata):
    raise ValueError("Feature and metadata row counts do not match.")

if "simulationRun" not in metadata.columns:
    raise ValueError("simulationRun column is missing.")

groups = metadata["simulationRun"].to_numpy()

if pd.isna(groups).any():
    raise ValueError("Missing simulationRun values detected.")

if not np.isfinite(X).all():
    raise ValueError("Non-finite feature values detected.")

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Unique simulation runs: {len(np.unique(groups)):,}")


# ==================================================
# 5. FIRST GROUP SPLIT
# ==================================================

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


# ==================================================
# 6. SECOND GROUP SPLIT
# ==================================================

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


# ==================================================
# 7. VERIFY RUN SEPARATION
# ==================================================

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


# ==================================================
# 8. SELECT OBSERVATIONS
# ==================================================

print("\nSelecting observations...")

rng = np.random.default_rng(RANDOM_STATE)

selected_training_indices = select_subset(
    training_indices,
    TRAIN_SIZE,
    rng
)

selected_validation_indices = select_subset(
    validation_indices,
    VALIDATION_SIZE,
    rng
)

selected_calibration_indices = select_subset(
    calibration_indices,
    CALIBRATION_SIZE,
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


# ==================================================
# 9. INITIALISE ONE-CLASS SVM
# ==================================================

print("\nInitialising One-Class SVM...")

model = OneClassSVM(
    kernel="rbf",
    nu=NU,
    gamma=GAMMA,
    cache_size=256,
    tol=1e-3,
    max_iter=10_000
)

print("\nModel configuration:")
print(f"Kernel: {model.kernel}")
print(f"Nu: {NU}")
print(f"Gamma: {GAMMA}")
print(f"Cache size: {model.cache_size} MB")


# ==================================================
# 10. TRAIN MODEL
# ==================================================

print("\nTraining One-Class SVM...")
print("Only normal-operation observations are used.")

start_time = time.time()

model.fit(X_train)

training_time = time.time() - start_time

print("\nTraining completed.")
print(f"Training time: {training_time:.2f} seconds")
print(f"Support vectors: {len(model.support_vectors_):,}")


# ==================================================
# 11. VALIDATION SCORING
# ==================================================

print("\nScoring normal validation observations...")

validation_scores = []

for start in range(0, len(X_validation), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(X_validation))

    batch = X_validation[start:end]

    scores = -model.decision_function(batch)

    validation_scores.extend(scores.tolist())

validation_scores = np.asarray(validation_scores)

print("Validation scoring completed.")


# ==================================================
# 12. CALIBRATION SCORING
# ==================================================

print("\nScoring normal calibration observations...")

calibration_scores = []

for start in range(0, len(X_calibration), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(X_calibration))

    batch = X_calibration[start:end]

    scores = -model.decision_function(batch)

    calibration_scores.extend(scores.tolist())

calibration_scores = np.asarray(calibration_scores)

print("Calibration scoring completed.")


# ==================================================
# 13. CALCULATE THRESHOLD
# ==================================================

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

print("\nThreshold calibration completed.")

print(f"Anomaly threshold: {threshold:.6f}")

print(
    "Calibration false alarm rate: "
    f"{calibration_false_alarm_rate:.4%}"
)

print(
    "Validation false alarm rate: "
    f"{validation_false_alarm_rate:.4%}"
)


# ==================================================
# 14. SAVE MODEL
# ==================================================

print("\nSaving run-based One-Class SVM...")

joblib.dump(model, MODEL_FILE)

threshold_information = {
    "model": "One-Class SVM",
    "experiment": "run_based",
    "kernel": "rbf",
    "nu": NU,
    "gamma": GAMMA,
    "threshold": threshold,
    "calibration_percentile": CALIBRATION_PERCENTILE,
    "training_observations": int(len(X_train)),
    "validation_observations": int(len(X_validation)),
    "calibration_observations": int(len(X_calibration)),
    "training_runs": int(len(training_runs)),
    "validation_runs": int(len(validation_runs)),
    "calibration_runs": int(len(calibration_runs)),
    "number_of_features": int(X_train.shape[1]),
    "calibration_false_alarm_rate": calibration_false_alarm_rate,
    "validation_false_alarm_rate": validation_false_alarm_rate,
    "training_time_seconds": training_time,
    "support_vectors": int(len(model.support_vectors_)),
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


# ==================================================
# 15. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("RUN-BASED ONE-CLASS SVM TRAINING COMPLETED")
print("=" * 65)

print(f"Training observations: {len(X_train):,}")
print(f"Validation observations: {len(X_validation):,}")
print(f"Calibration observations: {len(X_calibration):,}")

print(f"Training time: {training_time:.2f} seconds")
print(f"Support vectors: {len(model.support_vectors_):,}")
print(f"Anomaly threshold: {threshold:.6f}")

print(
    "Calibration false alarm rate: "
    f"{calibration_false_alarm_rate:.4%}"
)

print(
    "Validation false alarm rate: "
    f"{validation_false_alarm_rate:.4%}"
)

print(f"\nModel saved:\n{MODEL_FILE}")
print(f"\nThreshold saved:\n{THRESHOLD_FILE}")

print("\nNext: Evaluate the run-based model.")
