
from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "X_train_normal.npy"
)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "isolation_forest.joblib"
THRESHOLD_FILE = MODEL_DIR / "isolation_forest_threshold.json"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
TRAINING_SAMPLES = 100_000
VALIDATION_PERCENTILE = 99.0

N_ESTIMATORS = 200


# --------------------------------------------------
# 3. LOAD NORMAL TRAINING DATA
# --------------------------------------------------

print("=" * 60)
print("ISOLATION FOREST TRAINING")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Normal training dataset not found: {DATA_FILE}"
    )

print("\nLoading normal training data...")

X = np.load(DATA_FILE, mmap_mode="r")

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

if X.ndim != 2:
    raise ValueError("Expected a two-dimensional feature array.")

if not np.isfinite(X).all():
    raise ValueError("Dataset contains non-finite values.")


# --------------------------------------------------
# 4. SPLIT NORMAL DATA INTO TRAINING AND VALIDATION
# --------------------------------------------------

print("\nPreparing training and validation samples...")

rng = np.random.default_rng(RANDOM_STATE)

indices = rng.permutation(X.shape[0])

split_index = int(X.shape[0] * 0.8)

training_indices = indices[:split_index]
validation_indices = indices[split_index:]

if TRAINING_SAMPLES > len(training_indices):
    raise ValueError("Not enough training observations.")

selected_training_indices = training_indices[
    :TRAINING_SAMPLES
]

X_train = np.asarray(
    X[selected_training_indices],
    dtype=np.float32
)

X_validation = np.asarray(
    X[validation_indices],
    dtype=np.float32
)

print(f"Training observations: {len(X_train):,}")
print(f"Validation observations: {len(X_validation):,}")


# --------------------------------------------------
# 5. INITIALISE ISOLATION FOREST
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
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")
print("Training uses normal-operation observations only.")

model.fit(X_train)

print("Model training completed.")


# --------------------------------------------------
# 7. CALCULATE NORMAL VALIDATION SCORES
# --------------------------------------------------

print("\nCalculating validation anomaly scores...")

# Higher scores indicate greater anomaly likelihood.
validation_scores = -model.score_samples(X_validation)

threshold = float(
    np.percentile(
        validation_scores,
        VALIDATION_PERCENTILE
    )
)

validation_predictions = (
    validation_scores >= threshold
).astype(np.int8)

false_alarm_rate = float(
    validation_predictions.mean()
)

print(f"Anomaly threshold: {threshold:.6f}")
print(f"Validation false alarm rate: {false_alarm_rate:.4%}")


# --------------------------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------------------------

print("\nSaving trained model...")

joblib.dump(model, MODEL_FILE)

threshold_information = {
    "algorithm": "Isolation Forest",
    "threshold": threshold,
    "validation_percentile": VALIDATION_PERCENTILE,
    "validation_false_alarm_rate": false_alarm_rate,
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_validation)),
    "number_of_features": int(X.shape[1]),
    "random_state": RANDOM_STATE
}

with open(THRESHOLD_FILE, "w", encoding="utf-8") as file:
    json.dump(threshold_information, file, indent=4)


# --------------------------------------------------
# 9. COMPLETION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("ISOLATION FOREST TRAINING COMPLETED")
print("=" * 60)

print(f"Model saved: {MODEL_FILE}")
print(f"Threshold saved: {THRESHOLD_FILE}")
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_validation):,}")
print(f"Validation false alarm rate: {false_alarm_rate:.4%}")

print("\nNext: Evaluate the model on unseen normal and faulty data.")
