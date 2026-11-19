
from pathlib import Path
import json
import time

import joblib
import numpy as np

from sklearn.svm import OneClassSVM


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "X_train_normal.npy"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "one_class_svm.joblib"
THRESHOLD_FILE = MODEL_DIR / "one_class_svm_threshold.json"


# ==================================================
# 2. CONFIGURATION
# ==================================================

TRAIN_SIZE = 5_000
VALIDATION_SIZE = 10_000

RANDOM_STATE = 42

NU = 0.01
GAMMA = "scale"

BATCH_SIZE = 2_000


# ==================================================
# 3. LOAD NORMAL TRAINING DATA
# ==================================================

print("=" * 65)
print("ONE-CLASS SVM TRAINING")
print("=" * 65)

print("\nLoading normal training data...")

X = np.load(DATA_FILE, mmap_mode="r")

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

if X.shape[0] < TRAIN_SIZE + VALIDATION_SIZE:
    raise ValueError("Insufficient observations for the experiment.")


# ==================================================
# 4. SELECT TRAINING AND VALIDATION SAMPLES
# ==================================================

print("\nSelecting training and validation observations...")

rng = np.random.default_rng(RANDOM_STATE)

selected_indices = rng.choice(
    X.shape[0],
    size=TRAIN_SIZE + VALIDATION_SIZE,
    replace=False
)

train_indices = selected_indices[:TRAIN_SIZE]
validation_indices = selected_indices[TRAIN_SIZE:]

X_train = np.asarray(
    X[train_indices],
    dtype=np.float32
)

X_validation = np.asarray(
    X[validation_indices],
    dtype=np.float32
)

del X
del selected_indices
del train_indices
del validation_indices

print(f"Training observations: {len(X_train):,}")
print(f"Validation observations: {len(X_validation):,}")


# ==================================================
# 5. INITIALIZE ONE-CLASS SVM
# ==================================================

print("\nInitializing One-Class SVM...")

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
# 6. TRAIN MODEL
# ==================================================

print("\nTraining One-Class SVM...")
print("This may take some time.")

start_time = time.time()

model.fit(X_train)

training_time = time.time() - start_time

print("\nTraining completed.")

print(f"Training time: {training_time:.2f} seconds")
print(f"Support vectors: {len(model.support_vectors_):,}")


# ==================================================
# 7. VALIDATE MODEL
# ==================================================

print("\nEvaluating normal validation observations...")

validation_scores = []

for start in range(0, len(X_validation), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(X_validation))

    batch = X_validation[start:end]

    # One-Class SVM decision_function:
    # Higher values indicate more normal observations.
    # Negating the scores makes higher values more anomalous.

    scores = -model.decision_function(batch)

    validation_scores.extend(scores)

validation_scores = np.asarray(validation_scores)

print("Validation scoring completed.")


# ==================================================
# 8. CALCULATE ANOMALY THRESHOLD
# ==================================================

threshold = float(
    np.percentile(validation_scores, 99)
)

predictions = (
    validation_scores >= threshold
).astype(np.int8)

false_positives = int(np.sum(predictions == 1))

false_alarm_rate = (
    false_positives / len(validation_scores)
)

print("\nThreshold calibration completed.")

print(f"Anomaly threshold: {threshold:.6f}")
print(f"Validation false alarms: {false_positives:,}")
print(f"Validation false alarm rate: {false_alarm_rate:.4%}")


# ==================================================
# 9. SAVE TRAINED MODEL
# ==================================================

print("\nSaving trained model...")

joblib.dump(model, MODEL_FILE)

threshold_information = {
    "model": "One-Class SVM",
    "kernel": "rbf",
    "nu": NU,
    "gamma": GAMMA,
    "threshold": threshold,
    "training_observations": TRAIN_SIZE,
    "validation_observations": VALIDATION_SIZE,
    "number_of_features": int(X_train.shape[1]),
    "validation_false_alarm_rate": false_alarm_rate,
    "training_time_seconds": training_time,
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
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("ONE-CLASS SVM TRAINING SUMMARY")
print("=" * 65)

print(f"Training observations: {TRAIN_SIZE:,}")
print(f"Validation observations: {VALIDATION_SIZE:,}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Support vectors: {len(model.support_vectors_):,}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Anomaly threshold: {threshold:.6f}")
print(f"Validation false alarm rate: {false_alarm_rate:.4%}")

print(f"\nModel saved to:\n{MODEL_FILE}")
print(f"\nThreshold saved to:\n{THRESHOLD_FILE}")

print("\nOne-Class SVM training completed.")
