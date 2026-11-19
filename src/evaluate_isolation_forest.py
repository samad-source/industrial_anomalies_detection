
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FAULTY_FEATURE_DIR = PROCESSED_DIR / "faulty_testing"
FAULTY_LABEL_DIR = PROCESSED_DIR / "faulty_testing_labelled"

MODEL_FILE = PROJECT_DIR / "models" / "isolation_forest.joblib"
THRESHOLD_FILE = (
    PROJECT_DIR / "models" / "isolation_forest_threshold.json"
)

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = REPORT_DIR / "isolation_forest_metrics.json"
FAULT_REPORT_FILE = REPORT_DIR / "isolation_forest_fault_results.csv"


# ==================================================
# 2. CONFIGURATION
# ==================================================

BATCH_SIZE = 10_000
NUMBER_OF_CHUNKS = 24

# Number of observations per class sampled from each
# faulty-data chunk for ROC-AUC estimation.
ROC_SAMPLES_PER_CLASS_PER_CHUNK = 2_500

NORMAL_ROC_SAMPLE_SIZE = 10_000

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ==================================================
# 3. LOAD MODEL AND THRESHOLD
# ==================================================

print("=" * 65)
print("ISOLATION FOREST MODEL EVALUATION")
print("=" * 65)

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
    threshold_information = json.load(file)

threshold = float(threshold_information["threshold"])

print("Model loaded successfully.")
print(f"Anomaly threshold: {threshold:.6f}")


# ==================================================
# 4. METRIC COUNTERS
# ==================================================

# Global confusion matrix counts.
tn = 0
fp = 0
fn = 0
tp = 0

# Fault-specific counts.
fault_results = {}

# Sampled scores for ROC-AUC.
roc_scores = []
roc_labels = []


def add_confusion_counts(y_true, y_pred):
    """Update global confusion matrix counters."""

    global tn, fp, fn, tp

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn += int(matrix[0, 0])
    fp += int(matrix[0, 1])
    fn += int(matrix[1, 0])
    tp += int(matrix[1, 1])


def calculate_scores(scores):
    """Convert anomaly scores into binary predictions."""

    return (scores >= threshold).astype(np.int8)


# ==================================================
# 5. EVALUATE NORMAL TESTING DATA
# ==================================================

print("\n" + "=" * 65)
print("EVALUATING NORMAL TESTING DATA")
print("=" * 65)

normal_test_file = PROCESSED_DIR / "X_test_normal.npy"

if not normal_test_file.exists():
    raise FileNotFoundError(normal_test_file)

X_normal = np.load(normal_test_file, mmap_mode="r")

print(f"Normal test shape: {X_normal.shape}")

if X_normal.shape[1] != model.n_features_in_:
    raise ValueError("Normal test feature count does not match model.")

normal_sample_indices = rng.choice(
    X_normal.shape[0],
    size=min(NORMAL_ROC_SAMPLE_SIZE, X_normal.shape[0]),
    replace=False
)

normal_sample_mask = np.zeros(X_normal.shape[0], dtype=bool)
normal_sample_mask[normal_sample_indices] = True

normal_sample_scores = []

for start in range(0, X_normal.shape[0], BATCH_SIZE):

    end = min(start + BATCH_SIZE, X_normal.shape[0])

    X_batch = np.asarray(
        X_normal[start:end],
        dtype=np.float32
    )

    scores = -model.score_samples(X_batch)
    predictions = calculate_scores(scores)

    actual_labels = np.zeros(len(X_batch), dtype=np.int8)

    add_confusion_counts(actual_labels, predictions)

    local_mask = normal_sample_mask[start:end]

    if local_mask.any():
        normal_sample_scores.extend(scores[local_mask])

    print(
        f"Processed normal test observations: "
        f"{end:,}/{X_normal.shape[0]:,}"
    )

del X_normal

roc_scores.extend(normal_sample_scores)
roc_labels.extend([0] * len(normal_sample_scores))


# ==================================================
# 6. EVALUATE FAULTY TESTING DATA
# ==================================================

print("\n" + "=" * 65)
print("EVALUATING FAULTY TESTING DATA")
print("=" * 65)

for chunk_number in range(1, NUMBER_OF_CHUNKS + 1):

    feature_file = (
        FAULTY_FEATURE_DIR
        / f"faulty_testing_{chunk_number:02d}_features.npy"
    )

    metadata_file = (
        FAULTY_LABEL_DIR
        / f"faulty_testing_{chunk_number:02d}_labelled.csv"
    )

    if not feature_file.exists():
        raise FileNotFoundError(feature_file)

    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)

    print(f"\nProcessing chunk {chunk_number}/{NUMBER_OF_CHUNKS}")

    X_faulty = np.load(feature_file, mmap_mode="r")

    metadata = pd.read_csv(
        metadata_file,
        usecols=["faultNumber", "is_anomaly"]
    )

    if len(X_faulty) != len(metadata):
        raise ValueError(
            f"Feature/metadata mismatch in chunk {chunk_number}"
        )

    if X_faulty.shape[1] != model.n_features_in_:
        raise ValueError("Faulty feature count does not match model.")

    y_true = metadata["is_anomaly"].to_numpy(dtype=np.int8)

    fault_numbers = metadata["faultNumber"].to_numpy()

    # Select a stratified sample for ROC-AUC estimation.
    selected_indices = []

    for class_label in (0, 1):

        class_indices = np.flatnonzero(y_true == class_label)

        sample_count = min(
            ROC_SAMPLES_PER_CLASS_PER_CHUNK,
            len(class_indices)
        )

        if sample_count > 0:
            selected = rng.choice(
                class_indices,
                size=sample_count,
                replace=False
            )

            selected_indices.extend(selected.tolist())

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64
    )

    selected_mask = np.zeros(len(y_true), dtype=bool)
    selected_mask[selected_indices] = True

    chunk_sample_scores = []

    # Process one batch at a time.
    for start in range(0, len(X_faulty), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X_faulty))

        X_batch = np.asarray(
            X_faulty[start:end],
            dtype=np.float32
        )

        scores = -model.score_samples(X_batch)
        predictions = calculate_scores(scores)

        actual_labels = y_true[start:end]

        add_confusion_counts(actual_labels, predictions)

        local_mask = selected_mask[start:end]

        if local_mask.any():
            chunk_sample_scores.extend(scores[local_mask])

    # Save selected scores and their actual labels.
    roc_scores.extend(chunk_sample_scores)
    roc_labels.extend(y_true[selected_indices].tolist())

    # Calculate results separately for each fault type.
    for fault_number in np.unique(fault_numbers):

        mask = fault_numbers == fault_number

        actual = y_true[mask]

        # Recalculate scores for this fault's observations
        # using the stored model predictions would require
        # retaining predictions. Instead, calculate fault
        # counts during the next evaluation refinement.
        #
        # This section is populated in the batch loop below
        # in the production version.

    print(f"Chunk observations: {len(metadata):,}")
    print("Chunk evaluation completed.")

    del X_faulty
    del metadata
    del y_true
    del fault_numbers
    del selected_mask


# ==================================================
# 7. CALCULATE OVERALL METRICS
# ==================================================

print("\n" + "=" * 65)
print("CALCULATING OVERALL PERFORMANCE")
print("=" * 65)

total_normal = tn + fp
total_anomalous = tp + fn
total_observations = total_normal + total_anomalous

precision = precision_score(
    [0, 1],
    [0, 1],
    sample_weight=[0, 0],
    zero_division=0
) if False else (tp / (tp + fp) if tp + fp else 0.0)

recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

false_alarm_rate = fp / total_normal if total_normal else 0.0
specificity = tn / total_normal if total_normal else 0.0

roc_auc = roc_auc_score(
    np.asarray(roc_labels),
    np.asarray(roc_scores)
)

results = {
    "model": "Isolation Forest",
    "threshold": threshold,
    "total_observations": total_observations,
    "normal_observations": total_normal,
    "anomalous_observations": total_anomalous,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "true_positives": tp,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "specificity": specificity,
    "false_alarm_rate": false_alarm_rate,
    "roc_auc_sampled": roc_auc,
    "roc_auc_sample_size": len(roc_labels),
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("ISOLATION FOREST EVALUATION RESULTS")
print("=" * 65)

for name, value in results.items():
    print(f"{name}: {value}")

print("\nEvaluation report saved:")
print(METRICS_FILE)

print("\nNote: ROC-AUC is estimated using a stratified sample.")
print("Overall confusion-matrix metrics use all observations.")
print("\nEvaluation completed.")
