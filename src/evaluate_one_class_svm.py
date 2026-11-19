
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = ROOT / "data" / "processed"

FAULTY_FEATURE_DIR = PROCESSED_DIR / "faulty_testing"
FAULTY_LABEL_DIR = PROCESSED_DIR / "faulty_testing_labelled"

MODEL_FILE = ROOT / "models" / "one_class_svm.joblib"
THRESHOLD_FILE = ROOT / "models" / "one_class_svm_threshold.json"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = REPORT_DIR / "one_class_svm_metrics.json"

# ==================================================
# 2. CONFIGURATION
# ==================================================

BATCH_SIZE = 10_000
NUMBER_OF_CHUNKS = 24

ROC_SAMPLES_PER_CLASS_PER_CHUNK = 2_500
NORMAL_ROC_SAMPLE_SIZE = 10_000

RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ==================================================
# 3. LOAD MODEL AND THRESHOLD
# ==================================================

print("=" * 65)
print("ONE-CLASS SVM MODEL EVALUATION")
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

tn = 0
fp = 0
fn = 0
tp = 0

roc_scores = []
roc_labels = []


def update_confusion_counts(y_true, y_pred):

    global tn, fp, fn, tp

    tn += int(np.sum((y_true == 0) & (y_pred == 0)))
    fp += int(np.sum((y_true == 0) & (y_pred == 1)))
    fn += int(np.sum((y_true == 1) & (y_pred == 0)))
    tp += int(np.sum((y_true == 1) & (y_pred == 1)))


def calculate_anomaly_scores(X_batch):

    # One-Class SVM decision_function:
    # Lower values mean observations are less normal.
    # Negating converts them into anomaly scores.

    return -model.decision_function(X_batch)


# ==================================================
# 5. EVALUATE NORMAL TESTING DATA
# ==================================================

print("\n" + "=" * 65)
print("EVALUATING NORMAL TESTING DATA")
print("=" * 65)

normal_test_file = PROCESSED_DIR / "X_test_normal.npy"

X_normal = np.load(normal_test_file, mmap_mode="r")

print(f"Normal test shape: {X_normal.shape}")

if X_normal.shape[1] != model.n_features_in_:
    raise ValueError("Feature count does not match model.")

normal_sample_indices = rng.choice(
    X_normal.shape[0],
    size=min(NORMAL_ROC_SAMPLE_SIZE, X_normal.shape[0]),
    replace=False
)

normal_sample_mask = np.zeros(
    X_normal.shape[0],
    dtype=bool
)

normal_sample_mask[normal_sample_indices] = True

normal_sample_scores = []

for start in range(0, len(X_normal), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(X_normal))

    X_batch = np.asarray(
        X_normal[start:end],
        dtype=np.float32
    )

    scores = calculate_anomaly_scores(X_batch)

    predictions = (scores >= threshold).astype(np.int8)

    actual = np.zeros(len(X_batch), dtype=np.int8)

    update_confusion_counts(actual, predictions)

    local_mask = normal_sample_mask[start:end]

    if local_mask.any():
        normal_sample_scores.extend(scores[local_mask])

    print(
        f"Processed normal observations: "
        f"{end:,}/{len(X_normal):,}"
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

    print(f"\nProcessing chunk {chunk_number}/{NUMBER_OF_CHUNKS}")

    X_faulty = np.load(feature_file, mmap_mode="r")

    metadata = pd.read_csv(
        metadata_file,
        usecols=["is_anomaly"]
    )

    if len(X_faulty) != len(metadata):
        raise ValueError("Feature and metadata sizes do not match.")

    y_true = metadata["is_anomaly"].to_numpy(dtype=np.int8)

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
    selected_indices.sort()

    selected_mask = np.zeros(len(y_true), dtype=bool)
    selected_mask[selected_indices] = True

    chunk_sample_scores = []

    for start in range(0, len(X_faulty), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X_faulty))

        X_batch = np.asarray(
            X_faulty[start:end],
            dtype=np.float32
        )

        scores = calculate_anomaly_scores(X_batch)

        predictions = (scores >= threshold).astype(np.int8)

        actual = y_true[start:end]

        update_confusion_counts(actual, predictions)

        local_mask = selected_mask[start:end]

        if local_mask.any():
            chunk_sample_scores.extend(scores[local_mask])

    roc_scores.extend(chunk_sample_scores)
    roc_labels.extend(y_true[selected_indices].tolist())

    print(f"Chunk observations: {len(metadata):,}")
    print("Chunk evaluation completed.")

    del X_faulty
    del metadata
    del y_true
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

precision = tp / (tp + fp) if tp + fp else 0.0

recall = tp / (tp + fn) if tp + fn else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

specificity = (
    tn / total_normal
    if total_normal else 0.0
)

false_alarm_rate = (
    fp / total_normal
    if total_normal else 0.0
)

roc_auc = roc_auc_score(
    np.asarray(roc_labels),
    np.asarray(roc_scores)
)


# ==================================================
# 8. SAVE RESULTS
# ==================================================

results = {
    "model": "One-Class SVM",
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
    "roc_auc_sampled": float(roc_auc),
    "roc_auc_sample_size": len(roc_labels)
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:

    json.dump(results, file, indent=4)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("ONE-CLASS SVM EVALUATION RESULTS")
print("=" * 65)

for name, value in results.items():
    print(f"{name}: {value}")

print(f"\nEvaluation report saved to:\n{METRICS_FILE}")

print("\nNote: ROC-AUC is estimated using a stratified sample.")
print("Confusion-matrix metrics use all observations.")

print("\nOne-Class SVM evaluation completed.")
