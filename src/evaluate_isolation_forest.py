
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FAULTY_FEATURE_DIR = PROCESSED_DIR / "faulty_testing"
FAULTY_LABEL_DIR = PROCESSED_DIR / "faulty_testing_labelled"

# Load the NEW run-based model.
MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "isolation_forest_run_based.joblib"
)

THRESHOLD_FILE = (
    PROJECT_DIR
    / "models"
    / "isolation_forest_run_based_threshold.json"
)

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# New report names preserve preliminary results.
METRICS_FILE = (
    REPORT_DIR / "isolation_forest_run_based_metrics.json"
)

FAULT_REPORT_FILE = (
    REPORT_DIR / "isolation_forest_run_based_faultwise.csv"
)


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
print("RUN-BASED ISOLATION FOREST EVALUATION")
print("=" * 65)

print("\nLoading run-based model...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(MODEL_FILE)

if not THRESHOLD_FILE.exists():
    raise FileNotFoundError(THRESHOLD_FILE)

model = joblib.load(MODEL_FILE)

with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as file:
    threshold_information = json.load(file)

threshold = float(threshold_information["threshold"])

print("Run-based model loaded successfully.")
print(f"Anomaly threshold: {threshold:.6f}")
print(f"Model features: {model.n_features_in_}")


# ==================================================
# 4. METRIC COUNTERS
# ==================================================

tn = 0
fp = 0
fn = 0
tp = 0

# Each fault gets its own confusion matrix counts.
fault_results = {}

# Sampled scores and corresponding labels for ROC-AUC.
roc_scores = []
roc_labels = []


def calculate_predictions(scores):
    """Convert anomaly scores into binary predictions."""

    return (scores >= threshold).astype(np.int8)


def update_global_counts(y_true, y_pred):
    """Update global confusion matrix counts."""

    global tn, fp, fn, tp

    tn += int(np.sum((y_true == 0) & (y_pred == 0)))
    fp += int(np.sum((y_true == 0) & (y_pred == 1)))

    fn += int(np.sum((y_true == 1) & (y_pred == 0)))
    tp += int(np.sum((y_true == 1) & (y_pred == 1)))


def update_fault_counts(fault_numbers, y_true, y_pred):
    """Update confusion counts for each fault type."""

    for fault_number in np.unique(fault_numbers):

        mask = fault_numbers == fault_number

        actual = y_true[mask]
        predicted = y_pred[mask]

        fault_number = int(fault_number)

        if fault_number not in fault_results:

            fault_results[fault_number] = {
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 0
            }

        result = fault_results[fault_number]

        result["true_negatives"] += int(
            np.sum((actual == 0) & (predicted == 0))
        )

        result["false_positives"] += int(
            np.sum((actual == 0) & (predicted == 1))
        )

        result["false_negatives"] += int(
            np.sum((actual == 1) & (predicted == 0))
        )

        result["true_positives"] += int(
            np.sum((actual == 1) & (predicted == 1))
        )


# ==================================================
# 5. EVALUATE NORMAL TESTING DATA
# ==================================================

print("\n" + "=" * 65)
print("EVALUATING NORMAL TESTING DATA")
print("=" * 65)

normal_test_file = PROCESSED_DIR / "X_test_normal.npy"

if not normal_test_file.exists():
    raise FileNotFoundError(normal_test_file)

X_normal = np.load(
    normal_test_file,
    mmap_mode="r"
)

print(f"Normal test shape: {X_normal.shape}")

if X_normal.shape[1] != model.n_features_in_:
    raise ValueError(
        "Normal test feature count does not match model."
    )

normal_sample_indices = rng.choice(
    X_normal.shape[0],
    size=min(
        NORMAL_ROC_SAMPLE_SIZE,
        X_normal.shape[0]
    ),
    replace=False
)

normal_sample_mask = np.zeros(
    X_normal.shape[0],
    dtype=bool
)

normal_sample_mask[normal_sample_indices] = True

normal_sample_scores = []

for start in range(0, X_normal.shape[0], BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        X_normal.shape[0]
    )

    X_batch = np.asarray(
        X_normal[start:end],
        dtype=np.float32
    )

    scores = -model.score_samples(X_batch)

    predictions = calculate_predictions(scores)

    actual_labels = np.zeros(
        len(X_batch),
        dtype=np.int8
    )

    update_global_counts(
        actual_labels,
        predictions
    )

    local_mask = normal_sample_mask[start:end]

    if local_mask.any():

        normal_sample_scores.extend(
            scores[local_mask].tolist()
        )

    print(
        f"Processed normal test observations: "
        f"{end:,}/{X_normal.shape[0]:,}"
    )

del X_normal

roc_scores.extend(normal_sample_scores)

roc_labels.extend(
    [0] * len(normal_sample_scores)
)


# ==================================================
# 6. EVALUATE FAULTY TESTING DATA
# ==================================================

print("\n" + "=" * 65)
print("EVALUATING FAULTY TESTING DATA")
print("=" * 65)

for chunk_number in range(
    1,
    NUMBER_OF_CHUNKS + 1
):

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

    print(
        f"\nProcessing chunk "
        f"{chunk_number}/{NUMBER_OF_CHUNKS}"
    )

    X_faulty = np.load(
        feature_file,
        mmap_mode="r"
    )

    metadata = pd.read_csv(
        metadata_file,
        usecols=["faultNumber", "is_anomaly"]
    )

    if len(X_faulty) != len(metadata):
        raise ValueError(
            f"Feature/metadata mismatch in chunk {chunk_number}"
        )

    if X_faulty.shape[1] != model.n_features_in_:
        raise ValueError(
            "Faulty feature count does not match model."
        )

    y_true = metadata["is_anomaly"].to_numpy(
        dtype=np.int8
    )

    fault_numbers = metadata["faultNumber"].to_numpy()

    # Select a balanced sample from both classes.
    selected_indices = []

    for class_label in (0, 1):

        class_indices = np.flatnonzero(
            y_true == class_label
        )

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

            selected_indices.extend(
                selected.tolist()
            )

    # CRITICAL: Sort selected indices so sampled labels
    # and scores follow exactly the same observation order.
    selected_indices = np.sort(
        np.asarray(
            selected_indices,
            dtype=np.int64
        )
    )

    selected_mask = np.zeros(
        len(y_true),
        dtype=bool
    )

    selected_mask[selected_indices] = True

    chunk_sample_scores = []

    # Process faulty observations in batches.
    for start in range(
        0,
        len(X_faulty),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_faulty)
        )

        X_batch = np.asarray(
            X_faulty[start:end],
            dtype=np.float32
        )

        scores = -model.score_samples(X_batch)

        predictions = calculate_predictions(scores)

        actual_labels = y_true[start:end]

        batch_fault_numbers = fault_numbers[start:end]

        # Update overall metrics.
        update_global_counts(
            actual_labels,
            predictions
        )

        # Update individual fault metrics.
        update_fault_counts(
            batch_fault_numbers,
            actual_labels,
            predictions
        )

        # Collect scores in ascending observation order.
        local_mask = selected_mask[start:end]

        if local_mask.any():

            chunk_sample_scores.extend(
                scores[local_mask].tolist()
            )

    # The sorted indices and batch-order score collection
    # now have matching observation order.
    chunk_sample_labels = y_true[selected_indices]

    if len(chunk_sample_scores) != len(chunk_sample_labels):
        raise ValueError(
            "ROC-AUC sample score/label counts do not match."
        )

    roc_scores.extend(chunk_sample_scores)

    roc_labels.extend(
        chunk_sample_labels.tolist()
    )

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

total_observations = (
    total_normal + total_anomalous
)

precision = (
    tp / (tp + fp)
    if tp + fp
    else 0.0
)

recall = (
    tp / (tp + fn)
    if tp + fn
    else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0.0
)

specificity = (
    tn / total_normal
    if total_normal
    else 0.0
)

false_alarm_rate = (
    fp / total_normal
    if total_normal
    else 0.0
)

roc_labels_array = np.asarray(
    roc_labels,
    dtype=np.int8
)

roc_scores_array = np.asarray(
    roc_scores,
    dtype=np.float64
)

if len(np.unique(roc_labels_array)) < 2:
    raise ValueError(
        "ROC-AUC requires both normal and anomalous samples."
    )

roc_auc = roc_auc_score(
    roc_labels_array,
    roc_scores_array
)


# ==================================================
# 8. SAVE OVERALL RESULTS
# ==================================================

results = {
    "model": "Isolation Forest",
    "experiment": "run_based",
    "threshold": threshold,
    "total_observations": int(total_observations),
    "normal_observations": int(total_normal),
    "anomalous_observations": int(total_anomalous),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "specificity": specificity,
    "false_alarm_rate": false_alarm_rate,
    "roc_auc_sampled": float(roc_auc),
    "roc_auc_sample_size": int(len(roc_labels)),
    "roc_auc_sampling_method": (
        "Normal test random sample plus "
        "stratified faulty-data chunk samples"
    ),
    "evaluation_batch_size": BATCH_SIZE
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==================================================
# 9. CALCULATE FAULT-SPECIFIC METRICS
# ==================================================

print("\nCalculating fault-specific metrics...")

fault_rows = []

for fault_number in sorted(fault_results):

    counts = fault_results[fault_number]

    fault_tn = counts["true_negatives"]
    fault_fp = counts["false_positives"]
    fault_fn = counts["false_negatives"]
    fault_tp = counts["true_positives"]

    fault_precision = (
        fault_tp / (fault_tp + fault_fp)
        if fault_tp + fault_fp
        else 0.0
    )

    fault_recall = (
        fault_tp / (fault_tp + fault_fn)
        if fault_tp + fault_fn
        else 0.0
    )

    fault_f1 = (
        2 * fault_precision * fault_recall
        / (fault_precision + fault_recall)
        if fault_precision + fault_recall
        else 0.0
    )

    fault_total_normal = fault_tn + fault_fp

    fault_specificity = (
        fault_tn / fault_total_normal
        if fault_total_normal
        else 0.0
    )

    fault_false_alarm_rate = (
        fault_fp / fault_total_normal
        if fault_total_normal
        else 0.0
    )

    fault_rows.append({
        "fault_number": fault_number,
        "true_negatives": fault_tn,
        "false_positives": fault_fp,
        "false_negatives": fault_fn,
        "true_positives": fault_tp,
        "precision": fault_precision,
        "recall": fault_recall,
        "f1_score": fault_f1,
        "specificity": fault_specificity,
        "false_alarm_rate": fault_false_alarm_rate
    })


fault_dataframe = pd.DataFrame(fault_rows)

fault_dataframe.to_csv(
    FAULT_REPORT_FILE,
    index=False
)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 65)
print("RUN-BASED ISOLATION FOREST RESULTS")
print("=" * 65)

for name, value in results.items():

    if isinstance(value, float):
        print(f"{name}: {value:.6f}")

    else:
        print(f"{name}: {value}")

print("\n" + "=" * 65)
print("FAULT-SPECIFIC RECALL")
print("=" * 65)

for row in fault_rows:

    print(
        f"Fault {row['fault_number']:02d}: "
        f"Precision={row['precision']:.4f}, "
        f"Recall={row['recall']:.4f}, "
        f"F1={row['f1_score']:.4f}"
    )

print("\nReports saved successfully:")

print(METRICS_FILE)
print(FAULT_REPORT_FILE)

print("\nROC-AUC is estimated using aligned sampled scores and labels.")
print("Confusion-matrix metrics use all observations.")

print("\nRun-based Isolation Forest evaluation completed.")
