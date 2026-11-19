
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "one_class_svm.joblib"
THRESHOLD_PATH = ROOT / "models" / "one_class_svm_threshold.json"

FEATURE_DIR = ROOT / "data" / "processed" / "faulty_testing"
LABEL_DIR = ROOT / "data" / "processed" / "faulty_testing_labelled"

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "one_class_svm_faultwise.csv"

BATCH_SIZE = 10000
FAULT_NUMBERS = range(1, 21)


# ============================================================
# LOAD MODEL AND THRESHOLD
# ============================================================

print("=" * 65)
print("ONE-CLASS SVM FAULT-SPECIFIC EVALUATION")
print("=" * 65)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

# Support common threshold JSON formats.
if isinstance(threshold_data, (int, float)):
    threshold = float(threshold_data)
else:
    threshold = float(
        threshold_data.get(
            "threshold",
            threshold_data.get("anomaly_threshold")
        )
    )

print("Model loaded successfully.")
print(f"Anomaly threshold: {threshold:.6f}")


# ============================================================
# INITIALISE METRICS
# ============================================================

results = {}

for fault in FAULT_NUMBERS:
    results[fault] = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0
    }


# ============================================================
# PROCESS FAULTY TESTING CHUNKS
# ============================================================

feature_files = sorted(
    FEATURE_DIR.glob("faulty_testing_*_features.npy")
)

if len(feature_files) != 24:
    raise FileNotFoundError(
        f"Expected 24 feature chunks, found {len(feature_files)}."
    )

print(f"\nFound {len(feature_files)} feature chunks.")

for chunk_number, feature_path in enumerate(feature_files, start=1):

    chunk_name = feature_path.name.replace(
        "_features.npy", ""
    )

    label_path = LABEL_DIR / f"{chunk_name}_labelled.csv"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    print(f"\nProcessing chunk {chunk_number}/24")
    print(f"Chunk: {chunk_name}")

    # Memory-map feature data to reduce RAM usage.
    X = np.load(feature_path, mmap_mode="r")

    # Load only the two columns required for evaluation.
    metadata = pd.read_csv(
        label_path,
        usecols=["faultNumber", "is_anomaly"]
    )

    if len(X) != len(metadata):
        raise ValueError(
            f"Feature/label length mismatch in {chunk_name}: "
            f"{len(X)} features versus {len(metadata)} labels."
        )

    fault_values = metadata["faultNumber"].to_numpy(dtype=np.int16)
    true_labels = metadata["is_anomaly"].to_numpy(dtype=np.int8)

    total_rows = len(X)

    for start in range(0, total_rows, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total_rows)

        X_batch = X[start:end]

        batch_faults = fault_values[start:end]
        batch_labels = true_labels[start:end]

        # Higher score means more anomalous.
        anomaly_scores = -model.decision_function(X_batch)

        predictions = (
            anomaly_scores >= threshold
        ).astype(np.int8)

        # Accumulate confusion-matrix counts per fault.
        for fault in FAULT_NUMBERS:

            mask = batch_faults == fault

            if not np.any(mask):
                continue

            y_true = batch_labels[mask]
            y_pred = predictions[mask]

            results[fault]["TP"] += int(
                np.sum((y_true == 1) & (y_pred == 1))
            )

            results[fault]["TN"] += int(
                np.sum((y_true == 0) & (y_pred == 0))
            )

            results[fault]["FP"] += int(
                np.sum((y_true == 0) & (y_pred == 1))
            )

            results[fault]["FN"] += int(
                np.sum((y_true == 1) & (y_pred == 0))
            )

    print(f"Completed {total_rows:,} observations.")


# ============================================================
# CALCULATE FAULT-SPECIFIC METRICS
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING FAULT-SPECIFIC METRICS")
print("=" * 65)

records = []

for fault in FAULT_NUMBERS:

    counts = results[fault]

    TP = counts["TP"]
    TN = counts["TN"]
    FP = counts["FP"]
    FN = counts["FN"]

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    specificity = (
        TN / (TN + FP)
        if TN + FP else 0.0
    )

    false_alarm_rate = (
        FP / (FP + TN)
        if FP + TN else 0.0
    )

    records.append({
        "fault_number": fault,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate
    })


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(records)

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 65)
print("ONE-CLASS SVM FAULT-SPECIFIC RESULTS")
print("=" * 65)

print(
    df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1_score": "{:.4f}".format,
            "specificity": "{:.4f}".format,
            "false_alarm_rate": "{:.4f}".format
        }
    )
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nFault-specific evaluation completed.")
