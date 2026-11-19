
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "isolation_forest.joblib"

THRESHOLD_PATH = (
    ROOT / "models" / "isolation_forest_threshold.json"
)

FEATURE_DIR = ROOT / "data" / "processed" / "faulty_testing"

LABEL_DIR = (
    ROOT / "data" / "processed" / "faulty_testing_labelled"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "isolation_forest_faultwise.csv"


# ==================================================
# 2. CONFIGURATION
# ==================================================

NUMBER_OF_CHUNKS = 24
BATCH_SIZE = 10_000

print("=" * 60)
print("FAULT-SPECIFIC ISOLATION FOREST EVALUATION")
print("=" * 60)


# ==================================================
# 3. LOAD MODEL
# ==================================================

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_data = json.load(file)

threshold = float(threshold_data["threshold"])

print(f"\nModel loaded.")
print(f"Threshold: {threshold:.6f}")


# ==================================================
# 4. INITIALISE FAULT COUNTERS
# ==================================================

results = {}

for fault in range(1, 21):

    results[fault] = {
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0
    }


# ==================================================
# 5. PROCESS EACH CHUNK
# ==================================================

for chunk_number in range(1, NUMBER_OF_CHUNKS + 1):

    print(f"\nProcessing chunk {chunk_number}/{NUMBER_OF_CHUNKS}")

    feature_path = (
        FEATURE_DIR /
        f"faulty_testing_{chunk_number:02d}_features.npy"
    )

    metadata_path = (
        LABEL_DIR /
        f"faulty_testing_{chunk_number:02d}_labelled.csv"
    )

    X = np.load(feature_path, mmap_mode="r")

    metadata = pd.read_csv(
        metadata_path,
        usecols=["faultNumber", "is_anomaly"]
    )

    if len(X) != len(metadata):
        raise ValueError("Feature and metadata sizes do not match.")

    # Process observations in manageable batches.
    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        X_batch = np.asarray(
            X[start:end],
            dtype=np.float32
        )

        scores = -model.score_samples(X_batch)

        predictions = (
            scores >= threshold
        ).astype(np.int8)

        batch_faults = metadata["faultNumber"].to_numpy()[start:end]
        actual = metadata["is_anomaly"].to_numpy()[start:end]

        for fault in np.unique(batch_faults):

            mask = batch_faults == fault

            y_true = actual[mask]
            y_pred = predictions[mask]

            matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            results[int(fault)]["tn"] += int(matrix[0, 0])
            results[int(fault)]["fp"] += int(matrix[0, 1])
            results[int(fault)]["fn"] += int(matrix[1, 0])
            results[int(fault)]["tp"] += int(matrix[1, 1])

    print("Chunk completed.")

    del X
    del metadata


# ==================================================
# 6. CALCULATE FAULT-SPECIFIC METRICS
# ==================================================

report = []

for fault, counts in results.items():

    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]

    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    false_alarm_rate = (
        fp / (tn + fp)
        if tn + fp else 0.0
    )

    report.append({
        "fault_number": fault,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate
    })


# ==================================================
# 7. SAVE RESULTS
# ==================================================

report_df = pd.DataFrame(report)

report_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("FAULT-SPECIFIC RESULTS")
print("=" * 60)

print(
    report_df[
        [
            "fault_number",
            "precision",
            "recall",
            "f1_score",
            "false_alarm_rate"
        ]
    ].to_string(index=False)
)

print(f"\nReport saved to: {OUTPUT_FILE}")

print("\nFault-specific evaluation completed.")
