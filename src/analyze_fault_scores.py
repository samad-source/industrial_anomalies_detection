
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "isolation_forest.joblib"
THRESHOLD_PATH = ROOT / "models" / "isolation_forest_threshold.json"

FEATURE_DIR = ROOT / "data" / "processed" / "faulty_testing"
LABEL_DIR = ROOT / "data" / "processed" / "faulty_testing_labelled"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "isolation_forest_score_diagnostics.csv"

NUMBER_OF_CHUNKS = 24
BATCH_SIZE = 10_000

# Maximum number of scores sampled per class,
# per fault, per batch.
SAMPLES_PER_CLASS = 20

rng = np.random.default_rng(42)

print("=" * 65)
print("ISOLATION FOREST SCORE DIAGNOSTICS")
print("=" * 65)

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_data = json.load(file)

threshold = float(threshold_data["threshold"])

print(f"\nModel loaded.")
print(f"Threshold: {threshold:.6f}")

# Store sampled scores separately for each fault.
score_samples = {
    fault: {0: [], 1: []}
    for fault in range(1, 21)
}

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

    fault_numbers = metadata["faultNumber"].to_numpy()
    labels = metadata["is_anomaly"].to_numpy()

    if len(X) != len(metadata):
        raise ValueError("Feature and metadata sizes do not match.")

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        X_batch = np.asarray(X[start:end], dtype=np.float32)

        scores = -model.score_samples(X_batch)

        batch_faults = fault_numbers[start:end]
        batch_labels = labels[start:end]

        for fault in range(1, 21):

            fault_mask = batch_faults == fault

            if not np.any(fault_mask):
                continue

            for class_label in (0, 1):

                class_mask = (
                    fault_mask &
                    (batch_labels == class_label)
                )

                class_scores = scores[class_mask]

                if len(class_scores) == 0:
                    continue

                sample_count = min(
                    SAMPLES_PER_CLASS,
                    len(class_scores)
                )

                selected = rng.choice(
                    class_scores,
                    size=sample_count,
                    replace=False
                )

                score_samples[fault][class_label].append(selected)

    del X
    del metadata
    del fault_numbers
    del labels

    print("Chunk completed.")

print("\nCalculating fault-specific ROC-AUC...")

results = []

for fault in range(1, 21):

    normal_scores = np.concatenate(
        score_samples[fault][0]
    )

    anomaly_scores = np.concatenate(
        score_samples[fault][1]
    )

    y_true = np.concatenate([
        np.zeros(len(normal_scores), dtype=np.int8),
        np.ones(len(anomaly_scores), dtype=np.int8)
    ])

    y_scores = np.concatenate([
        normal_scores,
        anomaly_scores
    ])

    auc = roc_auc_score(y_true, y_scores)

    results.append({
        "fault_number": fault,
        "normal_sample_count": len(normal_scores),
        "anomaly_sample_count": len(anomaly_scores),
        "normal_mean_score": float(np.mean(normal_scores)),
        "anomaly_mean_score": float(np.mean(anomaly_scores)),
        "normal_median_score": float(np.median(normal_scores)),
        "anomaly_median_score": float(np.median(anomaly_scores)),
        "normal_95th_percentile": float(
            np.percentile(normal_scores, 95)
        ),
        "anomaly_5th_percentile": float(
            np.percentile(anomaly_scores, 5)
        ),
        "roc_auc": float(auc)
    })

report = pd.DataFrame(results)

report.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FAULT-SPECIFIC SCORE DIAGNOSTICS")
print("=" * 65)

print(
    report[
        [
            "fault_number",
            "normal_mean_score",
            "anomaly_mean_score",
            "roc_auc"
        ]
    ].round(4).to_string(index=False)
)

print(f"\nReport saved to: {OUTPUT_FILE}")
print("\nScore diagnostics completed.")
