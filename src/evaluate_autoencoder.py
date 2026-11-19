
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
FAULTY_DIR = DATA_DIR / "faulty_testing"
LABEL_DIR = DATA_DIR / "faulty_testing_labelled"

MODEL_PATH = ROOT / "models" / "autoencoder.pth"
THRESHOLD_PATH = ROOT / "models" / "autoencoder_threshold.json"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "autoencoder_evaluation.json"

INPUT_DIM = 52
BATCH_SIZE = 4096
AUC_SAMPLES_PER_CLASS_PER_BATCH = 100

DEVICE = torch.device("cpu")

torch.set_num_threads(4)

rng = np.random.default_rng(42)


# ============================================================
# 2. DEFINE THE AUTOENCODER
# ============================================================

class IndustrialAutoencoder(nn.Module):

    def __init__(self, input_dim=52):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(encoded)

        return reconstructed


# ============================================================
# 3. LOAD MODEL AND THRESHOLD
# ============================================================

print("=" * 65)
print("AUTOENCODER EVALUATION")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(THRESHOLD_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = IndustrialAutoencoder(
    input_dim=checkpoint["input_dim"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

with open(THRESHOLD_PATH, "r") as file:
    threshold_report = json.load(file)

THRESHOLD = float(threshold_report["threshold"])

print("\nModel loaded successfully.")
print(f"Architecture: {checkpoint['architecture']}")
print(f"Threshold: {THRESHOLD:.8f}")
print(f"Batch size: {BATCH_SIZE:,}")


# ============================================================
# 4. INITIALIZE EVALUATION COUNTS
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0

total_normal = 0
total_anomalous = 0
total_observations = 0

sampled_scores = []
sampled_labels = []


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate_batch(features, labels):

    global TP, TN, FP, FN
    global total_normal, total_anomalous
    global total_observations

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int8
    )

    if len(features) != len(labels):
        raise ValueError(
            "Feature and label counts do not match."
        )

    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected feature shape (n, {INPUT_DIM}), "
            f"received {features.shape}."
        )

    # Convert the batch into a PyTorch tensor.
    batch_tensor = torch.from_numpy(
        np.ascontiguousarray(features)
    ).to(DEVICE)

    # Generate reconstructed observations.
    with torch.inference_mode():

        reconstructed = model(batch_tensor)

        # Mean squared reconstruction error
        # for each individual observation.
        errors = torch.mean(
            (batch_tensor - reconstructed) ** 2,
            dim=1
        )

    scores = errors.cpu().numpy()

    # Higher reconstruction error means more anomalous.
    predictions = (
        scores >= THRESHOLD
    ).astype(np.int8)

    # Calculate confusion matrix counts.
    batch_tn, batch_fp, batch_fn, batch_tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    TN += int(batch_tn)
    FP += int(batch_fp)
    FN += int(batch_fn)
    TP += int(batch_tp)

    normal_indices = np.flatnonzero(labels == 0)
    anomaly_indices = np.flatnonzero(labels == 1)

    total_normal += len(normal_indices)
    total_anomalous += len(anomaly_indices)
    total_observations += len(labels)

    # Collect a balanced sample for ROC-AUC.
    # Scores and labels are selected together to preserve alignment.

    for indices, class_label in [
        (normal_indices, 0),
        (anomaly_indices, 1)
    ]:

        if len(indices) == 0:
            continue

        sample_size = min(
            AUC_SAMPLES_PER_CLASS_PER_BATCH,
            len(indices)
        )

        selected = rng.choice(
            indices,
            size=sample_size,
            replace=False
        )

        sampled_scores.extend(
            scores[selected].tolist()
        )

        sampled_labels.extend(
            [class_label] * sample_size
        )


# ============================================================
# 6. EVALUATE NORMAL TESTING DATA
# ============================================================

print("\n" + "=" * 65)
print("EVALUATING NORMAL TESTING DATA")
print("=" * 65)

normal_path = DATA_DIR / "X_test_normal.npy"

if not normal_path.exists():
    raise FileNotFoundError(normal_path)

X_normal = np.load(
    normal_path,
    mmap_mode="r"
)

print(f"Normal observations: {len(X_normal):,}")

start_time = time.time()

for start in range(0, len(X_normal), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(X_normal)
    )

    batch = X_normal[start:end]

    labels = np.zeros(
        len(batch),
        dtype=np.int8
    )

    evaluate_batch(batch, labels)

print("Normal testing data completed.")


# ============================================================
# 7. EVALUATE ALL FAULTY TESTING CHUNKS
# ============================================================

print("\n" + "=" * 65)
print("EVALUATING FAULTY TESTING DATA")
print("=" * 65)

for chunk_number in range(1, 25):

    chunk_name = f"faulty_testing_{chunk_number:02d}"

    feature_path = (
        FAULTY_DIR / f"{chunk_name}_features.npy"
    )

    label_path = (
        LABEL_DIR / f"{chunk_name}_labelled.csv"
    )

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    X_faulty = np.load(
        feature_path,
        mmap_mode="r"
    )

    metadata = pd.read_csv(
        label_path,
        usecols=["is_anomaly"]
    )

    labels = metadata["is_anomaly"].to_numpy(
        dtype=np.int8
    )

    if len(X_faulty) != len(labels):
        raise ValueError(
            f"{chunk_name}: Feature and label counts differ."
        )

    print(
        f"\nProcessing chunk {chunk_number}/24: "
        f"{chunk_name}"
    )

    for start in range(0, len(X_faulty), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(X_faulty)
        )

        batch = X_faulty[start:end]
        batch_labels = labels[start:end]

        evaluate_batch(
            batch,
            batch_labels
        )

    del X_faulty
    del metadata
    del labels

    print("Chunk completed.")


# ============================================================
# 8. CALCULATE FINAL PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING PERFORMANCE METRICS")
print("=" * 65)

precision = (
    TP / (TP + FP)
    if TP + FP > 0 else 0.0
)

recall = (
    TP / (TP + FN)
    if TP + FN > 0 else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0
)

specificity = (
    TN / (TN + FP)
    if TN + FP > 0 else 0.0
)

false_alarm_rate = (
    FP / (FP + TN)
    if FP + TN > 0 else 0.0
)

accuracy = (
    (TP + TN) / total_observations
    if total_observations > 0 else 0.0
)

# Sampled ROC-AUC.
sampled_auc = roc_auc_score(
    sampled_labels,
    sampled_scores
)


# ============================================================
# 9. SAVE EVALUATION RESULTS
# ============================================================

results = {

    "model": "Autoencoder",

    "threshold": THRESHOLD,

    "total_observations": total_observations,

    "normal_observations": total_normal,

    "anomalous_observations": total_anomalous,

    "true_negatives": TN,

    "false_positives": FP,

    "false_negatives": FN,

    "true_positives": TP,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "specificity": specificity,

    "false_alarm_rate": false_alarm_rate,

    "roc_auc_sampled": float(sampled_auc),

    "roc_auc_sample_size": len(sampled_labels),

    "roc_auc_sampling_method": (
        "Balanced per-batch random sampling"
    ),

    "evaluation_batch_size": BATCH_SIZE

}

with open(OUTPUT_PATH, "w") as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# 10. DISPLAY FINAL RESULTS
# ============================================================

elapsed_time = time.time() - start_time

print("\n" + "=" * 65)
print("AUTOENCODER EVALUATION RESULTS")
print("=" * 65)

print(f"Threshold:             {THRESHOLD:.8f}")

print(f"Total observations:    {total_observations:,}")

print(f"Normal observations:   {total_normal:,}")

print(f"Anomalous observations:{total_anomalous:>10,}")

print("\nCONFUSION MATRIX")

print(f"True Negatives:        {TN:,}")
print(f"False Positives:       {FP:,}")
print(f"False Negatives:       {FN:,}")
print(f"True Positives:        {TP:,}")

print("\nPERFORMANCE METRICS")

print(f"Accuracy:              {accuracy:.4f}")
print(f"Precision:             {precision:.4f}")
print(f"Recall:                {recall:.4f}")
print(f"F1-Score:              {f1:.4f}")
print(f"Specificity:           {specificity:.4f}")
print(f"False Alarm Rate:      {false_alarm_rate:.4f}")
print(f"Sampled ROC-AUC:       {sampled_auc:.4f}")

print(f"\nROC-AUC sample size:   {len(sampled_labels):,}")

print(f"\nEvaluation time:       {elapsed_time / 60:.2f} minutes")

print(f"\nResults saved to:\n{OUTPUT_PATH}")

print("\nAUTOENCODER EVALUATION COMPLETED.")
