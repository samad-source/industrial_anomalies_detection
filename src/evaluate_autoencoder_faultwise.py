
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

OUTPUT_PATH = REPORT_DIR / "autoencoder_faultwise.csv"

INPUT_DIM = 52
BATCH_SIZE = 4096

DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# ============================================================
# 2. DEFINE AUTOENCODER
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
print("AUTOENCODER FAULT-SPECIFIC EVALUATION")
print("=" * 65)

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
print(f"Anomaly threshold: {THRESHOLD:.8f}")


# ============================================================
# 4. INITIALIZE FAULT COUNTERS
# ============================================================

fault_results = {}

for fault_number in range(1, 21):

    fault_results[fault_number] = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0
    }


# ============================================================
# 5. EVALUATE ONE BATCH
# ============================================================

def evaluate_batch(features, fault_numbers, labels):

    features = np.asarray(
        features,
        dtype=np.float32
    )

    fault_numbers = np.asarray(
        fault_numbers,
        dtype=np.int16
    )

    labels = np.asarray(
        labels,
        dtype=np.int8
    )

    if not (
        len(features) == len(fault_numbers) == len(labels)
    ):
        raise ValueError("Batch lengths do not match.")

    batch_tensor = torch.from_numpy(
        np.ascontiguousarray(features)
    ).to(DEVICE)

    with torch.inference_mode():

        reconstructed = model(batch_tensor)

        errors = torch.mean(
            (batch_tensor - reconstructed) ** 2,
            dim=1
        )

    scores = errors.cpu().numpy()

    predictions = (
        scores >= THRESHOLD
    ).astype(np.int8)

    # Calculate confusion matrix separately for each fault.
    for fault_number in np.unique(fault_numbers):

        mask = fault_numbers == fault_number

        actual = labels[mask]
        predicted = predictions[mask]

        tp = np.sum((actual == 1) & (predicted == 1))
        tn = np.sum((actual == 0) & (predicted == 0))
        fp = np.sum((actual == 0) & (predicted == 1))
        fn = np.sum((actual == 1) & (predicted == 0))

        result = fault_results[int(fault_number)]

        result["TP"] += int(tp)
        result["TN"] += int(tn)
        result["FP"] += int(fp)
        result["FN"] += int(fn)


# ============================================================
# 6. PROCESS ALL 24 FAULTY TESTING CHUNKS
# ============================================================

print("\nProcessing faulty testing chunks...")

start_time = time.time()

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
        usecols=["faultNumber", "is_anomaly"]
    )

    fault_numbers = metadata["faultNumber"].to_numpy(
        dtype=np.int16
    )

    labels = metadata["is_anomaly"].to_numpy(
        dtype=np.int8
    )

    if not (
        len(X_faulty) == len(fault_numbers) == len(labels)
    ):
        raise ValueError(
            f"{chunk_name}: Feature and metadata counts differ."
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

        evaluate_batch(
            X_faulty[start:end],
            fault_numbers[start:end],
            labels[start:end]
        )

    del X_faulty
    del metadata
    del fault_numbers
    del labels

    print("Chunk completed.")


# ============================================================
# 7. CALCULATE FAULT-SPECIFIC METRICS
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING FAULT-SPECIFIC METRICS")
print("=" * 65)

rows = []

for fault_number in range(1, 21):

    result = fault_results[fault_number]

    TP = result["TP"]
    TN = result["TN"]
    FP = result["FP"]
    FN = result["FN"]

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

    rows.append({

        "fault_number": fault_number,

        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,

        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_alarm_rate": false_alarm_rate,

        "total_observations": TP + TN + FP + FN

    })


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(rows)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("AUTOENCODER FAULT-SPECIFIC RESULTS")
print("=" * 65)

display_columns = [
    "fault_number",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "false_alarm_rate"
]

print(
    results_df[display_columns].to_string(
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

print("\nCONFUSION COUNTS")

print(
    results_df[
        ["fault_number", "TP", "TN", "FP", "FN"]
    ].to_string(index=False)
)

print(f"\nEvaluation time: {(time.time() - start_time) / 60:.2f} minutes")

print(f"\nResults saved to:\n{OUTPUT_PATH}")

print("\nAUTOENCODER FAULT-SPECIFIC EVALUATION COMPLETED.")
