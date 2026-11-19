
import os
import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. REPRODUCIBILITY AND CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(4, os.cpu_count() or 1))

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "X_train_normal.npy"
METADATA_PATH = ROOT / "data" / "processed" / "train_metadata.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "autoencoder.pth"
THRESHOLD_PATH = MODEL_DIR / "autoencoder_threshold.json"
HISTORY_PATH = REPORT_DIR / "autoencoder_training_history.csv"

INPUT_DIM = 52
BATCH_SIZE = 512
EPOCHS = 25
LEARNING_RATE = 0.001
PATIENCE = 5

MAX_TRAIN_SAMPLES = 100000
MAX_VALIDATION_SAMPLES = 20000
MAX_CALIBRATION_SAMPLES = 20000

DEVICE = torch.device("cpu")


# ============================================================
# 2. LOAD NORMAL DATA
# ============================================================

print("=" * 65)
print("AUTOENCODER TRAINING")
print("=" * 65)

print("\nLoading normal training data...")

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if not METADATA_PATH.exists():
    raise FileNotFoundError(METADATA_PATH)

X = np.load(DATA_PATH, mmap_mode="r")
metadata = pd.read_csv(METADATA_PATH)

if X.shape[1] != INPUT_DIM:
    raise ValueError(
        f"Expected {INPUT_DIM} features, found {X.shape[1]}."
    )

if len(X) != len(metadata):
    raise ValueError(
        "Feature and metadata row counts do not match."
    )

if "simulationRun" not in metadata.columns:
    raise ValueError(
        "simulationRun column is required for run-based splitting."
    )

groups = metadata["simulationRun"].to_numpy()

print(f"Observations: {len(X):,}")
print(f"Features: {X.shape[1]}")
print(f"Simulation runs: {len(np.unique(groups))}")


# ============================================================
# 3. GROUP-BASED DATA SPLITTING
# ============================================================

print("\nCreating run-based training, validation and calibration sets...")

# First separate training runs from the remaining runs.
split_one = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED
)

train_idx, remaining_idx = next(
    split_one.split(X, groups=groups)
)

# Divide remaining runs equally into validation and calibration.
remaining_groups = groups[remaining_idx]

split_two = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED
)

validation_relative, calibration_relative = next(
    split_two.split(
        remaining_idx,
        groups=remaining_groups
    )
)

validation_idx = remaining_idx[validation_relative]
calibration_idx = remaining_idx[calibration_relative]


# Select reproducible subsets to control memory and training time.
def select_subset(indices, maximum, seed):

    if len(indices) <= maximum:
        return indices

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        indices,
        size=maximum,
        replace=False
    )

    return np.sort(selected)


train_idx = select_subset(
    train_idx, MAX_TRAIN_SAMPLES, SEED
)

validation_idx = select_subset(
    validation_idx, MAX_VALIDATION_SAMPLES, SEED + 1
)

calibration_idx = select_subset(
    calibration_idx, MAX_CALIBRATION_SAMPLES, SEED + 2
)

# Verify that simulation runs do not overlap.
train_runs = set(groups[train_idx])
validation_runs = set(groups[validation_idx])
calibration_runs = set(groups[calibration_idx])

assert train_runs.isdisjoint(validation_runs)
assert train_runs.isdisjoint(calibration_runs)
assert validation_runs.isdisjoint(calibration_runs)

print(f"Training observations: {len(train_idx):,}")
print(f"Validation observations: {len(validation_idx):,}")
print(f"Calibration observations: {len(calibration_idx):,}")

print("Run-based separation verified.")


# ============================================================
# 4. DATASET CLASS
# ============================================================

class IndustrialDataset(Dataset):

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        actual_index = self.indices[position]

        row = np.array(
            self.data[actual_index],
            dtype=np.float32,
            copy=True
        )

        return torch.from_numpy(row)


train_dataset = IndustrialDataset(X, train_idx)
validation_dataset = IndustrialDataset(X, validation_idx)
calibration_dataset = IndustrialDataset(X, calibration_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. DEFINE AUTOENCODER ARCHITECTURE
# ============================================================

class IndustrialAutoencoder(nn.Module):

    def __init__(self, input_dim=52):

        super().__init__()

        # Encoder compresses the input.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8)
        )

        # Decoder reconstructs the original input.
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


model = IndustrialAutoencoder(INPUT_DIM).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nAutoencoder architecture:")
print(model)


# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================

def calculate_validation_loss():

    model.eval()

    total_loss = 0.0
    total_observations = 0

    with torch.no_grad():

        for batch in validation_loader:

            batch = batch.to(DEVICE)

            reconstructed = model(batch)

            loss = criterion(reconstructed, batch)

            total_loss += loss.item() * len(batch)

            total_observations += len(batch)

    return total_loss / total_observations


# ============================================================
# 7. TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("STARTING AUTOENCODER TRAINING")
print("=" * 65)

best_validation_loss = float("inf")
best_state = None
epochs_without_improvement = 0

history = []

for epoch in range(EPOCHS):

    model.train()

    total_training_loss = 0.0
    total_observations = 0

    for batch in train_loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        loss.backward()

        optimizer.step()

        total_training_loss += loss.item() * len(batch)

        total_observations += len(batch)

    training_loss = (
        total_training_loss / total_observations
    )

    validation_loss = calculate_validation_loss()

    history.append({
        "epoch": epoch + 1,
        "training_loss": training_loss,
        "validation_loss": validation_loss
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Training Loss: {training_loss:.8f} | "
        f"Validation Loss: {validation_loss:.8f}"
    )

    # Save the best-performing model in memory.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_state = copy.deepcopy(model.state_dict())

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    # Early stopping prevents unnecessary training.
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")

        break


# ============================================================
# 8. RESTORE AND SAVE BEST MODEL
# ============================================================

if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": INPUT_DIM,
        "architecture": [52, 32, 16, 8, 16, 32, 52]
    },
    MODEL_PATH
)

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

print("\nBest model saved successfully.")


# ============================================================
# 9. CALCULATE RECONSTRUCTION ERRORS
# ============================================================

print("\nCalculating normal calibration reconstruction errors...")

model.eval()

calibration_errors = []

with torch.no_grad():

    for batch in calibration_loader:

        batch = batch.to(DEVICE)

        reconstructed = model(batch)

        # Calculate mean squared reconstruction error
        # separately for each observation.
        errors = torch.mean(
            (batch - reconstructed) ** 2,
            dim=1
        )

        calibration_errors.extend(
            errors.cpu().numpy()
        )

calibration_errors = np.asarray(
    calibration_errors,
    dtype=np.float64
)


# ============================================================
# 10. SELECT ANOMALY THRESHOLD
# ============================================================

# Use the 99th percentile of normal calibration errors.
threshold = float(
    np.percentile(calibration_errors, 99)
)

validation_false_alarm_rate = float(
    np.mean(calibration_errors >= threshold)
)

threshold_report = {
    "model": "Autoencoder",
    "threshold": threshold,
    "calibration_percentile": 99,
    "calibration_observations": len(calibration_errors),
    "calibration_false_alarm_rate": validation_false_alarm_rate,
    "best_validation_loss": float(best_validation_loss),
    "input_features": INPUT_DIM,
    "training_samples": len(train_idx),
    "validation_samples": len(validation_idx),
    "calibration_samples": len(calibration_idx)
}

with open(THRESHOLD_PATH, "w") as file:

    json.dump(
        threshold_report,
        file,
        indent=4
    )


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("AUTOENCODER TRAINING SUMMARY")
print("=" * 65)

print(f"Training observations: {len(train_idx):,}")
print(f"Validation observations: {len(validation_idx):,}")
print(f"Calibration observations: {len(calibration_idx):,}")

print(f"Epochs completed: {len(history)}")

print(
    f"Best validation loss: {best_validation_loss:.8f}"
)

print(f"Anomaly threshold: {threshold:.8f}")

print(
    "Calibration false alarm rate: "
    f"{validation_false_alarm_rate:.2%}"
)

print(f"\nModel saved to:\n{MODEL_PATH}")

print(f"\nThreshold saved to:\n{THRESHOLD_PATH}")

print(f"\nTraining history saved to:\n{HISTORY_PATH}")

print("\nAUTOENCODER TRAINING COMPLETED.")
