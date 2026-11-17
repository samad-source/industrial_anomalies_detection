
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import pyreadr
import joblib
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = (
    [f"xmeas_{i}" for i in range(1, 42)]
    + [f"xmv_{i}" for i in range(1, 12)]
)

METADATA_COLUMNS = ["faultNumber", "simulationRun", "sample"]


# --------------------------------------------------
# 2. LOAD AND PREPARE NORMAL TRAINING DATA
# --------------------------------------------------

print("\nSTEP 1: Loading normal training data...")

training_file = RAW_DIR / "TEP_FaultFree_Training.RData"

result = pyreadr.read_r(str(training_file))
training_df = result["fault_free_training"]
del result
gc.collect()

print("Training shape:", training_df.shape)

# Check expected columns
missing = set(FEATURE_COLUMNS + METADATA_COLUMNS) - set(training_df.columns)

if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Preserve metadata separately
training_metadata = training_df[METADATA_COLUMNS].copy()

# Extract only the 52 process variables
X_train = training_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32,
    copy=True
)

del training_df
gc.collect()

# Fit scaler using normal training data only
print("\nSTEP 2: Scaling normal training data...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(
    np.float32,
    copy=False
)

# Save training data, metadata and scaler
np.save(PROCESSED_DIR / "X_train_normal.npy", X_train_scaled)

training_metadata.to_csv(
    PROCESSED_DIR / "train_metadata.csv",
    index=False
)

joblib.dump(scaler, PROCESSED_DIR / "standard_scaler.joblib")

del X_train, X_train_scaled, training_metadata
gc.collect()

print("Normal training data saved.")


# --------------------------------------------------
# 3. LOAD AND PREPARE NORMAL TESTING DATA
# --------------------------------------------------

print("\nSTEP 3: Loading normal testing data...")

testing_file = RAW_DIR / "TEP_FaultFree_Testing.RData"

result = pyreadr.read_r(str(testing_file))
testing_df = result["fault_free_testing"]
del result
gc.collect()

print("Testing shape:", testing_df.shape)

testing_metadata = testing_df[METADATA_COLUMNS].copy()

X_test = testing_df[FEATURE_COLUMNS].to_numpy(
    dtype=np.float32,
    copy=True
)

del testing_df
gc.collect()

# Apply the existing training scaler.
# Never fit a new scaler on testing data.
print("\nSTEP 4: Scaling normal testing data...")

scaler = joblib.load(PROCESSED_DIR / "standard_scaler.joblib")

X_test_scaled = scaler.transform(X_test).astype(
    np.float32,
    copy=False
)

np.save(PROCESSED_DIR / "X_test_normal.npy", X_test_scaled)

testing_metadata.to_csv(
    PROCESSED_DIR / "test_metadata.csv",
    index=False
)

del X_test, X_test_scaled, testing_metadata, scaler
gc.collect()


# --------------------------------------------------
# 4. SAVE FEATURE INFORMATION
# --------------------------------------------------

with open(PROCESSED_DIR / "feature_columns.json", "w") as file:
    json.dump(FEATURE_COLUMNS, file, indent=4)


# --------------------------------------------------
# 5. COMPLETION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("NORMAL DATA PREPROCESSING COMPLETED")
print("=" * 55)

print("\nFiles created in data/processed:")

for file in sorted(PROCESSED_DIR.iterdir()):
    if file.is_file():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"{file.name}: {size_mb:.2f} MB")

print("\nThe scaler was fitted on normal training data only.")
print("Testing data used the same fitted scaler.")
print("Metadata was preserved separately.")
print("\nNext: Prepare the faulty testing dataset.")
