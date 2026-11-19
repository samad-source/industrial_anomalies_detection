
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import pyreadr
import joblib


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "faulty_testing_chunks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "faulty_testing"

SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "standard_scaler.joblib"
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_columns.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD EXISTING PREPROCESSING ARTIFACTS
# --------------------------------------------------

print("Loading feature columns...")

with open(FEATURE_PATH, "r") as file:
    feature_columns = json.load(file)

print("Loading existing normal-training scaler...")

scaler = joblib.load(SCALER_PATH)

if len(feature_columns) != scaler.n_features_in_:
    raise ValueError(
        "Feature count does not match the fitted scaler."
    )

print("Features:", len(feature_columns))
print("Scaler loaded successfully.")


# --------------------------------------------------
# 3. FIND RDATA CHUNKS
# --------------------------------------------------

chunk_files = sorted(INPUT_DIR.glob("faulty_testing_*.RData"))

if len(chunk_files) != 24:
    raise ValueError(
        f"Expected 24 RData chunks, found {len(chunk_files)}."
    )

print(f"\nFound {len(chunk_files)} RData chunks.")


# --------------------------------------------------
# 4. PROCESS EACH CHUNK
# --------------------------------------------------

total_rows = 0

for index, chunk_file in enumerate(chunk_files, start=1):

    print(f"\nProcessing chunk {index}/24: {chunk_file.name}")

    # Read one RData file
    result = pyreadr.read_r(str(chunk_file))
    df = result["chunk"]

    # Validate columns
    required_columns = [
        "faultNumber",
        "simulationRun",
        "sample"
    ] + feature_columns

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Preserve metadata
    metadata = df[
        ["faultNumber", "simulationRun", "sample"]
    ].copy()

    # Extract process variables
    X = df[feature_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite values found in {chunk_file.name}"
        )

    # Apply the scaler fitted on normal training data
    X_scaled = scaler.transform(X).astype(np.float32)

    # Save features and metadata separately
    chunk_name = chunk_file.stem

    np.save(
        OUTPUT_DIR / f"{chunk_name}_features.npy",
        X_scaled
    )

    metadata.to_csv(
        OUTPUT_DIR / f"{chunk_name}_metadata.csv",
        index=False
    )

    rows = len(df)
    total_rows += rows

    print(f"Saved {rows:,} observations.")

    # Release memory
    del result, df, metadata, X, X_scaled
    gc.collect()


# --------------------------------------------------
# 5. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("FAULTY TESTING PREPROCESSING COMPLETED")
print("=" * 55)

print("Chunks processed:", len(chunk_files))
print("Total observations:", f"{total_rows:,}")
print("Features per observation:", len(feature_columns))
print("Output directory:", OUTPUT_DIR)
