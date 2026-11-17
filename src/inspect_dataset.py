
from pathlib import Path
import pyreadr

# 1. Locate the project and dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "TEP_FaultFree_Training.RData"
)

print("=" * 65)
print("INDUSTRIAL ANOMALY DETECTION")
print("DATASET INSPECTION")
print("=" * 65)

# 2. Check whether the dataset exists

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("\nDataset found successfully!")
print(f"File: {DATA_PATH.name}")
print(f"Size: {DATA_PATH.stat().st_size / (1024 * 1024):.2f} MB")

# 3. Load the RData file

print("\nLoading dataset...")

result = pyreadr.read_r(str(DATA_PATH))

# 4. Identify the objects in the file

print("\nObjects found in the RData file:")

for object_name in result.keys():
    print(f"- {object_name}")

# 5. Inspect each dataset object

for object_name, df in result.items():

    print("\n" + "=" * 65)
    print(f"DATA OBJECT: {object_name}")
    print("=" * 65)

    # Dataset dimensions
    print("\nDataset dimensions:")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # Column names
    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    # First five observations
    print("\nFirst five observations:")
    print(df.head())

    # Data types
    print("\nData types:")
    print(df.dtypes.value_counts())

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum().sum())

    # Fault labels, if available
    if "faultNumber" in df.columns:

        print("\nFault number distribution:")
        print(df["faultNumber"].value_counts().sort_index())

print("\n" + "=" * 65)
print("DATASET INSPECTION COMPLETED")
print("=" * 65)
