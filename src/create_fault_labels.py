from pathlib import Path
import pandas as pd

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "data" / "processed" / "faulty_testing"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "faulty_testing_labelled"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAULT_ONSET_SAMPLE = 161
EXPECTED_CHUNKS = 24
EXPECTED_ROWS_PER_CHUNK = 400_000

total_rows = 0
total_normal = 0
total_anomalous = 0

print("CREATING FAULT LABELS")
print("=" * 55)

for chunk_number in range(1, EXPECTED_CHUNKS + 1):

    filename = f"faulty_testing_{chunk_number:02d}_metadata.csv"
    input_file = INPUT_DIR / filename

    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    print(f"\nProcessing {filename}")

    metadata = pd.read_csv(input_file)

    required_columns = {
        "faultNumber",
        "simulationRun",
        "sample"
    }

    if not required_columns.issubset(metadata.columns):
        raise ValueError(
            f"Required metadata columns missing in {filename}"
        )

    if len(metadata) != EXPECTED_ROWS_PER_CHUNK:
        raise ValueError(
            f"Unexpected row count in {filename}: {len(metadata)}"
        )

    # Samples 1-160 are normal.
    # Samples 161-960 are anomalous.
    metadata["is_anomaly"] = (
        metadata["sample"] >= FAULT_ONSET_SAMPLE
    ).astype("int8")

    output_file = (
        OUTPUT_DIR /
        f"faulty_testing_{chunk_number:02d}_labelled.csv"
    )

    metadata.to_csv(output_file, index=False)

    normal_count = int((metadata["is_anomaly"] == 0).sum())
    anomaly_count = int((metadata["is_anomaly"] == 1).sum())

    total_rows += len(metadata)
    total_normal += normal_count
    total_anomalous += anomaly_count

    print(f"Rows: {len(metadata):,}")
    print(f"Normal: {normal_count:,}")
    print(f"Anomalous: {anomaly_count:,}")
    print(f"Saved: {output_file.name}")

print("\n" + "=" * 55)
print("LABEL CREATION COMPLETED")
print("=" * 55)

print(f"Chunks processed: {EXPECTED_CHUNKS}")
print(f"Total observations: {total_rows:,}")
print(f"Normal observations: {total_normal:,}")
print(f"Anomalous observations: {total_anomalous:,}")

print("\nExpected totals:")
print("Normal: 1,600,000")
print("Anomalous: 8,000,000")

if total_rows != 9_600_000:
    raise ValueError("Total observation count is incorrect.")

if total_normal != 1_600_000:
    raise ValueError("Normal observation count is incorrect.")

if total_anomalous != 8_000_000:
    raise ValueError("Anomalous observation count is incorrect.")

print("\nAll label counts passed validation.")
print(f"Output directory: {OUTPUT_DIR}")