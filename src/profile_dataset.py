
from pathlib import Path
import pyreadr
import pandas as pd

# INDUSTRIAL ANOMALY DETECTION
# DATASET PROFILING AND STATISTICAL UNDERSTANDING

# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "TEP_FaultFree_Training.RData"
)

REPORTS_PATH = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("INDUSTRIAL ANOMALY DETECTION")
print("DATASET PROFILING")
print("=" * 65)

# 2. Load the training dataset
print("\nLoading fault-free training dataset...")

result = pyreadr.read_r(str(DATA_PATH))
df = result["fault_free_training"]

print("Dataset loaded successfully!")

# 3. Identify metadata and process variables
metadata_columns = [
    "faultNumber",
    "simulationRun",
    "sample"
]

feature_columns = [
    column for column in df.columns
    if column.startswith("xmeas_")
    or column.startswith("xmv_")
]

print("\nCOLUMN INFORMATION")
print("-" * 65)

print(f"Total columns: {len(df.columns)}")
print(f"Metadata columns: {len(metadata_columns)}")
print(f"Process variables: {len(feature_columns)}")

print("\nProcess variable groups:")
print(
    "Measurement variables:",
    len([c for c in feature_columns if c.startswith("xmeas_")])
)
print(
    "Manipulated variables:",
    len([c for c in feature_columns if c.startswith("xmv_")])
)

# 4. Examine simulation runs
print("\nSIMULATION RUN ANALYSIS")
print("-" * 65)

number_of_runs = df["simulationRun"].nunique()

observations_per_run = (
    df.groupby("simulationRun")
    .size()
)

print(f"Number of simulation runs: {number_of_runs}")
print(
    "Minimum observations per run:",
    observations_per_run.min()
)
print(
    "Maximum observations per run:",
    observations_per_run.max()
)
print(
    "Average observations per run:",
    observations_per_run.mean()
)

# Save run information
observations_per_run.to_csv(
    REPORTS_PATH / "observations_per_run.csv",
    header=["observation_count"]
)

# 5. Check missing values
print("\nMISSING VALUE ANALYSIS")
print("-" * 65)

missing_values = df[feature_columns].isnull().sum()

total_missing = missing_values.sum()

print(f"Total missing feature values: {total_missing}")

if total_missing > 0:
    print("\nVariables with missing values:")
    print(missing_values[missing_values > 0])
else:
    print("No missing values found in process variables.")

# 6. Check duplicate observations
print("\nDUPLICATE OBSERVATION ANALYSIS")
print("-" * 65)

duplicate_count = df.duplicated().sum()

print(f"Exact duplicate rows: {duplicate_count}")

# 7. Generate descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print("-" * 65)

statistics = df[feature_columns].describe().T

statistics["variance"] = df[feature_columns].var()

statistics["missing_values"] = (
    df[feature_columns].isnull().sum()
)

statistics.to_csv(
    REPORTS_PATH / "feature_descriptive_statistics.csv"
)

print(statistics.head(10).round(3))

# 8. Identify constant variables
print("\nCONSTANT VARIABLE ANALYSIS")
print("-" * 65)

constant_variables = [
    column
    for column in feature_columns
    if df[column].nunique() <= 1
]

if constant_variables:
    print("Constant variables found:")
    print(constant_variables)
else:
    print("No constant process variables found.")

# 9. Save a profiling summary
summary = {
    "total_observations": len(df),
    "total_columns": len(df.columns),
    "metadata_columns": len(metadata_columns),
    "process_variables": len(feature_columns),
    "measurement_variables": 41,
    "manipulated_variables": 11,
    "simulation_runs": number_of_runs,
    "minimum_observations_per_run": int(
        observations_per_run.min()
    ),
    "maximum_observations_per_run": int(
        observations_per_run.max()
    ),
    "missing_feature_values": int(total_missing),
    "duplicate_rows": int(duplicate_count),
    "constant_variables": len(constant_variables)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    REPORTS_PATH / "dataset_profiling_summary.csv",
    index=False
)

# 10. Display completion message
print("\n" + "=" * 65)
print("DATASET PROFILING COMPLETED")
print("=" * 65)

print("\nReports saved inside:")
print(REPORTS_PATH)

print("\nGenerated files:")
print("- dataset_profiling_summary.csv")
print("- feature_descriptive_statistics.csv")
print("- observations_per_run.csv")
