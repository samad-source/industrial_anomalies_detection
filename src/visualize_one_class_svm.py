
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SVM_METRICS_PATH = REPORT_DIR / "one_class_svm_metrics.json"
SVM_FAULTWISE_PATH = REPORT_DIR / "one_class_svm_faultwise.csv"

IF_METRICS_PATH = REPORT_DIR / "isolation_forest_metrics.json"
IF_FAULTWISE_PATH = REPORT_DIR / "isolation_forest_faultwise.csv"


# ============================================================
# LOAD REPORTS
# ============================================================

print("=" * 65)
print("ONE-CLASS SVM VISUALIZATION")
print("=" * 65)

required_files = [
    SVM_METRICS_PATH,
    SVM_FAULTWISE_PATH,
    IF_METRICS_PATH,
    IF_FAULTWISE_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {file_path}"
        )

with open(SVM_METRICS_PATH, "r") as file:
    svm_metrics = json.load(file)

with open(IF_METRICS_PATH, "r") as file:
    if_metrics = json.load(file)

svm_faultwise = pd.read_csv(SVM_FAULTWISE_PATH)
if_faultwise = pd.read_csv(IF_FAULTWISE_PATH)

# Sort fault numbers consistently.
svm_faultwise = svm_faultwise.sort_values("fault_number")
if_faultwise = if_faultwise.sort_values("fault_number")

print("\nAll evaluation reports loaded successfully.")


# ============================================================
# COMMON PLOTTING SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (11, 7)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


def save_figure(filename):
    """Save the current figure in the figures directory."""
    output_path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

tn = svm_metrics["true_negatives"]
fp = svm_metrics["false_positives"]
fn = svm_metrics["false_negatives"]
tp = svm_metrics["true_positives"]

confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

plt.figure(figsize=(8, 6))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt=",",
    cmap="Blues",
    xticklabels=["Predicted Normal", "Predicted Anomaly"],
    yticklabels=["Actual Normal", "Actual Anomaly"],
    cbar=True
)

plt.title("One-Class SVM - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

save_figure("07_one_class_svm_confusion_matrix.png")


# ============================================================
# 2. NORMALIZED CONFUSION MATRIX
# ============================================================

print("\nGenerating normalized confusion matrix...")

row_totals = confusion_matrix.sum(axis=1, keepdims=True)

normalized_matrix = np.divide(
    confusion_matrix,
    row_totals,
    out=np.zeros_like(confusion_matrix, dtype=float),
    where=row_totals != 0
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    normalized_matrix,
    annot=True,
    fmt=".2%",
    cmap="Greens",
    xticklabels=["Predicted Normal", "Predicted Anomaly"],
    yticklabels=["Actual Normal", "Actual Anomaly"],
    vmin=0,
    vmax=1
)

plt.title("One-Class SVM - Normalized Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

save_figure("08_one_class_svm_normalized_confusion_matrix.png")


# ============================================================
# 3. FAULT-SPECIFIC PRECISION, RECALL AND F1-SCORE
# ============================================================

print("\nGenerating fault-specific performance chart...")

faults = svm_faultwise["fault_number"].astype(str)

x = np.arange(len(faults))
width = 0.25

plt.figure(figsize=(15, 7))

plt.bar(
    x - width,
    svm_faultwise["precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    svm_faultwise["recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    svm_faultwise["f1_score"],
    width,
    label="F1-score"
)

plt.xticks(x, faults)
plt.ylim(0, 1.08)

plt.xlabel("Fault Number")
plt.ylabel("Score")
plt.title("One-Class SVM - Fault-Specific Performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)

save_figure("09_one_class_svm_faultwise_metrics.png")


# ============================================================
# 4. FAULT-SPECIFIC RECALL COMPARISON
# ============================================================

print("\nGenerating recall comparison...")

comparison = pd.merge(
    if_faultwise[["fault_number", "recall"]],
    svm_faultwise[["fault_number", "recall"]],
    on="fault_number",
    suffixes=("_isolation_forest", "_one_class_svm")
)

x = np.arange(len(comparison))
width = 0.38

plt.figure(figsize=(15, 7))

plt.bar(
    x - width / 2,
    comparison["recall_isolation_forest"],
    width,
    label="Isolation Forest"
)

plt.bar(
    x + width / 2,
    comparison["recall_one_class_svm"],
    width,
    label="One-Class SVM"
)

plt.xticks(
    x,
    comparison["fault_number"].astype(str)
)

plt.ylim(0, 1.08)

plt.xlabel("Fault Number")
plt.ylabel("Recall")
plt.title("Fault-Specific Recall: Isolation Forest vs One-Class SVM")
plt.legend()
plt.grid(axis="y", alpha=0.3)

save_figure("10_recall_comparison_if_vs_svm.png")


# ============================================================
# 5. OVERALL PERFORMANCE COMPARISON
# ============================================================

print("\nGenerating overall performance comparison...")

metric_names = [
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "false_alarm_rate",
    "roc_auc_sampled"
]

metric_labels = [
    "Precision",
    "Recall",
    "F1-score",
    "Specificity",
    "False Alarm Rate",
    "ROC-AUC*"
]

if_values = [
    if_metrics[name] for name in metric_names
]

svm_values = [
    svm_metrics[name] for name in metric_names
]

x = np.arange(len(metric_names))
width = 0.38

plt.figure(figsize=(13, 7))

plt.bar(
    x - width / 2,
    if_values,
    width,
    label="Isolation Forest"
)

plt.bar(
    x + width / 2,
    svm_values,
    width,
    label="One-Class SVM"
)

plt.xticks(x, metric_labels)
plt.ylim(0, 1.08)

plt.ylabel("Metric Value")
plt.title("Overall Model Performance Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)

save_figure("11_overall_performance_comparison.png")


# ============================================================
# 6. FAULT-SPECIFIC FALSE ALARM RATE
# ============================================================

print("\nGenerating false alarm rate chart...")

plt.figure(figsize=(14, 6))

plt.bar(
    svm_faultwise["fault_number"].astype(str),
    svm_faultwise["false_alarm_rate"]
)

plt.xlabel("Fault Number")
plt.ylabel("False Alarm Rate")
plt.title("One-Class SVM - Fault-Specific False Alarm Rate")

plt.grid(axis="y", alpha=0.3)

save_figure("12_one_class_svm_faultwise_false_alarm_rate.png")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED")
print("=" * 65)

print(f"\nAll figures saved in:\n{FIGURE_DIR}")

print("\nGenerated figures:")
for path in sorted(FIGURE_DIR.glob("*.png")):
    if path.name.startswith(("07_", "08_", "09_", "10_", "11_", "12_")):
        print(f" - {path.name}")

print("\nNote: ROC-AUC values are sampled estimates.")
