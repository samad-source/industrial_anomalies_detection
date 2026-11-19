
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = REPORT_DIR / "isolation_forest_metrics.json"
FAULT_FILE = REPORT_DIR / "isolation_forest_faultwise.csv"
SCORE_FILE = REPORT_DIR / "isolation_forest_score_diagnostics.csv"


# ==================================================
# 2. LOAD RESULTS
# ==================================================

print("=" * 60)
print("ISOLATION FOREST VISUALIZATION")
print("=" * 60)

with open(METRICS_FILE, "r", encoding="utf-8") as file:
    metrics = json.load(file)

fault_df = pd.read_csv(FAULT_FILE)
score_df = pd.read_csv(SCORE_FILE)

threshold = metrics["threshold"]

print("\nEvaluation reports loaded successfully.")


# ==================================================
# 3. CONFUSION MATRIX
# ==================================================

tn = metrics["true_negatives"]
fp = metrics["false_positives"]
fn = metrics["false_negatives"]
tp = metrics["true_positives"]

cm = np.array([
    [tn, fp],
    [fn, tp]
])

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format=",d",
    colorbar=True
)

ax.set_title("Isolation Forest - Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "01_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 01_confusion_matrix.png")


# ==================================================
# 4. NORMALIZED CONFUSION MATRIX
# ==================================================

row_totals = cm.sum(axis=1, keepdims=True)

normalized_cm = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=normalized_cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2%",
    colorbar=True
)

ax.set_title("Isolation Forest - Normalized Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "02_normalized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 02_normalized_confusion_matrix.png")


# ==================================================
# 5. FAULT-SPECIFIC METRICS
# ==================================================

fault_df = fault_df.sort_values("fault_number")

x = np.arange(len(fault_df))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 7))

ax.bar(
    x - width,
    fault_df["precision"],
    width,
    label="Precision"
)

ax.bar(
    x,
    fault_df["recall"],
    width,
    label="Recall"
)

ax.bar(
    x + width,
    fault_df["f1_score"],
    width,
    label="F1-score"
)

ax.set_title("Isolation Forest - Performance Across 20 Fault Types")
ax.set_xlabel("Fault Number")
ax.set_ylabel("Score")

ax.set_xticks(x)
ax.set_xticklabels(fault_df["fault_number"])

ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "03_faultwise_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 03_faultwise_metrics.png")


# ==================================================
# 6. FAULT-SPECIFIC ROC-AUC
# ==================================================

score_df = score_df.sort_values("fault_number")

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    score_df["fault_number"].astype(str),
    score_df["roc_auc"]
)

ax.axhline(
    y=0.5,
    linestyle="--",
    label="Random-ranking reference (0.50)"
)

ax.set_title("Isolation Forest - ROC-AUC by Fault Type")
ax.set_xlabel("Fault Number")
ax.set_ylabel("ROC-AUC")

ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "04_faultwise_roc_auc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 04_faultwise_roc_auc.png")


# ==================================================
# 7. NORMAL VS ANOMALY SCORE COMPARISON
# ==================================================

x = np.arange(len(score_df))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 7))

ax.bar(
    x - width / 2,
    score_df["normal_mean_score"],
    width,
    label="Normal observations"
)

ax.bar(
    x + width / 2,
    score_df["anomaly_mean_score"],
    width,
    label="Anomalous observations"
)

ax.axhline(
    y=threshold,
    linestyle="--",
    label=f"Detection threshold ({threshold:.4f})"
)

ax.set_title("Mean Anomaly Scores Across Fault Types")
ax.set_xlabel("Fault Number")
ax.set_ylabel("Mean Anomaly Score")

ax.set_xticks(x)
ax.set_xticklabels(score_df["fault_number"])

ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "05_anomaly_score_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 05_anomaly_score_comparison.png")


# ==================================================
# 8. OVERALL PERFORMANCE SUMMARY
# ==================================================

overall_names = [
    "Precision",
    "Recall",
    "F1-score",
    "Specificity",
    "ROC-AUC"
]

overall_values = [
    metrics["precision"],
    metrics["recall"],
    metrics["f1_score"],
    metrics["specificity"],
    metrics["roc_auc_sampled"]
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    overall_names,
    overall_values
)

ax.set_title("Overall Isolation Forest Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)

ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, overall_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "06_overall_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: 06_overall_performance.png")


# ==================================================
# 9. COMPLETION
# ==================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print(f"\nAll six figures saved in:\n{FIGURE_DIR}")
