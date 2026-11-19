
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_PATH = REPORT_DIR / "autoencoder_evaluation.json"
FAULTWISE_PATH = REPORT_DIR / "autoencoder_faultwise.csv"
HISTORY_PATH = REPORT_DIR / "autoencoder_training_history.csv"

IF_PATH = REPORT_DIR / "isolation_forest_faultwise.csv"
SVM_PATH = REPORT_DIR / "one_class_svm_faultwise.csv"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})


# ============================================================
# 2. LOAD RESULTS
# ============================================================

with open(EVALUATION_PATH, "r") as file:
    evaluation = json.load(file)

faultwise = pd.read_csv(FAULTWISE_PATH)

history = pd.read_csv(HISTORY_PATH)

TN = evaluation["true_negatives"]
FP = evaluation["false_positives"]
FN = evaluation["false_negatives"]
TP = evaluation["true_positives"]

cm = np.array([
    [TN, FP],
    [FN, TP]
])

print("=" * 65)
print("AUTOENCODER VISUALIZATION")
print("=" * 65)


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomalous"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format=","
)

ax.set_title(
    "Autoencoder - Confusion Matrix",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "13_autoencoder_confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("Saved confusion matrix.")


# ============================================================
# 4. NORMALIZED CONFUSION MATRIX
# ============================================================

row_sums = cm.sum(axis=1, keepdims=True)

normalized_cm = cm / row_sums

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=normalized_cm,
    display_labels=["Normal", "Anomalous"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format=".2%"
)

ax.set_title(
    "Autoencoder - Normalized Confusion Matrix",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "14_autoencoder_normalized_confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("Saved normalized confusion matrix.")


# ============================================================
# 5. FAULT-SPECIFIC METRICS
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6))

x = np.arange(len(faultwise))

width = 0.25

ax.bar(
    x - width,
    faultwise["precision"],
    width,
    label="Precision"
)

ax.bar(
    x,
    faultwise["recall"],
    width,
    label="Recall"
)

ax.bar(
    x + width,
    faultwise["f1_score"],
    width,
    label="F1-score"
)

ax.set_title(
    "Autoencoder - Fault-Specific Performance",
    fontweight="bold"
)

ax.set_xlabel("Fault Number")
ax.set_ylabel("Score")

ax.set_xticks(x)
ax.set_xticklabels(faultwise["fault_number"])

ax.set_ylim(0, 1.05)

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "15_autoencoder_faultwise_metrics.png",
    bbox_inches="tight"
)

plt.close()

print("Saved fault-specific metrics.")


# ============================================================
# 6. FAULT-SPECIFIC RECALL RANKING
# ============================================================

ranked = faultwise.sort_values(
    "recall",
    ascending=True
)

fig, ax = plt.subplots(figsize=(11, 8))

ax.barh(
    ranked["fault_number"].astype(str),
    ranked["recall"]
)

ax.set_title(
    "Autoencoder - Fault Detection Recall Ranking",
    fontweight="bold"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Fault Number")

ax.set_xlim(0, 1.05)

ax.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "16_autoencoder_faultwise_recall.png",
    bbox_inches="tight"
)

plt.close()

print("Saved fault recall ranking.")


# ============================================================
# 7. TRAINING HISTORY
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    history["epoch"],
    history["training_loss"],
    marker="o",
    label="Training Loss"
)

ax.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

ax.set_title(
    "Autoencoder Training and Validation Loss",
    fontweight="bold"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Squared Error")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "17_autoencoder_training_history.png",
    bbox_inches="tight"
)

plt.close()

print("Saved training history.")


# ============================================================
# 8. COMPARE FAULT-WISE RECALL ACROSS THREE MODELS
# ============================================================

if IF_PATH.exists() and SVM_PATH.exists():

    isolation_forest = pd.read_csv(IF_PATH)
    one_class_svm = pd.read_csv(SVM_PATH)

    comparison = faultwise[
        ["fault_number", "recall"]
    ].rename(
        columns={"recall": "Autoencoder"}
    )

    comparison = comparison.merge(
        isolation_forest[
            ["fault_number", "recall"]
        ].rename(
            columns={"recall": "Isolation Forest"}
        ),
        on="fault_number"
    )

    comparison = comparison.merge(
        one_class_svm[
            ["fault_number", "recall"]
        ].rename(
            columns={"recall": "One-Class SVM"}
        ),
        on="fault_number"
    )

    comparison.plot(
        x="fault_number",
        y=[
            "Isolation Forest",
            "One-Class SVM",
            "Autoencoder"
        ],
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(
        "Preliminary Fault-Wise Recall Comparison",
        fontweight="bold"
    )

    plt.xlabel("Fault Number")
    plt.ylabel("Recall")

    plt.ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "18_three_model_faultwise_recall.png",
        bbox_inches="tight"
    )

    plt.close()

    print("Saved three-model recall comparison.")

else:

    print(
        "Three-model comparison skipped: "
        "one or more model reports are missing."
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED")
print("=" * 65)

print("Generated figures:")

for path in sorted(FIGURE_DIR.glob("*autoencoder*.png")):
    print(path.name)

if (FIGURE_DIR / "18_three_model_faultwise_recall.png").exists():
    print("18_three_model_faultwise_recall.png")

print(f"\nFigures saved in:\n{FIGURE_DIR}")
