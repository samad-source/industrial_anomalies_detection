
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPORT FILES
# ============================================================

METRICS_FILES = {
    "Isolation Forest": REPORT_DIR / "isolation_forest_run_based_metrics.json",
    "One-Class SVM": REPORT_DIR / "one_class_svm_run_based_metrics.json",
    "Autoencoder": REPORT_DIR / "autoencoder_evaluation.json"
}

FAULTWISE_FILES = {
    "Isolation Forest": REPORT_DIR / "isolation_forest_run_based_faultwise.csv",
    "One-Class SVM": REPORT_DIR / "one_class_svm_run_based_faultwise.csv",
    "Autoencoder": REPORT_DIR / "autoencoder_faultwise.csv"
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_value(data, aliases, default=np.nan):

    normalized = {
        normalize_key(key): value
        for key, value in data.items()
    }

    for alias in aliases:
        key = normalize_key(alias)

        if key in normalized:
            return normalized[key]

    return default


def calculate_metrics(row):

    tn = row["tn"]
    fp = row["fp"]
    fn = row["fn"]
    tp = row["tp"]

    total = tn + fp + fn + tp

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else np.nan

    row["precision"] = safe_divide(tp, tp + fp)
    row["recall"] = safe_divide(tp, tp + fn)

    row["f1"] = safe_divide(
        2 * row["precision"] * row["recall"],
        row["precision"] + row["recall"]
    )

    row["specificity"] = safe_divide(tn, tn + fp)
    row["false_alarm_rate"] = safe_divide(fp, tn + fp)
    row["accuracy"] = safe_divide(tp + tn, total)

    return row


def load_overall_metrics(model_name, path):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    row = {
        "Model": model_name,

        "tn": get_value(
            data,
            ["true_negatives", "tn"]
        ),

        "fp": get_value(
            data,
            ["false_positives", "fp"]
        ),

        "fn": get_value(
            data,
            ["false_negatives", "fn"]
        ),

        "tp": get_value(
            data,
            ["true_positives", "tp"]
        ),

        "roc_auc": get_value(
            data,
            ["roc_auc_sampled", "roc_auc", "auc"]
        ),

        "roc_auc_sample_size": get_value(
            data,
            ["roc_auc_sample_size"]
        ),

        "total_observations": get_value(
            data,
            ["total_observations"]
        ),

        "normal_observations": get_value(
            data,
            ["normal_observations"]
        ),

        "anomalous_observations": get_value(
            data,
            ["anomalous_observations"]
        )
    }

    row = calculate_metrics(row)

    return row


def load_faultwise(path, model_name):

    df = pd.read_csv(path)

    df.columns = [
        normalize_key(column)
        for column in df.columns
    ]

    aliases = {
        "fault_number": ["faultnumber", "fault", "faultid"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1": ["f1", "f1score", "f1measure"],
        "specificity": ["specificity"],
        "false_alarm_rate": [
            "falsealarmrate",
            "far",
            "fpr"
        ]
    }

    result = pd.DataFrame()

    for target, candidates in aliases.items():

        matched = next(
            (
                candidate
                for candidate in candidates
                if candidate in df.columns
            ),
            None
        )

        if matched is not None:
            result[target] = df[matched]

    if "fault_number" not in result.columns:
        raise ValueError(
            f"Fault number column missing in {path}"
        )

    result = result.rename(
        columns={
            column: f"{model_name}_{column}"
            for column in result.columns
            if column != "fault_number"
        }
    )

    return result


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("=" * 70)
print("RUN-BASED THREE-MODEL COMPARISON")
print("=" * 70)

required_files = (
    list(METRICS_FILES.values())
    + list(FAULTWISE_FILES.values())
)

missing = [
    path for path in required_files
    if not path.exists()
]

if missing:

    for path in missing:
        print(f"Missing: {path}")

    raise FileNotFoundError(
        "One or more evaluation reports are missing."
    )

print("\nAll six evaluation reports found.")


# ============================================================
# 5. LOAD AND CALCULATE OVERALL METRICS
# ============================================================

print("\nLoading overall metrics...")

overall_rows = []

for model_name, path in METRICS_FILES.items():

    row = load_overall_metrics(model_name, path)

    overall_rows.append(row)

    print(f"Loaded: {model_name}")

overall = pd.DataFrame(overall_rows)


# ============================================================
# 6. VALIDATE CONFUSION MATRIX COUNTS
# ============================================================

print("\nValidating confusion matrix counts...")

for _, row in overall.iterrows():

    total = row["tn"] + row["fp"] + row["fn"] + row["tp"]

    if total != row["total_observations"]:

        raise ValueError(
            f"Confusion matrix total mismatch for {row['Model']}: "
            f"{total} != {row['total_observations']}"
        )

    print(
        f"{row['Model']}: "
        f"{int(total):,} observations verified."
    )


# ============================================================
# 7. SAVE OVERALL COMPARISON TABLE
# ============================================================

overall_csv = REPORT_DIR / "run_based_model_comparison.csv"

overall.to_csv(
    overall_csv,
    index=False
)

print("\nOverall comparison saved:")
print(overall_csv)


# ============================================================
# 8. SAVE PUBLICATION-FRIENDLY RESULTS TABLE
# ============================================================

publication_columns = [
    "Model",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_alarm_rate",
    "accuracy",
    "roc_auc"
]

publication_table = overall[
    publication_columns
].copy()

percentage_columns = [
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_alarm_rate",
    "accuracy",
    "roc_auc"
]

for column in percentage_columns:
    publication_table[column] = (
        publication_table[column] * 100
    ).round(2)

publication_table = publication_table.rename(
    columns={
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1": "F1-score (%)",
        "specificity": "Specificity (%)",
        "false_alarm_rate": "False Alarm Rate (%)",
        "accuracy": "Accuracy (%)",
        "roc_auc": "ROC-AUC (%)"
    }
)

publication_csv = (
    REPORT_DIR / "publication_overall_results.csv"
)

publication_table.to_csv(
    publication_csv,
    index=False
)

print("\nPublication results table:")
print(publication_table.to_string(index=False))

print("\nPublication table saved:")
print(publication_csv)


# ============================================================
# 9. CONFUSION MATRIX RESULTS TABLE
# ============================================================

confusion_table = overall[
    ["Model", "tn", "fp", "fn", "tp"]
].copy()

confusion_table = confusion_table.rename(
    columns={
        "tn": "True Negatives",
        "fp": "False Positives",
        "fn": "False Negatives",
        "tp": "True Positives"
    }
)

confusion_csv = REPORT_DIR / "confusion_matrix_comparison.csv"

confusion_table.to_csv(
    confusion_csv,
    index=False
)

print("\nConfusion matrix comparison:")
print(confusion_table.to_string(index=False))

print("\nConfusion matrix table saved:")
print(confusion_csv)


# ============================================================
# 10. LOAD FAULTWISE RESULTS
# ============================================================

print("\nLoading faultwise results...")

faultwise_frames = []

for model_name, path in FAULTWISE_FILES.items():

    df = load_faultwise(path, model_name)

    faultwise_frames.append(df)

    print(f"Loaded: {model_name}")

faultwise = faultwise_frames[0]

for df in faultwise_frames[1:]:

    faultwise = faultwise.merge(
        df,
        on="fault_number",
        how="outer"
    )

faultwise = faultwise.sort_values(
    "fault_number"
).reset_index(drop=True)


# ============================================================
# 11. SAVE FAULTWISE COMPARISON TABLE
# ============================================================

faultwise_csv = (
    REPORT_DIR / "run_based_faultwise_comparison.csv"
)

faultwise.to_csv(
    faultwise_csv,
    index=False
)

faultwise_publication = faultwise.copy()

for column in faultwise_publication.columns:

    if column != "fault_number":

        faultwise_publication[column] = (
            faultwise_publication[column] * 100
        ).round(2)

faultwise_publication = faultwise_publication.rename(
    columns={
        "fault_number": "Fault Number"
    }
)

faultwise_publication_csv = (
    REPORT_DIR / "publication_faultwise_results.csv"
)

faultwise_publication.to_csv(
    faultwise_publication_csv,
    index=False
)

print("\nFaultwise recall comparison:")

recall_columns = [
    "fault_number",
    "Isolation Forest_recall",
    "One-Class SVM_recall",
    "Autoencoder_recall"
]

print(
    faultwise[
        recall_columns
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ============================================================
# 12. FIGURE 19 — OVERALL PERFORMANCE
# ============================================================

print("\nGenerating Figure 19...")

plot_metrics = [
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc"
]

plot_data = overall.melt(
    id_vars="Model",
    value_vars=plot_metrics,
    var_name="Metric",
    value_name="Score"
)

plot_data["Metric"] = plot_data["Metric"].replace(
    {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "specificity": "Specificity",
        "roc_auc": "ROC-AUC"
    }
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=plot_data,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title(
    "Overall Performance Comparison of Three Algorithms",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Evaluation Metric")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Algorithm")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "19_run_based_overall_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ============================================================
# 13. FIGURE 20 — FAULTWISE RECALL
# ============================================================

print("Generating Figure 20...")

recall_plot = faultwise.melt(
    id_vars="fault_number",
    value_vars=[
        "Isolation Forest_recall",
        "One-Class SVM_recall",
        "Autoencoder_recall"
    ],
    var_name="Model",
    value_name="Recall"
)

recall_plot["Model"] = recall_plot["Model"].str.replace(
    "_recall",
    "",
    regex=False
)

plt.figure(figsize=(16, 7))

sns.barplot(
    data=recall_plot,
    x="fault_number",
    y="Recall",
    hue="Model"
)

plt.title(
    "Run-Based Faultwise Recall Comparison",
    fontsize=15,
    fontweight="bold",
    pad=35
)

plt.xlabel("TEP Fault Number")
plt.ylabel("Recall")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)

plt.legend(
    title="Algorithm",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "20_run_based_faultwise_recall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. FIGURE 21 — FAULTWISE F1-SCORE
# ============================================================

print("Generating Figure 21...")

f1_plot = faultwise.melt(
    id_vars="fault_number",
    value_vars=[
        "Isolation Forest_f1",
        "One-Class SVM_f1",
        "Autoencoder_f1"
    ],
    var_name="Model",
    value_name="F1-score"
)

f1_plot["Model"] = f1_plot["Model"].str.replace(
    "_f1",
    "",
    regex=False
)

plt.figure(figsize=(16, 7))

sns.barplot(
    data=f1_plot,
    x="fault_number",
    y="F1-score",
    hue="Model"
)

plt.title(
    "Run-Based Faultwise F1-score Comparison",
    fontsize=15,
    fontweight="bold",
    pad=35
)

plt.xlabel("TEP Fault Number")
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)

plt.legend(
    title="Algorithm",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=3,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "21_run_based_faultwise_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ============================================================
# 16. FIGURE 23 — FAULTWISE RECALL HEATMAP
# ============================================================

print("Generating Figure 23...")

heatmap_data = faultwise[
    [
        "fault_number",
        "Isolation Forest_recall",
        "One-Class SVM_recall",
        "Autoencoder_recall"
    ]
].copy()

heatmap_data = heatmap_data.set_index(
    "fault_number"
)

heatmap_data.columns = [
    "Isolation Forest",
    "One-Class SVM",
    "Autoencoder"
]

plt.figure(figsize=(9, 10))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    linewidths=0.5
)

plt.title(
    "Faultwise Recall Heatmap",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("TEP Fault Number")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "23_run_based_recall_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("THREE-ALGORITHM COMPARISON COMPLETED")
print("=" * 70)

print("\nGenerated result tables:")

for path in [
    overall_csv,
    publication_csv,
    confusion_csv,
    faultwise_csv,
    faultwise_publication_csv
]:
    print(f" - {path.name}")

print("\nGenerated figures:")

for number in range(19, 24):

    for path in FIGURE_DIR.glob(
        f"{number:02d}_run_based_*.png"
    ):
        print(f" - {path.name}")

print("\nROC-AUC values and confusion counts loaded from source reports.")
print("Existing preliminary figures 07–12 remain untouched.")
