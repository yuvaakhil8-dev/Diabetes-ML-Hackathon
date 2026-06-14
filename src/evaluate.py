# src/evaluate.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Evaluation â€” Metrics, Plots, and Comparisons
#
# This file contains all visualization and reporting functions
# used both in train.py (training run) and dashboard.py (UI).
#
# Functions:
#   plot_confusion_matrix      â†’ heatmap for any classifier
#   plot_information_gain      â†’ bar chart of IG per feature
#   plot_k_vs_accuracy         â†’ line chart of KNN accuracy across K
#   plot_model_comparison      â†’ bar chart comparing all models' F1
#   plot_risk_score_histogram  â†’ distribution of Stage 1 risk scores
#   print_metrics_table        â†’ console summary of all results
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend â€” safe for Streamlit
import seaborn as sns
import io


# â”€â”€ Shared style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PALETTE = {
    "teal"   : "#1D9E75",
    "purple" : "#7F77DD",
    "blue"   : "#378ADD",
    "amber"  : "#BA7517",
    "coral"  : "#D85A30",
    "gray"   : "#888780"
}


def _fig_to_bytes(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes for Streamlit st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> bytes:
    """
    Render a confusion matrix as a labeled heatmap.

    Rows = Actual class, Columns = Predicted class
    TP (top-left when negative=0): correctly predicted non-diabetic
    FP: predicted diabetic but actually not
    FN: predicted non-diabetic but actually diabetic
    TN: correctly predicted diabetic
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: No", "Predicted: Yes"],
        yticklabels=["Actual: No",    "Actual: Yes"],
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_information_gain(ig_scores: dict, title: str = "Information Gain per Feature") -> bytes:
    """
    Horizontal bar chart of Information Gain for each feature.
    Features are sorted highest to lowest.
    """
    features = list(ig_scores.keys())
    values   = list(ig_scores.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features[::-1], values[::-1],
                   color=PALETTE["teal"], edgecolor="white", height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, color="#333")

    ax.set_xlabel("Information Gain", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_k_vs_accuracy(k_results: dict) -> bytes:
    """
    Line chart comparing accuracy of KNN vs Weighted KNN across K values.
    Helps the panel visualize why a particular K was chosen.
    """
    k_vals       = k_results["k_values"]
    knn_acc      = k_results["knn_accuracy"]
    wknn_acc     = k_results["wknn_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, knn_acc,  marker="o", label="KNN (uniform)",
            color=PALETTE["blue"],   linewidth=2.5, markersize=8)
    ax.plot(k_vals, wknn_acc, marker="s", label="Weighted KNN (distance)",
            color=PALETTE["purple"], linewidth=2.5, markersize=8)

    ax.set_xlabel("K (number of neighbors)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("KNN vs Weighted KNN â€” Accuracy across K values", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(k_vals)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_model_comparison(all_results: list, metric: str = "f1") -> bytes:
    """
    Bar chart comparing a chosen metric across all models.
    all_results: list of result dicts from each stage.
    metric: one of 'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    """
    # Filter only results that have the requested metric
    names  = [r["model_name"] for r in all_results if metric in r]
    values = [r[metric]       for r in all_results if metric in r]

    colors = [
        PALETTE["teal"],
        PALETTE["purple"], PALETTE["purple"],
        PALETTE["blue"],   PALETTE["blue"]
    ][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    metric_label = metric.upper().replace("_", " ")
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Model Comparison â€” {metric_label}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", labelrotation=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_risk_score_histogram(scores: np.ndarray, y_true: np.ndarray,
                              threshold: float = 0.5) -> bytes:
    """
    Distribution of Stage 1 risk scores, colored by actual outcome.
    Shows how well the regression score separates the two classes.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(scores[y_true == 0], bins=20, alpha=0.65,
            color=PALETTE["teal"],  label="Non-Diabetic (actual)", edgecolor="white")
    ax.hist(scores[y_true == 1], bins=20, alpha=0.65,
            color=PALETTE["coral"], label="Diabetic (actual)",      edgecolor="white")
    ax.axvline(x=threshold, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold = {threshold}")

    ax.set_xlabel("Risk Score", fontsize=11)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_title("Stage 1 â€” Risk Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def print_metrics_table(all_results: list):
    """
    Print a clean ASCII table of all model metrics to the console.
    Used during training run in train.py.
    """
    print("\n" + "â•" * 78)
    print(f"{'MODEL':<30} {'ACC':>6} {'PREC':>6} {'REC':>6} {'F1':>6} {'AUC':>7}")
    print("â•" * 78)
    for r in all_results:
        if "accuracy" in r:
            print(
                f"{r['model_name']:<30} "
                f"{r['accuracy']:>6.4f} "
                f"{r.get('precision', 0):>6.4f} "
                f"{r.get('recall', 0):>6.4f} "
                f"{r.get('f1', 0):>6.4f} "
                f"{r.get('roc_auc', 0):>7.4f}"
            )
    print("â•" * 78 + "\n")
