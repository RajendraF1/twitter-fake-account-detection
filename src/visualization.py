from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_confusion_matrix_plot(conf_matrix, path: str, title: str = "Confusion Matrix") -> None:
    """
    Menyimpan confusion matrix sebagai gambar.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_matrix, interpolation="nearest")
    fig.colorbar(im)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_weights_plot(feature_weights_df: pd.DataFrame, path: str, top_n: int = None) -> None:
    """
    Menyimpan bar chart bobot fitur.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = feature_weights_df.copy().sort_values("abs_weight", ascending=True)

    if top_n is not None:
        df_plot = df_plot.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["feature"], df_plot["abs_weight"])
    ax.set_title("Feature Importance Based on Absolute Weights")
    ax.set_xlabel("Absolute Weight")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_cv_scores_plot(cv_scores, path: str) -> None:
    """
    Menyimpan line plot cross-validation scores.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    folds = np.arange(1, len(cv_scores) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(folds, cv_scores, marker="o")
    ax.set_title("Cross-Validation Accuracy per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(folds)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_distribution_plot(
    df: pd.DataFrame,
    column: str,
    path: str,
    bins: int = 20
) -> None:
    """
    Menyimpan histogram distribusi satu fitur.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[column].dropna(), bins=bins)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()