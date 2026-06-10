from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

from src.scaling import FEATURE_COLUMNS


def evaluate_train_performance(model, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Menghitung performa model pada data train.
    """
    y_train_pred = model.predict(X_train)

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_classification_report": classification_report(y_train, y_train_pred, zero_division=0),
        "train_confusion_matrix": confusion_matrix(y_train, y_train_pred)
    }

    return results


def evaluate_test_performance(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Menghitung performa model pada data test.
    """
    y_test_pred = model.predict(X_test)

    results = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_classification_report": classification_report(y_test, y_test_pred, zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred)
    }

    return results


def evaluate_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> dict:
    """
    Menghitung cross-validation accuracy.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    results = {
        "cv_scores": cv_scores,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std()
    }

    return results


def extract_feature_weights(model) -> pd.DataFrame:
    if hasattr(model, "coefs_"):
        weights = model.coefs_[0]
        if weights.ndim == 2 and weights.shape[1] == 1:
            weights_flat = weights[:, 0]
        else:
            weights_flat = abs(weights).mean(axis=1)
    elif hasattr(model, "coef_"):
        weights_flat = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        weights_flat = model.feature_importances_
    else:
        weights_flat = [0] * len(FEATURE_COLUMNS)

    feature_weights_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "weight": weights_flat,
        "abs_weight": abs(weights_flat)
    }).sort_values(by="abs_weight", ascending=False)

    return feature_weights_df


def save_text_report(text: str, path: str) -> None:
    """
    Menyimpan laporan teks ke file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_feature_weights(feature_weights_df: pd.DataFrame, path: str) -> None:
    """
    Menyimpan bobot fitur ke CSV.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_weights_df.to_csv(output_path, index=False)


def build_evaluation_summary(
    train_results: dict,
    cv_results: dict,
    test_results: dict = None,
    model_name: str = "Model"
) -> str:
    lines = []

    lines.append(f"=== HASIL EVALUASI MODEL {model_name.upper()} ===\n")

    lines.append(f"Train Accuracy: {train_results['train_accuracy']:.4f}")
    lines.append(f"CV Mean Accuracy: {cv_results['cv_mean_accuracy']:.4f}")
    lines.append(f"CV Std Accuracy: {cv_results['cv_std_accuracy']:.4f}\n")

    lines.append("Train Classification Report:")
    lines.append(train_results["train_classification_report"])
    lines.append("Train Confusion Matrix:")
    lines.append(str(train_results["train_confusion_matrix"]))
    lines.append("")

    if test_results is not None:
        lines.append(f"Test Accuracy: {test_results['test_accuracy']:.4f}\n")
        lines.append("Test Classification Report:")
        lines.append(test_results["test_classification_report"])
        lines.append("Test Confusion Matrix:")
        lines.append(str(test_results["test_confusion_matrix"]))
        lines.append("")

    lines.append("Cross Validation Scores:")
    lines.append(str(cv_results["cv_scores"]))

    return "\n".join(lines)