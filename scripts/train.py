import argparse
import joblib
from pathlib import Path
from src.config import load_config
from src.data_loader import load_train_test_data
from src.preprocessing import fit_preprocessor, transform_preprocessor
from src.scaling import fit_scaler, transform_scaler, combine_scaled_features, save_scaler
from src.model import prepare_train_data, prepare_test_data, save_model
from src.evaluation import (
    evaluate_train_performance,
    evaluate_cross_validation,
    evaluate_test_performance,
    extract_feature_weights,
    build_evaluation_summary,
    save_text_report,
    save_feature_weights,
)
from src.visualization import (
    save_confusion_matrix_plot,
    save_feature_weights_plot,
    save_cv_scores_plot,
)
from src.pipeline import build_model, MODEL_REGISTRY


def train_model_on_data(model_name, df_train_final, df_test_final, models_dir, reports_dir, figures_dir, cv_folds):
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    model = build_model(model_name, load_config().get("model", {}))
    X_train, y_train = prepare_train_data(df_train_final)

    has_target = "fake" in df_test_final.columns
    X_test, y_test = prepare_test_data(df_test_final) if has_target else (None, None)

    model.fit(X_train, y_train)
    model_path = models_dir / f"{model_name}_model.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    train_results = evaluate_train_performance(model, X_train, y_train)
    cv_results = evaluate_cross_validation(model, X_train, y_train, cv=cv_folds)
    test_results = evaluate_test_performance(model, X_test, y_test) if y_test is not None else None
    feature_weights_df = extract_feature_weights(model)
    summary = build_evaluation_summary(train_results, cv_results, test_results, model_name)
    save_text_report(summary, reports_dir / f"{model_name}_metrics.txt")
    save_feature_weights(feature_weights_df, reports_dir / f"{model_name}_feature_weights.csv")

    save_confusion_matrix_plot(
        train_results["train_confusion_matrix"],
        figures_dir / f"{model_name}_train_confusion_matrix.png",
        title=f"{model_name} - Train Confusion Matrix",
    )
    if test_results is not None:
        save_confusion_matrix_plot(
            test_results["test_confusion_matrix"],
            figures_dir / f"{model_name}_test_confusion_matrix.png",
            title=f"{model_name} - Test Confusion Matrix",
        )
    save_feature_weights_plot(
        feature_weights_df, figures_dir / f"{model_name}_feature_weights.png"
    )
    save_cv_scores_plot(
        cv_results["cv_scores"], figures_dir / f"{model_name}_cv_scores.png"
    )

    print(summary)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train one or all models"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train (default: train all models)",
    )
    args = parser.parse_args()

    config = load_config()
    data_cfg = config["data"]
    paths = config["paths"]
    training_cfg = config.get("training", {})

    processed_dir = Path(paths["processed_dir"])
    models_dir = Path(paths["models_dir"])
    reports_dir = Path(paths["reports_dir"])
    figures_dir = Path(paths["figures_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reference_date = data_cfg.get("reference_date", "2025-12-02")

    df_train, df_test = load_train_test_data(
        data_cfg["train_path"], data_cfg["test_path"]
    )

    df_train_processed, lang_mapping = fit_preprocessor(
        df_train, reference_date
    )
    joblib.dump(lang_mapping, models_dir / "lang_mapping.pkl")

    scaler, X_train_scaled = fit_scaler(df_train_processed)
    df_train_final = combine_scaled_features(df_train_processed, X_train_scaled)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train_final.to_csv(processed_dir / "train_scaled.csv", index=False)
    save_scaler(scaler, models_dir / "scaler.pkl")

    df_test_processed = transform_preprocessor(
        df_test, lang_mapping, reference_date
    )
    X_test_scaled = transform_scaler(df_test_processed, scaler)
    df_test_final = combine_scaled_features(df_test_processed, X_test_scaled)
    df_test_final.to_csv(processed_dir / "test_scaled.csv", index=False)

    cv_folds = training_cfg.get("cv_folds", 5)

    if args.model:
        models_to_train = [args.model]
    else:
        models_to_train = list(MODEL_REGISTRY.keys())

    for model_name in models_to_train:
        train_model_on_data(
            model_name,
            df_train_final,
            df_test_final,
            models_dir,
            reports_dir,
            figures_dir,
            cv_folds,
        )


if __name__ == "__main__":
    main()
