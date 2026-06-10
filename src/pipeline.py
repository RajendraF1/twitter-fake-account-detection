from pathlib import Path
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import load_config
from src.data_loader import load_train_test_data
from src.preprocessing import fit_preprocessor, transform_preprocessor
from src.scaling import (
    fit_scaler,
    transform_scaler,
    combine_scaled_features,
    save_scaler,
    FEATURE_COLUMNS,
)
from src.model import prepare_train_data, prepare_test_data, save_model
from src.evaluation import (
    evaluate_train_performance,
    evaluate_test_performance,
    evaluate_cross_validation,
    extract_feature_weights,
    build_evaluation_summary,
    save_text_report,
    save_feature_weights,
)
from src.visualization import (
    save_confusion_matrix_plot,
    save_feature_weights_plot,
    save_cv_scores_plot,
    save_feature_distribution_plot,
)


MODEL_REGISTRY = {
    "slp": (MLPClassifier, "slp"),
    "logistic_regression": (LogisticRegression, "logistic_regression"),
    "random_forest": (RandomForestClassifier, "random_forest"),
    "xgboost": (XGBClassifier, "xgboost"),
}


def build_model(model_name: str, model_config: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )

    cls, key = MODEL_REGISTRY[model_name]
    params = dict(model_config.get(key, {}))

    if model_name == "slp":
        params["hidden_layer_sizes"] = tuple(params.get("hidden_layer_sizes", []))

    return cls(**params)


class Pipeline:
    def __init__(self, config_path=None):
        self.config = load_config() if config_path is None else load_config(config_path)
        self._set_paths()

    def _set_paths(self):
        paths = self.config["paths"]
        self.processed_dir = Path(paths["processed_dir"])
        self.models_dir = Path(paths["models_dir"])
        self.reports_dir = Path(paths["reports_dir"])
        self.figures_dir = Path(paths["figures_dir"])

    def _create_dirs(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run(self, train_path=None, test_path=None, model_name="slp"):
        data_cfg = self.config["data"]
        train_path = train_path or data_cfg["train_path"]
        test_path = test_path or data_cfg["test_path"]
        reference_date = data_cfg.get("reference_date", "2025-12-02")

        self._create_dirs()

        print("Loading raw datasets...")
        df_train, df_test = load_train_test_data(train_path, test_path)
        print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

        print("Preprocessing...")
        df_train_processed, lang_mapping = fit_preprocessor(
            df_train, reference_date
        )
        joblib.dump(lang_mapping, self.models_dir / "lang_mapping.pkl")
        df_test_processed = transform_preprocessor(
            df_test, lang_mapping, reference_date
        )
        df_train_processed.to_csv(
            self.processed_dir / "train_processed.csv", index=False
        )
        df_test_processed.to_csv(
            self.processed_dir / "test_processed.csv", index=False
        )

        print("Scaling features...")
        scaler, X_train_scaled = fit_scaler(df_train_processed)
        X_test_scaled = transform_scaler(df_test_processed, scaler)
        df_train_final = combine_scaled_features(
            df_train_processed, X_train_scaled
        )
        df_test_final = combine_scaled_features(
            df_test_processed, X_test_scaled
        )
        df_train_final.to_csv(
            self.processed_dir / "train_scaled.csv", index=False
        )
        df_test_final.to_csv(
            self.processed_dir / "test_scaled.csv", index=False
        )
        save_scaler(scaler, self.models_dir / "scaler.pkl")

        print(f"Training model '{model_name}'...")
        model = build_model(model_name, self.config.get("model", {}))
        X_train, y_train = prepare_train_data(df_train_final)
        X_test, y_test = prepare_test_data(df_test_final)
        model.fit(X_train, y_train)
        model_path = self.models_dir / f"{model_name}_model.pkl"
        save_model(model, model_path)
        print(f"Model saved to {model_path}")

        print("Evaluating...")
        train_results = evaluate_train_performance(model, X_train, y_train)
        cv_folds = self.config.get("training", {}).get("cv_folds", 5)
        cv_results = evaluate_cross_validation(
            model, X_train, y_train, cv=cv_folds
        )
        test_results = None
        if y_test is not None:
            test_results = evaluate_test_performance(model, X_test, y_test)
        feature_weights_df = extract_feature_weights(model)
        summary = build_evaluation_summary(
            train_results, cv_results, test_results, model_name
        )
        save_text_report(summary, self.reports_dir / f"{model_name}_metrics.txt")
        save_feature_weights(
            feature_weights_df, self.reports_dir / f"{model_name}_feature_weights.csv"
        )

        print("Generating visualizations...")
        save_confusion_matrix_plot(
            train_results["train_confusion_matrix"],
            self.figures_dir / f"{model_name}_train_confusion_matrix.png",
            title=f"{model_name} - Train Confusion Matrix",
        )
        if test_results is not None:
            save_confusion_matrix_plot(
                test_results["test_confusion_matrix"],
                self.figures_dir / f"{model_name}_test_confusion_matrix.png",
                title=f"{model_name} - Test Confusion Matrix",
            )
        save_feature_weights_plot(
            feature_weights_df, self.figures_dir / f"{model_name}_feature_weights.png"
        )
        save_cv_scores_plot(
            cv_results["cv_scores"], self.figures_dir / f"{model_name}_cv_scores.png"
        )
        save_feature_distribution_plot(
            df_train_final,
            "account_age_days",
            self.figures_dir / "account_age_days_distribution.png",
        )

        print("\nPipeline completed successfully.")
        print(summary)

        return model
