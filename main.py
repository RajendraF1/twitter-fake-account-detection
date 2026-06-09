from pathlib import Path
import joblib
from src.data_loader import load_train_test_data
from src.preprocessing import fit_preprocessor, transform_preprocessor
from src.scaling import (
    fit_scaler,
    transform_scaler,
    combine_scaled_features,
    save_scaler
)
from src.model import (
    prepare_train_data,
    prepare_test_data,
    train_model,
    save_model
)
from src.evaluation import (
    evaluate_train_performance,
    evaluate_test_performance,
    evaluate_cross_validation,
    extract_feature_weights,
    build_evaluation_summary,
    save_text_report,
    save_feature_weights
)
from src.visualization import (
    save_confusion_matrix_plot,
    save_feature_weights_plot,
    save_cv_scores_plot,
    save_feature_distribution_plot
)


TRAIN_PATH = "data/raw/train_data_eps1.xlsx"
TEST_PATH = "data/raw/test_data_eps1.xlsx"

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def create_directories() -> None:
    """
    Create output directories if they do not already exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """
    Load raw training and testing data.
    """
    print("Loading raw datasets...")
    df_train, df_test = load_train_test_data(TRAIN_PATH, TEST_PATH)

    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    return df_train, df_test


def run_preprocessing(df_train, df_test):
    """
    Run preprocessing on train and test data.
    """
    print("\nRunning preprocessing on training data...")
    df_train_processed, lang_mapping = fit_preprocessor(df_train)
    joblib.dump(lang_mapping, MODELS_DIR / "lang_mapping.pkl")

    print("Running preprocessing on testing data...")
    df_test_processed = transform_preprocessor(df_test, lang_mapping)

    df_train_processed.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    df_test_processed.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)

    print("Preprocessed datasets have been saved successfully.")

    return df_train_processed, df_test_processed, lang_mapping


def run_scaling(df_train_processed, df_test_processed):
    """
    Scale selected features using MinMaxScaler.
    """
    print("\nScaling training features...")
    scaler, X_train_scaled = fit_scaler(df_train_processed)

    print("Scaling testing features...")
    X_test_scaled = transform_scaler(df_test_processed, scaler)

    df_train_final = combine_scaled_features(df_train_processed, X_train_scaled)
    df_test_final = combine_scaled_features(df_test_processed, X_test_scaled)

    df_train_final.to_csv(PROCESSED_DIR / "train_scaled.csv", index=False)
    df_test_final.to_csv(PROCESSED_DIR / "test_scaled.csv", index=False)

    save_scaler(scaler, MODELS_DIR / "scaler.pkl")

    print("Scaled datasets and scaler have been saved successfully.")

    return df_train_final, df_test_final, scaler


def run_training(df_train_final, df_test_final):
    """
    Prepare data and train the SLP model.
    """
    print("\nPreparing training data...")
    X_train, y_train = prepare_train_data(df_train_final)

    print("Preparing testing data...")
    X_test, y_test = prepare_test_data(df_test_final)

    print("Training the SLP model...")
    model = train_model(X_train, y_train)
    save_model(model, MODELS_DIR / "slp_model.pkl")

    print("Model training completed successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return model, X_train, y_train, X_test, y_test


def run_evaluation(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train/test data and cross-validation.
    """
    print("\nEvaluating performance on training data...")
    train_results = evaluate_train_performance(model, X_train, y_train)

    print("Running cross-validation...")
    cv_results = evaluate_cross_validation(model, X_train, y_train, cv=5)

    test_results = None
    if y_test is not None:
        print("Evaluating performance on testing data...")
        test_results = evaluate_test_performance(model, X_test, y_test)
    else:
        print("No target column found in the test dataset. Test evaluation is skipped.")

    print("Extracting feature weights...")
    feature_weights_df = extract_feature_weights(model)

    summary_text = build_evaluation_summary(train_results, cv_results, test_results)

    save_text_report(summary_text, REPORTS_DIR / "metrics.txt")
    save_feature_weights(feature_weights_df, REPORTS_DIR / "feature_weights.csv")

    print("Evaluation results have been saved successfully.")

    return train_results, test_results, cv_results, feature_weights_df, summary_text


def run_visualization(train_results, test_results, cv_results, feature_weights_df, df_train_final):
    """
    Generate and save plots.
    """
    print("\nGenerating visualizations...")

    save_confusion_matrix_plot(
        train_results["train_confusion_matrix"],
        FIGURES_DIR / "train_confusion_matrix.png",
        title="Train Confusion Matrix"
    )

    if test_results is not None:
        save_confusion_matrix_plot(
            test_results["test_confusion_matrix"],
            FIGURES_DIR / "test_confusion_matrix.png",
            title="Test Confusion Matrix"
        )

    save_feature_weights_plot(
        feature_weights_df,
        FIGURES_DIR / "feature_weights.png"
    )

    save_cv_scores_plot(
        cv_results["cv_scores"],
        FIGURES_DIR / "cv_scores.png"
    )

    save_feature_distribution_plot(
        df_train_final,
        "account_age_days",
        FIGURES_DIR / "account_age_days_distribution.png"
    )

    print("All visualizations have been saved successfully.")


def print_final_summary(summary_text, feature_weights_df):
    """
    Print final summary to console.
    """
    print("\n===== FINAL EVALUATION SUMMARY =====")
    print(summary_text)

    print("\nTop feature weights:")
    print(feature_weights_df.head())

    print("\nPipeline execution completed successfully.")
    print(f"Processed data directory : {PROCESSED_DIR}")
    print(f"Models directory         : {MODELS_DIR}")
    print(f"Reports directory        : {REPORTS_DIR}")
    print(f"Figures directory        : {FIGURES_DIR}")


def main():
    try:
        create_directories()

        df_train, df_test = load_raw_data()

        df_train_processed, df_test_processed, _ = run_preprocessing(df_train, df_test)

        df_train_final, df_test_final, _ = run_scaling(df_train_processed, df_test_processed)

        model, X_train, y_train, X_test, y_test = run_training(df_train_final, df_test_final)

        train_results, test_results, cv_results, feature_weights_df, summary_text = run_evaluation(
            model, X_train, y_train, X_test, y_test
        )

        run_visualization(
            train_results,
            test_results,
            cv_results,
            feature_weights_df,
            df_train_final
        )

        print_final_summary(summary_text, feature_weights_df)

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()