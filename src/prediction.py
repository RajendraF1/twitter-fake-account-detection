import joblib
import pandas as pd

from src.preprocessing import transform_preprocessor
from src.scaling import transform_scaler, combine_scaled_features, FEATURE_COLUMNS


def load_artifacts(
    model_path: str = "models/slp_model.pkl",
    scaler_path: str = "models/scaler.pkl",
    lang_mapping_path: str = "models/lang_mapping.pkl"
):
    """
    Load trained model, scaler, and language mapping.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    lang_mapping = joblib.load(lang_mapping_path)

    return model, scaler, lang_mapping


def assign_risk_level(probability: float) -> str:
    """
    Convert fake probability into a risk category.
    """
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"


def prepare_new_data_for_prediction(
    df_new: pd.DataFrame,
    scaler,
    lang_mapping: dict
):
    """
    Apply preprocessing and scaling to new input data.
    """
    df_processed = transform_preprocessor(df_new, lang_mapping)
    X_scaled = transform_scaler(df_processed, scaler)
    df_final = combine_scaled_features(df_processed, X_scaled)
    X = df_final[FEATURE_COLUMNS]

    return df_processed, df_final, X


def predict_new_data(
    df_new: pd.DataFrame,
    model,
    scaler,
    lang_mapping: dict
) -> pd.DataFrame:
    """
    Predict labels, probabilities, and risk levels for new data.
    """
    _, _, X = prepare_new_data_for_prediction(
        df_new=df_new,
        scaler=scaler,
        lang_mapping=lang_mapping
    )

    raw_predictions = model.predict(X)
    prediction_labels = ["Fake" if pred == 1 else "Real" for pred in raw_predictions]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if probabilities.shape[1] == 2:
            fake_probabilities = probabilities[:, 1]
        else:
            fake_probabilities = probabilities.max(axis=1)
    else:
        fake_probabilities = [None] * len(df_new)

    result_df = df_new.copy().reset_index(drop=True)
    result_df["prediction"] = prediction_labels
    result_df["fake_probability"] = pd.Series(fake_probabilities).round(4)
    result_df["risk_level"] = [
        assign_risk_level(prob) if prob is not None else "Unknown"
        for prob in fake_probabilities
    ]

    return result_df