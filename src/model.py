from pathlib import Path
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.scaling import FEATURE_COLUMNS


TARGET_COLUMN = "fake"


def validate_target_column(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan di dataframe.")


def prepare_train_data(df_train: pd.DataFrame):
    """
    Memisahkan fitur dan target dari data train.
    """
    validate_target_column(df_train)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df_train.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan: {missing_features}")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]

    return X_train, y_train


def prepare_test_data(df_test: pd.DataFrame):
    """
    Mengambil fitur dari data test.
    Jika test memiliki label, label juga akan diambil.
    """
    missing_features = [col for col in FEATURE_COLUMNS if col not in df_test.columns]
    if missing_features:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan: {missing_features}")

    X_test = df_test[FEATURE_COLUMNS]

    y_test = None
    if TARGET_COLUMN in df_test.columns:
        y_test = df_test[TARGET_COLUMN]

    return X_test, y_test


def build_slp_model():
    """
    Membuat model Single Layer Perceptron menggunakan MLPClassifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(),
        activation="logistic",
        solver="adam",
        max_iter=1000,
        random_state=42
    )
    return model


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Melatih model SLP.
    """
    model = build_slp_model()
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str) -> None:
    """
    Menyimpan model ke file .pkl
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)