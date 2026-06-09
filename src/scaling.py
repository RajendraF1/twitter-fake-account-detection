from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from src.features import NEW_FEATURE_COLUMNS


FEATURE_COLUMNS = [
    "followers_count",
    "friends_count",
    "post_count",
    "location_available",
    "lang_encode",
    "desc_len",
    "account_age_days",
] + NEW_FEATURE_COLUMNS


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan: {missing_cols}")


def fit_scaler(df_train: pd.DataFrame):
    """
    Fit MinMaxScaler menggunakan data train.
    """
    validate_feature_columns(df_train)

    scaler = MinMaxScaler()
    X_train = df_train[FEATURE_COLUMNS]
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=FEATURE_COLUMNS,
        index=df_train.index
    )

    return scaler, X_train_scaled_df


def transform_scaler(df: pd.DataFrame, scaler: MinMaxScaler):
    """
    Transform data menggunakan scaler yang sudah di-fit dari train.
    """
    validate_feature_columns(df)

    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS,
        index=df.index
    )

    return X_scaled_df


def combine_scaled_features(
    original_df: pd.DataFrame,
    scaled_features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Menggabungkan dataframe asli dengan fitur yang sudah di-scale.
    Kolom fitur lama diganti dengan versi hasil scaling.
    """
    df_result = original_df.copy()
    df_result = df_result.drop(columns=FEATURE_COLUMNS)
    df_result = pd.concat([df_result, scaled_features_df], axis=1)
    return df_result


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """
    Menyimpan scaler ke file .pkl
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)