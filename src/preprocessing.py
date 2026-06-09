import pandas as pd


REQUIRED_COLUMNS = [
    "location",
    "lang",
    "description",
    "created_at",
    "followers_count",
    "friends_count",
    "post_count"
]


def validate_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan: {missing_cols}")


def build_language_mapping(df_train: pd.DataFrame) -> dict:
    """
    Membuat mapping bahasa dari data train agar encoding konsisten.
    Contoh:
    {'en': 0, 'id': 1, 'ja': 2}
    """
    unique_langs = (
        df_train["lang"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
        .unique()
    )

    unique_langs = sorted(unique_langs)
    return {lang: idx for idx, lang in enumerate(unique_langs)}


def preprocess_data(
    df: pd.DataFrame,
    lang_mapping: dict,
    reference_date: str = "2025-12-02"
) -> pd.DataFrame:
    """
    Melakukan preprocessing pada dataframe:
    - location -> location_available
    - lang -> lang_encode
    - description -> desc_len
    - created_at -> account_age_days
    """
    df = df.copy()
    validate_columns(df)

    # 1. Location availability
    df["location_available"] = df["location"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
    )

    # 2. Language normalization
    df["lang"] = (
        df["lang"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )

    # Jika ada bahasa di test yang tidak ada di train, beri kode -1
    df["lang_encode"] = df["lang"].map(lang_mapping).fillna(-1).astype(int)

    # 3. Description length
    df["description"] = df["description"].fillna("").astype(str)
    df["desc_len"] = df["description"].apply(len)

    # 4. Created at -> account age in days
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    ref_date = pd.Timestamp(reference_date)
    df["account_age_days"] = (ref_date - df["created_at"]).dt.days

    # Jika created_at gagal dibaca, hasilnya NaN -> isi 0
    df["account_age_days"] = df["account_age_days"].fillna(0).astype(int)

    return df


def fit_preprocessor(
    df_train: pd.DataFrame,
    reference_date: str = "2025-12-02"
):
    """
    Fit aturan preprocessing dari data train,
    lalu kembalikan:
    - dataframe train yang sudah diproses
    - language mapping
    """
    validate_columns(df_train)
    lang_mapping = build_language_mapping(df_train)
    df_train_processed = preprocess_data(df_train, lang_mapping, reference_date)

    return df_train_processed, lang_mapping


def transform_preprocessor(
    df: pd.DataFrame,
    lang_mapping: dict,
    reference_date: str = "2025-12-02"
):
    """
    Terapkan preprocessing ke data lain (misalnya test)
    menggunakan aturan dari train.
    """
    return preprocess_data(df, lang_mapping, reference_date)