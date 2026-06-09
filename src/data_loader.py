from pathlib import Path
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Membaca file dataset dari format CSV atau Excel.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".xlsx":
        return pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Format file tidak didukung: {suffix}. Gunakan .csv atau .xlsx"
        )


def load_train_test_data(train_path: str, test_path: str):
    """
    Membaca data train dan test sekaligus.
    """
    df_train = load_data(train_path)
    df_test = load_data(test_path)
    return df_train, df_test