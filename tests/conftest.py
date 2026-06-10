import pytest
import pandas as pd
import numpy as np
import tempfile
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "location": ["New York", None, "London", None, "Paris"],
        "lang": ["en", "fr", None, "en", ""],
        "description": ["Hello world", "", None, "Check this out https://example.com", "Short"],
        "created_at": pd.to_datetime(["2020-01-01", "2023-06-15", "2018-03-10", "2025-10-01", "2022-12-25"]),
        "followers_count": [100, 500, 50, 1000, 200],
        "friends_count": [50, 200, 10, 500, 100],
        "post_count": [10, 50, 5, 200, 30],
        "fake": [0, 1, 1, 0, 1],
        "screen_name": ["john_doe", "user12345", "bot_999", "realperson", "test42abc"],
    })


@pytest.fixture
def sample_df_no_target():
    return pd.DataFrame({
        "location": ["New York", None, "London"],
        "lang": ["en", "fr", None],
        "description": ["Hello", "", None],
        "created_at": pd.to_datetime(["2020-01-01", "2023-06-15", "2018-03-10"]),
        "followers_count": [100, 500, 50],
        "friends_count": [50, 200, 10],
        "post_count": [10, 50, 5],
        "screen_name": ["user1", "user2", "user3"],
    })


@pytest.fixture
def lang_mapping():
    return {"en": 0, "fr": 1, "de": 2, "unknown": 3}


@pytest.fixture
def temp_model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trained_model():
    model = MLPClassifier(
        hidden_layer_sizes=(),
        activation="logistic",
        solver="adam",
        max_iter=100,
        random_state=42,
    )
    X = np.random.rand(50, 11)
    y = np.random.randint(0, 2, 50)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_scaler():
    scaler = MinMaxScaler()
    X = np.random.rand(50, 11)
    scaler.fit(X)
    return scaler


@pytest.fixture
def temp_artifacts(temp_model_dir, trained_model, fitted_scaler, lang_mapping):
    joblib.dump(trained_model, temp_model_dir / "slp_model.pkl")
    joblib.dump(fitted_scaler, temp_model_dir / "scaler.pkl")
    joblib.dump(lang_mapping, temp_model_dir / "lang_mapping.pkl")
    return temp_model_dir
