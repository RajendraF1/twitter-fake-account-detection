import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.model import (
    prepare_train_data,
    prepare_test_data,
    build_slp_model,
    train_model,
    save_model,
)
from src.pipeline import build_model


class TestPrepareData:
    def test_prepare_train_data_shapes(self, sample_df, lang_mapping):
        from src.preprocessing import preprocess_data
        from src.features import add_all_features
        from src.scaling import fit_scaler, combine_scaled_features

        processed = preprocess_data(sample_df, lang_mapping)
        processed = add_all_features(processed)
        scaler, X_scaled = fit_scaler(processed)
        df_final = combine_scaled_features(processed, X_scaled)

        X, y = prepare_train_data(df_final)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_prepare_test_data_with_target(self, sample_df, lang_mapping):
        from src.preprocessing import preprocess_data
        from src.features import add_all_features
        from src.scaling import fit_scaler, combine_scaled_features

        processed = preprocess_data(sample_df, lang_mapping)
        processed = add_all_features(processed)
        scaler, X_scaled = fit_scaler(processed)
        df_final = combine_scaled_features(processed, X_scaled)

        X, y = prepare_test_data(df_final)
        assert X is not None
        assert y is not None

    def test_prepare_test_data_without_target(self, sample_df_no_target, lang_mapping):
        from src.preprocessing import preprocess_data
        from src.features import add_all_features
        from src.scaling import fit_scaler, combine_scaled_features

        processed = preprocess_data(sample_df_no_target, lang_mapping)
        processed = add_all_features(processed)
        scaler, X_scaled = fit_scaler(processed)
        df_final = combine_scaled_features(processed, X_scaled)

        X, y = prepare_test_data(df_final)
        assert X is not None
        assert y is None


class TestBuildSLP:
    def test_build_slp_model(self):
        model = build_slp_model()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_model_from_registry(self):
        model = build_model("slp", {"slp": {"hidden_layer_sizes": [], "activation": "logistic", "max_iter": 100, "random_state": 42}})
        assert hasattr(model, "fit")

    def test_build_model_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("invalid_model", {})


class TestTrainModel:
    def test_train_model_returns_model(self, sample_df, lang_mapping):
        from src.preprocessing import preprocess_data
        from src.features import add_all_features
        from src.scaling import fit_scaler, combine_scaled_features

        processed = preprocess_data(sample_df, lang_mapping)
        processed = add_all_features(processed)
        scaler, X_scaled = fit_scaler(processed)
        df_final = combine_scaled_features(processed, X_scaled)
        X, y = prepare_train_data(df_final)

        model = train_model(X, y)
        assert hasattr(model, "coefs_")
        assert model.n_features_in_ == X.shape[1]

    def test_train_model_prediction_shape(self, sample_df, lang_mapping):
        from src.preprocessing import preprocess_data
        from src.features import add_all_features
        from src.scaling import fit_scaler, combine_scaled_features

        processed = preprocess_data(sample_df, lang_mapping)
        processed = add_all_features(processed)
        scaler, X_scaled = fit_scaler(processed)
        df_final = combine_scaled_features(processed, X_scaled)
        X, y = prepare_train_data(df_final)

        model = train_model(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})


class TestSaveLoad:
    def test_save_model_creates_file(self, trained_model, temp_model_dir):
        path = temp_model_dir / "test_model.pkl"
        save_model(trained_model, str(path))
        assert path.exists()

    def test_save_model_loadable(self, trained_model, temp_model_dir):
        import joblib
        path = temp_model_dir / "test_model.pkl"
        save_model(trained_model, str(path))
        loaded = joblib.load(path)
        assert hasattr(loaded, "predict")
