import pytest
import pandas as pd
from src.prediction import assign_risk_level, load_artifacts, predict_new_data


class TestAssignRiskLevel:
    def test_high_risk(self):
        assert assign_risk_level(0.70) == "High"
        assert assign_risk_level(0.85) == "High"
        assert assign_risk_level(1.00) == "High"

    def test_medium_risk(self):
        assert assign_risk_level(0.40) == "Medium"
        assert assign_risk_level(0.55) == "Medium"
        assert assign_risk_level(0.69) == "Medium"

    def test_low_risk(self):
        assert assign_risk_level(0.00) == "Low"
        assert assign_risk_level(0.20) == "Low"
        assert assign_risk_level(0.39) == "Low"

    def test_boundary_values(self, lang_mapping):
        assert assign_risk_level(0.39) == "Low"
        assert assign_risk_level(0.40) == "Medium"
        assert assign_risk_level(0.69) == "Medium"
        assert assign_risk_level(0.70) == "High"


class TestLoadArtifacts:
    def test_load_all_artifacts(self, temp_artifacts):
        model, scaler, mapping = load_artifacts(
            model_path=str(temp_artifacts / "slp_model.pkl"),
            scaler_path=str(temp_artifacts / "scaler.pkl"),
            lang_mapping_path=str(temp_artifacts / "lang_mapping.pkl"),
        )
        assert hasattr(model, "predict")
        assert hasattr(scaler, "transform")
        assert isinstance(mapping, dict)

    def test_load_missing_model(self):
        with pytest.raises(FileNotFoundError):
            load_artifacts(
                model_path="nonexistent.pkl",
                scaler_path="nonexistent.pkl",
                lang_mapping_path="nonexistent.pkl",
            )


class TestPredictNewData:
    def test_prediction_columns_added(self, sample_df, temp_artifacts):
        model, scaler, mapping = load_artifacts(
            model_path=str(temp_artifacts / "slp_model.pkl"),
            scaler_path=str(temp_artifacts / "scaler.pkl"),
            lang_mapping_path=str(temp_artifacts / "lang_mapping.pkl"),
        )
        result = predict_new_data(sample_df, model, scaler, mapping)
        assert "prediction" in result.columns
        assert "fake_probability" in result.columns
        assert "risk_level" in result.columns

    def test_prediction_length_match(self, sample_df, temp_artifacts):
        model, scaler, mapping = load_artifacts(
            model_path=str(temp_artifacts / "slp_model.pkl"),
            scaler_path=str(temp_artifacts / "scaler.pkl"),
            lang_mapping_path=str(temp_artifacts / "lang_mapping.pkl"),
        )
        result = predict_new_data(sample_df, model, scaler, mapping)
        assert len(result) == len(sample_df)

    def test_prediction_labels_valid(self, sample_df, temp_artifacts):
        model, scaler, mapping = load_artifacts(
            model_path=str(temp_artifacts / "slp_model.pkl"),
            scaler_path=str(temp_artifacts / "scaler.pkl"),
            lang_mapping_path=str(temp_artifacts / "lang_mapping.pkl"),
        )
        result = predict_new_data(sample_df, model, scaler, mapping)
        assert result["prediction"].isin(["Fake", "Real"]).all()

    def test_risk_levels_valid(self, sample_df, temp_artifacts):
        model, scaler, mapping = load_artifacts(
            model_path=str(temp_artifacts / "slp_model.pkl"),
            scaler_path=str(temp_artifacts / "scaler.pkl"),
            lang_mapping_path=str(temp_artifacts / "lang_mapping.pkl"),
        )
        result = predict_new_data(sample_df, model, scaler, mapping)
        assert result["risk_level"].isin(["High", "Medium", "Low"]).all()
