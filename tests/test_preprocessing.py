import pytest
import pandas as pd
from src.preprocessing import (
    validate_columns,
    build_language_mapping,
    preprocess_data,
    fit_preprocessor,
    transform_preprocessor,
)
from src.features import add_all_features, NEW_FEATURE_COLUMNS


class TestValidateColumns:
    def test_valid_columns(self, sample_df):
        validate_columns(sample_df)

    def test_missing_column(self, sample_df):
        df_bad = sample_df.drop(columns=["location"])
        with pytest.raises(ValueError, match="location"):
            validate_columns(df_bad)


class TestBuildLanguageMapping:
    def test_mapping_keys(self, sample_df, lang_mapping):
        mapping = build_language_mapping(sample_df)
        assert isinstance(mapping, dict)
        assert "en" in mapping
        assert "fr" in mapping

    def test_unknown_handling(self, sample_df):
        mapping = build_language_mapping(sample_df)
        assert "unknown" in mapping

    def test_empty_string_handling(self):
        df = pd.DataFrame({"lang": ["en", "", "fr", None]})
        mapping = build_language_mapping(df)
        assert "unknown" in mapping


class TestPreprocessData:
    def test_preprocess_columns_added(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        expected_cols = [
            "location_available", "lang_encode",
            "desc_len", "account_age_days"
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_location_available(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        assert list(result["location_available"]) == [1, 0, 1, 0, 1]

    def test_lang_encode_mapped(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        assert result.loc[0, "lang_encode"] == 0
        assert result.loc[1, "lang_encode"] == 1

    def test_unknown_lang_negative_one(self):
        df = pd.DataFrame({
            "location": [""], "lang": ["ja"],
            "description": [""], "created_at": pd.to_datetime(["2020-01-01"]),
            "followers_count": [0], "friends_count": [0], "post_count": [0],
        })
        lang_mapping = {"en": 0, "fr": 1, "unknown": 2}
        result = preprocess_data(df, lang_mapping)
        assert result.loc[0, "lang_encode"] == -1

    def test_desc_len_calculation(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        assert result.loc[0, "desc_len"] == len("Hello world")
        assert result.loc[2, "desc_len"] == 0

    def test_account_age_positive(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        assert (result["account_age_days"] >= 0).all()

    def test_features_added(self, sample_df, lang_mapping):
        result = preprocess_data(sample_df, lang_mapping)
        result = add_all_features(result)
        for col in NEW_FEATURE_COLUMNS:
            assert col in result.columns


class TestFitTransformPreprocessor:
    def test_fit_preprocessor_returns_mapping(self, sample_df):
        _, mapping = fit_preprocessor(sample_df)
        assert isinstance(mapping, dict)

    def test_transform_preprocessor(self, sample_df, lang_mapping):
        result = transform_preprocessor(sample_df, lang_mapping)
        assert "location_available" in result.columns

    def test_fit_transform_consistency(self, sample_df):
        processed, mapping = fit_preprocessor(sample_df)
        transformed = transform_preprocessor(sample_df, mapping)
        pd.testing.assert_frame_equal(processed, transformed)
