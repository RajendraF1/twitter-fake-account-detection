from pathlib import Path
from typing import Any, Dict
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_config() -> Dict[str, Any]:
    return load_config()["data"]


def get_feature_config() -> Dict[str, Any]:
    return load_config()["features"]


def get_model_config() -> Dict[str, Any]:
    return load_config()["model"]


def get_training_config() -> Dict[str, Any]:
    return load_config()["training"]


def get_paths_config() -> Dict[str, Any]:
    return load_config()["paths"]


FEATURE_COLUMNS = get_feature_config()["columns"]
TARGET_COLUMN = get_feature_config()["target"]
