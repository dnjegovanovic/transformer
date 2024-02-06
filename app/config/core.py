from pathlib import Path
from typing import Dict
from pydantic import BaseModel
from strictyaml import YAML, load
from yaml.loader import FullLoader
import yaml
import app

# Project Directories
PACKAGE_ROOT = Path(app.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    save_file: str


class Transformer(BaseModel):
    TransformerParam: Dict


class Config(BaseModel):
    """Master config object."""

    model_transformer: Transformer
    app_config: AppConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as stream:
        try:
            # Converts yaml document to python object
            parsed_config = yaml.load(stream, Loader=FullLoader)
            return parsed_config
        except yaml.YAMLError as e:
            print(e)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        for k, v in parsed_config.items():
            if k == "TransforemrParams":
                pass
            else:
                Exception("No configuration in config file.")

    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_UnetSP=Transformer(**parsed_config),
    )

    return _config


config = create_and_validate_config()
