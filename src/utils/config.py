from pathlib import Path
from typing import Any
import yaml
import os
from dotenv import load_dotenv


load_dotenv()  # so a CONFIG_PATH set in .env is visible before the config is read

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_path = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)

with open(_path) as f:
	config = yaml.safe_load(f) # Only read once at first import
del f, _path