# config.py
import json
from pathlib import Path

_config_data = None  # cached config

def get_config(path: Path = None):
    """Return global config, loading it from JSON if not already loaded."""
    global _config_data
    if _config_data is None:
        if path is None:
            path = Path(__file__).parent / "configs" / "default_config.json"
        with open(path) as f:
            _config_data = json.load(f)
    return _config_data
