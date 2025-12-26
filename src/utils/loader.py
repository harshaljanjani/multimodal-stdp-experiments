import json
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data
