"""
Define working paths, including:
    - pRT input data path
    - Project root
        - input data directory
        - output data directory
"""
import os
import getpass
from pathlib import Path

# pRT input data path
def setup_prt_path():
    if getpass.getuser() == "chenyangji":
        os.environ.setdefault(
            "pRT_input_data_path",
            "/shared/petitRADTRANS/"
        )

# Project root
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "input"
print(f"Data directory set to: {DATA_DIR}")

# Output directory, create one if not exist
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# config directory
CONFIG_DIR = PROJECT_ROOT / "configs"