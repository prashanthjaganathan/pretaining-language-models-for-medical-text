from enum import Enum
from pathlib import Path

class ProjectPaths(Enum):
    PROJECT_DIR = Path(__file__).resolve().parent
    DATASET_DIR = PROJECT_DIR / 'dataset'
    CONFIG_DIR = PROJECT_DIR / 'config'