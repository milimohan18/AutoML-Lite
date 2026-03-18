import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "static" / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

# Database URLs
DATABASE_URL = f"sqlite:///{BASE_DIR / 'automl_lite.db'}"

# Create required directories
os.makedirs(RUNS_DIR, exist_ok=True)

# Application Config
MAX_UPLOAD_SIZE_MB = 50
DEFAULT_TEST_SIZE = 0.2
RANDOM_SEED = 42

def get_run_dir(run_id: str) -> Path:
    """Get the directory path for a specific run and create it if not exists."""
    run_dir = RUNS_DIR / str(run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
