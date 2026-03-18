import joblib
from pathlib import Path
from typing import Any

def save_model(model: Any, run_dir: Path):
    """Saves the scikit-learn pipeline object."""
    model_path = run_dir / 'best_model.joblib'
    joblib.dump(model, model_path)

def load_model(run_dir: Path) -> Any:
    """Loads a previously saved model."""
    model_path = run_dir / 'best_model.joblib'
    return joblib.load(model_path)
