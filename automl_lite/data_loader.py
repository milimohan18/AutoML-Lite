import pandas as pd
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV with error handling for common encodings."""
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="ISO-8859-1")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    # Strip stray whitespace from headers so "mean area " matches "mean area"
    df.columns = df.columns.str.strip()
    return df

def infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    """Infer whether task is classification or regression based on target_column."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    
    target_series = df[target_column]
    
    # If the target is string/object -> classification
    if target_series.dtype == "object" or target_series.dtype.name == "category":
        return "classification"
    
    # If target is numeric but has few unique values -> classification
    n_unique = target_series.nunique()
    if n_unique <= 15:
        return "classification"
    
    return "regression"

def preprocess_data_splits(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target from features."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
