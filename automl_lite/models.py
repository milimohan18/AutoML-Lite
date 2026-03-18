from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

def get_models(task_type: str) -> dict:
    """Returns a dictionary of uninitialized models for the specified task type."""
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42)
        }
    elif task_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def get_primary_metric(task_type: str) -> str:
    """Returns the scikit-learn scoring string for the primary metric."""
    if task_type == "classification":
        return "accuracy" # easier default for multiclass compatibility
    elif task_type == "regression":
        return "r2"
    return "accuracy"
