import pandas as pd
import time
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, r2_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, root_mean_squared_error
)

from automl_lite.preprocessing import build_preprocessor
from automl_lite.models import get_models, get_primary_metric

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Trains multiple models, evaluates via CV, selects back, and evaluates on test set."""
    
    preprocessor = build_preprocessor(X_train)
    models = get_models(task_type)
    primary_metric = get_primary_metric(task_type)
    
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'accuracy': 'accuracy'}
        primary_score_key = 'test_accuracy'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'r2': 'r2'}
        primary_score_key = 'test_r2'

    leaderboard = {}
    best_score = -float('inf')
    best_model_name = None
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        start_time = time.time()
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        train_time = time.time() - start_time
        
        mean_score = cv_results[primary_score_key].mean()
        leaderboard[name] = {
            "cv_score": float(mean_score),
            "train_time_seconds": float(train_time)
        }
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_pipeline = pipeline

    # Final fit on full training data using the best model
    best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    test_metrics = {}
    if task_type == 'classification':
        test_metrics['accuracy']  = float(accuracy_score(y_test, y_pred))
        # 'weighted' works for any labels (e.g. "benign"/"malignant"); 'binary' assumes pos_label=1
        test_metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        test_metrics['recall']    = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        test_metrics['f1_score']  = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    else:
        test_metrics['r2']   = float(r2_score(y_test, y_pred))
        test_metrics['mae']  = float(mean_absolute_error(y_test, y_pred))
        test_metrics['rmse'] = float(root_mean_squared_error(y_test, y_pred))
        
    best_model_info = {
        "name": best_model_name,
        "cv_score": float(best_score),
        "test_metrics": test_metrics,
        "primary_metric": primary_metric
    }

    return best_pipeline, leaderboard, best_model_info
