import traceback
from datetime import datetime
from sqlalchemy.orm import Session

from backend.db import SessionLocal
from backend.models import Run, RunStatus
from automl_lite.config import get_run_dir
from automl_lite.data_loader import load_data, infer_task_type, preprocess_data_splits
from automl_lite.eda import generate_eda
from automl_lite.trainer import train_and_evaluate
from automl_lite.reporting import generate_html_report, save_metrics_json
from automl_lite.serialization import save_model

def execute_run(run_id: str):
    """Background task to execute the ML pipeline for a given run."""
    db: Session = SessionLocal()
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        db.close()
        return

    try:
        run.status = RunStatus.RUNNING.value
        run.started_at = datetime.utcnow()
        db.commit()

        run_dir = get_run_dir(run_id)
        
        # 1. Load Data
        df = load_data(run.filename)
        
        # 2. Infer Task Type
        if run.requested_task_type == "auto":
            task_type = infer_task_type(df, run.target_column)
        else:
            task_type = run.requested_task_type
        
        run.task_type = task_type
        db.commit()

        # 3. EDA
        eda_summary = generate_eda(df, run.target_column, task_type, run_dir)
        run.dataset_summary = eda_summary
        db.commit()

        # 4. Preprocessing splits
        X, y = preprocess_data_splits(df, run.target_column)
        
        # Simple train/test split (80/20) for final evaluation
        from sklearn.model_selection import train_test_split
        stratify = y if task_type == 'classification' else None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
        except ValueError:
            # Fallback if stratify fails due to single instance classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # 5. Training
        best_pipeline, leaderboard, best_model_info = train_and_evaluate(
            X_train, y_train, X_test, y_test, task_type
        )

        # 6. Serialization
        save_model(best_pipeline, run_dir)

        # 7. Reporting
        run_data_dict = {
            "task_type": task_type,
            "filename": run.filename,
            "target_column": run.target_column,
            "status": "COMPLETED"
        }
        generate_html_report(run_data_dict, leaderboard, best_model_info, run_dir)
        save_metrics_json(leaderboard, best_model_info, run_dir)

        # 8. Update DB
        run.status = RunStatus.COMPLETED.value
        run.completed_at = datetime.utcnow()
        run.best_metric_score = best_model_info["cv_score"]
        run.best_metric_name = best_model_info["primary_metric"]
        run.metrics_json = {"leaderboard": leaderboard, "best_model": best_model_info}
        
        db.commit()

    except Exception as e:
        run.status = RunStatus.FAILED.value
        run.error_message = str(e) + "\n" + traceback.format_exc()
        run.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()
