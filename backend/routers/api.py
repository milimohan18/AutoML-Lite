import os
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime

from backend.db import get_db
from backend.models import Run, RunStatus, TaskType
from backend.schemas import RunResponse
from backend.background import execute_run
from automl_lite.config import get_run_dir, BASE_DIR

router = APIRouter(prefix="/api/runs", tags=["runs"])

UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=RunResponse)
async def create_run(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: TaskType = Form(TaskType.AUTO),
    time_budget: int = Form(None),
    db: Session = Depends(get_db)
):
    run_id = str(uuid.uuid4())
    
    file_extension = Path(file.filename).suffix
    local_filename = f"{run_id}{file_extension}"
    file_path = UPLOAD_DIR / local_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    new_run = Run(
        id=run_id,
        filename=str(file_path),
        target_column=target_column,
        requested_task_type=task_type.value,
        task_type="pending...",
        time_budget=time_budget
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    
    background_tasks.add_task(execute_run, run_id)
    
    return new_run

@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

@router.get("/{run_id}/download/model")
def download_model(run_id: str):
    run_dir = get_run_dir(run_id)
    model_path = run_dir / "best_model.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=f"model_{run_id}.joblib")

@router.get("/{run_id}/download/report")
def download_report(run_id: str):
    run_dir = get_run_dir(run_id)
    report_path = run_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, filename=f"report_{run_id}.html")

@router.get("/{run_id}/download/metrics")
def download_metrics(run_id: str):
    run_dir = get_run_dir(run_id)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    return FileResponse(metrics_path, filename=f"metrics_{run_id}.json")
