from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from datetime import datetime
from backend.models import RunStatus, TaskType

class RunCreate(BaseModel):
    filename: str
    target_column: str
    task_type: TaskType = TaskType.AUTO
    time_budget: Optional[int] = None

class RunResponse(BaseModel):
    id: str
    filename: str
    target_column: str
    task_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    dataset_summary: Optional[Dict[str, Any]] = None
    best_metric_score: Optional[float] = None
    best_metric_name: Optional[str] = None
    metrics_json: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
