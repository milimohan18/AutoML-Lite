from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, JSON
from datetime import datetime
import enum
from backend.db import Base

class RunStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"

class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    target_column = Column(String)
    task_type = Column(String)  # The resolved task type
    requested_task_type = Column(String)  # What user asked for
    time_budget = Column(Integer, nullable=True) # Seconds
    
    status = Column(String, default=RunStatus.QUEUED.value)
    error_message = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    dataset_summary = Column(JSON, nullable=True)
    best_metric_score = Column(Float, nullable=True)
    best_metric_name = Column(String, nullable=True)
    metrics_json = Column(JSON, nullable=True)
