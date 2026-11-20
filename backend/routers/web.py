import os
import pandas as pd
from fastapi import APIRouter, Request, Depends, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from backend.db import get_db
from backend.models import Run

router = APIRouter(tags=["web"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the homepage with the upload form."""
    return templates.TemplateResponse(request, "index.html", {})

@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_status(request: Request, run_id: str, db: Session = Depends(get_db)):
    """Render the status page for a specific run."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        return HTMLResponse("Run not found", status_code=404)
        
    return templates.TemplateResponse(request, "run_status.html", {"run": run})

@router.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Render the analytics BI dashboard page."""
    return templates.TemplateResponse(request, "analytics.html", {})
