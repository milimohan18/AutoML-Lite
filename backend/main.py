from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from backend.db import engine, Base
from backend.routers import api, web
from automl_lite.config import ARTIFACTS_DIR
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AutoML Lite API", version="0.1.0")

# Mount static directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Create static directories if they don't exist
static_dir = BASE_DIR / "static"
os.makedirs(static_dir, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# User-uploaded generated artifacts
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
# App static assets
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(web.router)
app.include_router(api.router)
