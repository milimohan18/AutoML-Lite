import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db import Base, get_db
from backend.models import Run, RunStatus
from backend.routers import web

POST_URL = "http://automl-lite.local/automl-lite-report/"


@pytest.fixture
def client():
    # In-memory database so tests never touch automl_lite.db
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    db = TestingSession()
    db.add(Run(id="run-1", filename="data.csv", target_column="target",
               task_type="classification", requested_task_type="auto",
               status=RunStatus.COMPLETED.value, wordpress_url=POST_URL))
    db.commit()
    db.close()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(web.router)
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/analytics", "/runs/run-1"])
def test_page_loads(client, path):
    res = client.get(path)
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_run_page_shows_wordpress_link(client):
    res = client.get("/runs/run-1")
    assert f'href="{POST_URL}"' in res.text
    assert "View on WordPress" in res.text


def test_unknown_run_page_returns_404(client):
    assert client.get("/runs/missing").status_code == 404
