from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db import Base, get_db
from backend.models import Run, RunStatus
from backend.routers import api
from automl_lite.wordpress_publisher import WordPressAuthError, WordPressConnectionError

POST_URL = "http://automl-lite.local/automl-lite-report/"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # In-memory database so tests never touch automl_lite.db
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    # Each run's report lives in tmp_path/<run_id>/report.html
    def fake_run_dir(run_id):
        run_dir = tmp_path / run_id
        run_dir.mkdir(exist_ok=True)
        return run_dir
    monkeypatch.setattr(api, "get_run_dir", fake_run_dir)

    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_db] = override_get_db

    test_client = TestClient(app)
    test_client.session = TestingSession
    test_client.run_dir = fake_run_dir
    return test_client


def add_run(client, run_id, status=RunStatus.COMPLETED.value, with_report=True):
    db = client.session()
    db.add(Run(id=run_id, filename="data.csv", target_column="target",
               task_type="classification", requested_task_type="auto", status=status))
    db.commit()
    db.close()
    if with_report:
        (client.run_dir(run_id) / "report.html").write_text("<html><body><h2>Report</h2></body></html>")


def stored_url(client, run_id):
    db = client.session()
    url = db.get(Run, run_id).wordpress_url
    db.close()
    return url


def test_publish_success_stores_url(client):
    add_run(client, "run-1")
    with patch.object(api, "publish_report", return_value=POST_URL) as mock_publish:
        res = client.post("/api/runs/run-1/publish")

    assert res.status_code == 200
    assert res.json() == {"run_id": "run-1", "wordpress_url": POST_URL}
    assert stored_url(client, "run-1") == POST_URL
    title, html = mock_publish.call_args[0]
    assert title == "AutoML Lite Report: target (classification)"
    assert "<h2>Report</h2>" in html


def test_publish_already_published_does_not_post_again(client):
    add_run(client, "run-2")
    with patch.object(api, "publish_report", return_value=POST_URL) as mock_publish:
        client.post("/api/runs/run-2/publish")
        res = client.post("/api/runs/run-2/publish")

    assert res.json()["wordpress_url"] == POST_URL
    assert mock_publish.call_count == 1


def test_publish_run_not_found(client):
    res = client.post("/api/runs/missing/publish")
    assert res.status_code == 404


def test_publish_run_not_finished(client):
    add_run(client, "run-3", status=RunStatus.RUNNING.value, with_report=False)
    with patch.object(api, "publish_report") as mock_publish:
        res = client.post("/api/runs/run-3/publish")

    assert res.status_code == 400
    assert "not finished" in res.json()["detail"]
    mock_publish.assert_not_called()


def test_publish_run_without_report(client):
    add_run(client, "run-4", with_report=False)
    res = client.post("/api/runs/run-4/publish")

    assert res.status_code == 404
    assert "no HTML report" in res.json()["detail"]


def test_publish_wrong_credentials(client):
    add_run(client, "run-5")
    with patch.object(api, "publish_report", side_effect=WordPressAuthError("WordPress rejected the credentials")):
        res = client.post("/api/runs/run-5/publish")

    assert res.status_code == 502
    assert "rejected the credentials" in res.json()["detail"]
    assert stored_url(client, "run-5") is None


def test_publish_site_unreachable(client):
    add_run(client, "run-6")
    with patch.object(api, "publish_report", side_effect=WordPressConnectionError("Could not reach WordPress")):
        res = client.post("/api/runs/run-6/publish")

    assert res.status_code == 503
