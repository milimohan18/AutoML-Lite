import base64
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from automl_lite.wordpress_publisher import (
    extract_report_body,
    publish_report,
    WordPressAuthError,
    WordPressConfigError,
    WordPressConnectionError,
    WordPressError,
)

REPORT_HTML = """<!DOCTYPE html>
<html>
<head><title>AutoML Lite Report</title><style>body { color: red; }</style></head>
<body>
    <h1>AutoML Lite Report</h1>
    <h2>Leaderboard</h2>
    <table>
        <tr><th>Model</th><th>CV Score</th></tr>
        <tr><td>RandomForest</td><td>0.9500</td></tr>
    </table>
    <img src="target_distribution.png" alt="Target Distribution">
    <script>alert('hi')</script>
</body>
</html>"""


@pytest.fixture
def wp_env(monkeypatch):
    monkeypatch.setenv("WP_SITE_URL", "http://automl-lite.local/")
    monkeypatch.setenv("WP_USERNAME", "admin")
    monkeypatch.setenv("WP_APP_PASSWORD", "abcd efgh ijkl mnop")


def fake_response(body: dict):
    """A stand-in for the object urllib.request.urlopen returns."""
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__.return_value = response
    return response


def http_error(code: int, body: dict):
    return urllib.error.HTTPError(
        "http://automl-lite.local/wp-json/wp/v2/posts", code, "Error", {},
        io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def test_extract_report_body_keeps_tables_and_headings():
    content = extract_report_body(REPORT_HTML)

    assert "<h2>Leaderboard</h2>" in content
    assert "<table style=" in content and "<td style=" in content
    assert "RandomForest" in content
    # Removed: page wrapper, <style>, <script>, duplicate <h1>, relative image
    for removed in ("<html", "<head", "<body", "<style", "<script", "<h1", "<img"):
        assert removed not in content


def test_publish_report_success(wp_env):
    with patch("urllib.request.urlopen", return_value=fake_response(
        {"id": 7, "link": "http://automl-lite.local/automl-lite-report/"}
    )) as mock_urlopen:
        url = publish_report("My Report", REPORT_HTML)

    assert url == "http://automl-lite.local/automl-lite-report/"

    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "http://automl-lite.local/wp-json/wp/v2/posts"
    assert request.get_method() == "POST"
    expected_token = base64.b64encode(b"admin:abcd efgh ijkl mnop").decode()
    assert request.get_header("Authorization") == f"Basic {expected_token}"

    payload = json.loads(request.data)
    assert payload["title"] == "My Report"
    assert payload["status"] == "publish"
    assert "<h2>Leaderboard</h2>" in payload["content"]


def test_publish_report_missing_config(monkeypatch):
    for key in ("WP_SITE_URL", "WP_USERNAME", "WP_APP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(WordPressConfigError, match="WP_SITE_URL"):
        publish_report("My Report", REPORT_HTML)


def test_publish_report_wrong_credentials(wp_env):
    error = http_error(401, {"code": "incorrect_password", "message": "The password is incorrect."})
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(WordPressAuthError, match="rejected the credentials"):
            publish_report("My Report", REPORT_HTML)


def test_publish_report_site_unreachable(wp_env):
    error = urllib.error.URLError("[Errno 11001] getaddrinfo failed")
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(WordPressConnectionError, match="Could not reach WordPress"):
            publish_report("My Report", REPORT_HTML)


def test_publish_report_failed_request(wp_env):
    error = http_error(500, {"code": "internal_error", "message": "Something broke."})
    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(WordPressError, match="HTTP 500"):
            publish_report("My Report", REPORT_HTML)
