"""
Publish AutoML Lite reports to a WordPress site through the WordPress REST API.

Connection settings come from the .env file in the project root (see .env.example):
  WP_SITE_URL      e.g. http://automl-lite.local  (your LocalWP site)
  WP_USERNAME      the WordPress user the posts are created as
  WP_APP_PASSWORD  an Application Password (Users > Profile > Application Passwords)
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request

from dotenv import load_dotenv

from automl_lite.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")

REQUEST_TIMEOUT_SECONDS = 15

# The report's <style> block is dropped by WordPress, so tables get inline styles instead
TABLE_STYLE = "border-collapse: collapse; width: 100%; margin-bottom: 20px;"
CELL_STYLE = "border: 1px solid #ddd; padding: 8px; text-align: left;"


class WordPressError(Exception):
    """Publishing failed (base class for all WordPress errors)."""


class WordPressConfigError(WordPressError):
    """WP_SITE_URL, WP_USERNAME or WP_APP_PASSWORD is missing."""


class WordPressAuthError(WordPressError):
    """WordPress rejected the username or application password."""


class WordPressConnectionError(WordPressError):
    """The WordPress site could not be reached."""


def get_wordpress_config() -> tuple:
    """Read the site URL, username and application password from the environment."""
    site_url = os.getenv("WP_SITE_URL", "").strip().rstrip("/")
    username = os.getenv("WP_USERNAME", "").strip()
    app_password = os.getenv("WP_APP_PASSWORD", "").strip()

    missing = [name for name, value in [
        ("WP_SITE_URL", site_url),
        ("WP_USERNAME", username),
        ("WP_APP_PASSWORD", app_password),
    ] if not value]
    if missing:
        raise WordPressConfigError(
            f"WordPress is not configured. Set {', '.join(missing)} in the .env file."
        )
    return site_url, username, app_password


def extract_report_body(report_html: str) -> str:
    """Turn a full HTML report into content that WordPress can show as a post body."""
    # Keep only what is inside <body>...</body>
    match = re.search(r"<body[^>]*>(.*)</body>", report_html, re.S | re.I)
    content = match.group(1) if match else report_html

    # WordPress strips <script> and most <style> tags, so remove them up front
    content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", content, flags=re.S | re.I)

    # The post title already names the report, so drop the duplicate top heading
    content = re.sub(r"<h1[^>]*>.*?</h1>", "", content, count=1, flags=re.S | re.I)

    # Chart images use relative paths (e.g. "target_distribution.png") that don't exist on
    # the WordPress site, so remove them rather than publish broken images
    content = re.sub(r'<img[^>]*src="(?!https?://)[^"]*"[^>]*>', "", content, flags=re.I)

    # Keep tables readable without the report's stylesheet
    content = content.replace("<table>", f'<table style="{TABLE_STYLE}">')
    content = content.replace("<th>", f'<th style="{CELL_STYLE} background-color: #f2f2f2;">')
    content = content.replace("<td>", f'<td style="{CELL_STYLE}">')

    return content.strip()


def publish_report(title: str, report_html: str) -> str:
    """Create a published WordPress post from a report and return the post's URL."""
    site_url, username, app_password = get_wordpress_config()
    endpoint = f"{site_url}/wp-json/wp/v2/posts"

    payload = json.dumps({
        "title": title,
        "content": extract_report_body(report_html),
        "status": "publish",
    }).encode("utf-8")

    # HTTP Basic Auth: "username:application password", base64-encoded
    token = base64.b64encode(f"{username}:{app_password}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(endpoint, data=payload, method="POST")
    request.add_header("Authorization", f"Basic {token}")
    request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            post = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # WordPress explains errors in a JSON body like {"code": ..., "message": ...}
        try:
            message = json.loads(e.read().decode("utf-8")).get("message", e.reason)
        except ValueError:
            message = e.reason
        if e.code in (401, 403):
            raise WordPressAuthError(
                f"WordPress rejected the credentials (HTTP {e.code}): {message}. "
                "Check WP_USERNAME and WP_APP_PASSWORD."
            ) from e
        raise WordPressError(f"WordPress request failed (HTTP {e.code}): {message}") from e
    except OSError as e:
        # URLError, timeouts and refused connections are all OSErrors
        reason = getattr(e, "reason", e)
        raise WordPressConnectionError(
            f"Could not reach WordPress at {site_url}: {reason}. Is the LocalWP site running?"
        ) from e
    except ValueError as e:
        raise WordPressError("WordPress returned a response that is not valid JSON.") from e

    if "link" not in post:
        raise WordPressError("WordPress did not return a link for the new post.")
    return post["link"]
