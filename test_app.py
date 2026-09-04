import time

import pytest
import requests

app_url = "http://localhost:8501"
timeout_seconds = 120
poll_interval_seconds = 2
max_backoff_seconds = 10


# get the status code
def get_app_status(url: str) -> int | None:
    try:
        response = requests.get(url, timeout=5)
        status_code = response.status_code
    except requests.RequestException:
        status_code = None
    return status_code


@pytest.mark.integration
def test_app_loading() -> None:
    """Smoke test for the running Streamlit app.

    Requires a live server on :8501 (e.g. `streamlit run app.py` or the
    docker-compose `app` service). Skipped by default in unit-test runs so
    the CI unit-test job does not depend on a cold-starting server; the
    dedicated streamlit-smoke / docker-build jobs cover this.
    """
    # poll the app until it responds or the timeout is reached
    # uses exponential backoff to avoid hammering a cold-starting server
    deadline = time.time() + timeout_seconds
    status_code = None
    delay = poll_interval_seconds
    while time.time() < deadline:
        status_code = get_app_status(app_url)
        if status_code == 200:
            break
        time.sleep(delay)
        delay = min(delay * 2, max_backoff_seconds)
    assert status_code == 200, "Unable to load Streamlit App"
