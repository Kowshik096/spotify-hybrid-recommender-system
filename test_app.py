import time

import requests

app_url = "http://localhost:8501"
timeout_seconds = 120
poll_interval_seconds = 2


# get the status code
def get_app_status(url):
    try:
        response = requests.get(url, timeout=5)
        status_code = response.status_code
    except requests.RequestException:
        status_code = None
    return status_code


# test for the app home page loading
def test_app_loading():
    # poll the app until it responds or the timeout is reached
    deadline = time.time() + timeout_seconds
    status_code = None
    while time.time() < deadline:
        status_code = get_app_status(app_url)
        if status_code == 200:
            break
        time.sleep(poll_interval_seconds)
    assert status_code == 200, "Unable to load Streamlit App"
