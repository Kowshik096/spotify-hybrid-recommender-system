"""Centralised configuration for the Hybrid Recommender System.

All settings can be overridden via environment variables. Defaults point
at the project's data directory and conventional ports so the app works
out-of-the-box, while allowing operators to redirect paths/ports/URIs
without editing source files.

Environment variables:
    DATA_DIR        — directory containing the pre-computed artifacts
                      (default: <project_root>/data)
    METRICS_PORT    — Prometheus metrics HTTP port (default: 9090)
    STREAMLIT_PORT  — Streamlit server port (default: 8501)
    MLFLOW_TRACKING_URI — MLflow tracking server URI
                      (default: file://<project_root>/mlruns)
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data")))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9090"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlruns'}")

# Artifact file names (paths are resolved lazily from DATA_DIR so that
# overriding DATA_DIR at runtime — e.g. in tests — takes effect)
_ARTIFACT_FILES = {
    "cleaned_data": "cleaned_data.csv",
    "transformed_data": "transformed_data.npz",
    "track_ids": "track_ids.npy",
    "collab_filtered_data": "collab_filtered_data.csv",
    "interaction_matrix": "interaction_matrix.npz",
    "transformed_hybrid_data": "transformed_hybrid_data.npz",
}


def artifact_path(name: str) -> str:
    """Return the absolute path for a named artifact, raising if unknown.

    Paths are derived from DATA_DIR at call time, so the data directory
    can be redirected via the DATA_DIR environment variable or by
    monkeypatching config.DATA_DIR in tests.
    """
    if name not in _ARTIFACT_FILES:
        raise KeyError(f"Unknown artifact '{name}'. Known: {list(_ARTIFACT_FILES)}")
    return str(DATA_DIR / _ARTIFACT_FILES[name])
