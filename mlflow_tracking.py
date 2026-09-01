"""
MLflow Experiment Tracking for Hybrid Recommender System.

Provides utilities to log parameters, metrics, and artifacts during training and evaluation.
"""

import json
import os
from typing import Any

import mlflow
import mlflow.sklearn
from scipy.sparse import csr_matrix


class MLflowTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "spotify-hybrid-recsys",
        tracking_uri: str | None = None,
        run_name: str | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (e.g., "http://localhost:5000")
            run_name: Optional name for the run
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file store
            mlflow.set_tracking_uri("file:./mlruns")

        self.experiment_name = experiment_name
        self.run_name = run_name

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        self.run = None

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        """Start a new MLflow run."""
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name or self.run_name, tags=tags or {}
        )
        return self.run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        self.run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact (file or directory)."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Log all artifacts in a directory."""
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Any = None,
        input_example: Any = None,
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as JSON artifact."""
        with open(artifact_file, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(artifact_file)
        os.remove(artifact_file)


# Convenience functions for pipeline stages


def log_data_cleaning_stage(
    tracker: MLflowTracker,
    raw_path: str,
    cleaned_path: str,
    raw_rows: int,
    cleaned_rows: int,
    dropped_columns: list,
):
    """Log data cleaning stage parameters and metrics."""
    tracker.log_params(
        {"stage": "data_cleaning", "raw_data_path": raw_path, "cleaned_data_path": cleaned_path}
    )
    tracker.log_metrics(
        {
            "raw_rows": raw_rows,
            "cleaned_rows": cleaned_rows,
            "rows_dropped": raw_rows - cleaned_rows,
            "drop_rate": (raw_rows - cleaned_rows) / raw_rows if raw_rows > 0 else 0,
        }
    )
    tracker.log_artifact(cleaned_path, "data")


def log_content_features_stage(
    tracker: MLflowTracker,
    transformer,
    transformed_data: csr_matrix,
    feature_names: list,
    n_samples: int,
    n_features: int,
    params: dict[str, Any],
):
    """Log content-based feature engineering stage."""
    tracker.log_params(
        {"stage": "content_features", "n_samples": n_samples, "n_features": n_features, **params}
    )
    tracker.log_metrics(
        {
            "sparsity": 1.0 - (transformed_data.nnz / (n_samples * n_features)),
            "nnz": transformed_data.nnz,
        }
    )
    # Log transformer as model artifact
    tracker.log_model(transformer, "transformer", registered_model_name="content_transformer")
    # Log feature names
    tracker.log_dict({"feature_names": feature_names}, "feature_names.json")


def log_collaborative_stage(
    tracker: MLflowTracker,
    interaction_matrix: csr_matrix,
    n_tracks: int,
    n_users: int,
    params: dict[str, Any],
):
    """Log collaborative filtering matrix construction stage."""
    tracker.log_params(
        {"stage": "collaborative_filtering", "n_tracks": n_tracks, "n_users": n_users, **params}
    )
    tracker.log_metrics(
        {
            "matrix_nnz": interaction_matrix.nnz,
            "sparsity": 1.0 - (interaction_matrix.nnz / (n_tracks * n_users)),
            "avg_interactions_per_user": interaction_matrix.nnz / n_users if n_users > 0 else 0,
            "avg_interactions_per_track": interaction_matrix.nnz / n_tracks if n_tracks > 0 else 0,
        }
    )


def log_hybrid_stage(tracker: MLflowTracker, weight_content: float, params: dict[str, Any]):
    """Log hybrid model configuration."""
    tracker.log_params(
        {
            "stage": "hybrid",
            "weight_content": weight_content,
            "weight_collaborative": 1.0 - weight_content,
            **params,
        }
    )


def log_evaluation_metrics(tracker: MLflowTracker, results: dict[str, Any], prefix: str = ""):
    """Log evaluation metrics from evaluate.py results.

    Metric names are sanitized for MLflow compatibility: '@' is replaced
    with '_at_' since MLflow metric names only allow alphanumerics,
    underscores, dashes, periods, spaces, colons, and slashes.
    """
    for model_name, model_metrics in results.items():
        if model_name == "metadata":
            continue
        for k, metrics in model_metrics.items():
            for metric_name, value in metrics.items():
                # Sanitize metric name: replace @ with _at_ for MLflow compatibility
                safe_key = f"{prefix}{model_name}_{metric_name}_at_k{k}"
                tracker.log_metrics({safe_key: value})


# Context manager for automatic run management
class MlflowRun:
    """Context manager for MLflow runs."""

    def __init__(
        self,
        tracker: MLflowTracker,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags

    def __enter__(self):
        self.tracker.start_run(run_name=self.run_name, tags=self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = "FAILED" if exc_type else "FINISHED"
        self.tracker.end_run(status=status)
        return False


class NullContext:
    """No-op context manager for when MLflow is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


# Decorator for auto-tracking functions
def mlflow_track(
    tracker: MLflowTracker, run_name: str | None = None, tags: dict[str, str] | None = None
):
    """Decorator to automatically track a function with MLflow."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with MlflowRun(tracker, run_name or func.__name__, tags) as t:
                return func(*args, tracker=t, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    # Demo usage
    tracker = MLflowTracker("demo-experiment")

    with MlflowRun(tracker, "demo-run", {"type": "demo"}):
        tracker.log_params({"learning_rate": 0.01, "n_estimators": 100, "max_depth": 5})
        tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.92})
        tracker.log_dict({"config": "demo"}, "config.json")
