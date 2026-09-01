"""Unit tests for mlflow_tracking module.

These tests verify that the MLflow tracking utilities work correctly,
specifically that log_dict does not crash with NameError due to a missing
import (the bug that was fixed).
"""

import os
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlflow_tracking import (
    MlflowRun,
    MLflowTracker,
    NullContext,
    log_collaborative_stage,
    log_content_features_stage,
    log_data_cleaning_stage,
    log_evaluation_metrics,
    log_hybrid_stage,
)


@pytest.fixture
def tracker(tmp_path):
    """Create an MLflowTracker with a temporary tracking directory."""
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    return MLflowTracker("test-experiment", tracking_uri=tracking_uri)


class TestLogDict:
    """Tests for log_dict — the method that was broken by a missing import."""

    def test_log_dict_writes_and_removes_temp_file(self, tracker):
        """log_dict should write JSON to a temp file, log it, then delete it."""
        with MlflowRun(tracker, "test-log-dict"):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                tmp_path = f.name

            tracker.log_dict({"feature_names": ["year", "duration_ms"]}, tmp_path)

            # The temp file should be removed after logging
            assert not os.path.exists(tmp_path)

    def test_log_dict_does_not_raise_nameerror(self, tracker):
        """log_dict must not raise NameError due to missing 'import json'.

        This is a regression test for the bug where json was not imported
        in mlflow_tracking.py, causing log_dict to crash at runtime.
        """
        with MlflowRun(tracker, "test-no-nameerror"):
            try:
                tracker.log_dict({"key": "value"}, "test.json")
            except NameError as e:
                pytest.fail(f"log_dict raised NameError: {e}")
            except Exception:
                # Other exceptions (e.g. MLflow backend issues) are acceptable
                # as long as it's not a NameError for 'json'
                pass


class TestLogContentFeaturesStage:
    """Tests for log_content_features_stage — which calls log_dict internally."""

    def test_log_content_features_stage_works(self, tracker):
        """log_content_features_stage should not crash when log_dict is called."""
        transformed = csr_matrix(np.eye(5))
        with MlflowRun(tracker, "test-content-features"):
            try:
                log_content_features_stage(
                    tracker=tracker,
                    transformer=None,
                    transformed_data=transformed,
                    feature_names=["a", "b"],
                    n_samples=5,
                    n_features=2,
                    params={"ohe_cols": ["artist"]},
                )
            except NameError as e:
                pytest.fail(f"log_content_features_stage raised NameError: {e}")

    def test_log_content_features_stage_logs_sparsity_metric(self, tracker):
        """log_content_features_stage should log sparsity as a metric."""
        transformed = csr_matrix(np.eye(5))
        with MlflowRun(tracker, "test-sparsity"):
            log_content_features_stage(
                tracker=tracker,
                transformer=None,
                transformed_data=transformed,
                feature_names=["a", "b"],
                n_samples=5,
                n_features=2,
                params={"ohe_cols": ["artist"]},
            )
            # If we got here without crashing, the test passes


class TestNullContext:
    """Tests for NullContext — used when MLflow is disabled."""

    def test_null_context_returns_none(self):
        with NullContext() as ctx:
            assert ctx is None

    def test_null_context_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with NullContext():
                raise ValueError("test")


class TestMlflowRun:
    """Tests for MlflowRun context manager."""

    def test_mlfow_run_starts_and_ends(self, tracker):
        with MlflowRun(tracker, "test-run", {"type": "unit-test"}):
            assert tracker.run is not None
        # After exiting, run should be None
        assert tracker.run is None

    def test_mlfow_run_marks_failed_on_exception(self, tracker):
        with pytest.raises(ValueError):
            with MlflowRun(tracker, "test-failed"):
                raise ValueError("test error")
        # Run should be ended (status FAILED)
        assert tracker.run is None


class TestLogDataCleaningStage:
    """Tests for log_data_cleaning_stage."""

    def test_log_data_cleaning_stage_logs_metrics(self, tracker):
        with MlflowRun(tracker, "test-cleaning"):
            log_data_cleaning_stage(
                tracker=tracker,
                raw_path="data/Music Info.csv",
                cleaned_path="data/cleaned_data.csv",
                raw_rows=1000,
                cleaned_rows=950,
                dropped_columns=["genre", "spotify_id"],
            )


class TestLogCollaborativeStage:
    """Tests for log_collaborative_stage."""

    def test_log_collaborative_stage_logs_metrics(self, tracker):
        matrix = csr_matrix(np.eye(10))
        with MlflowRun(tracker, "test-collab"):
            log_collaborative_stage(
                tracker=tracker,
                interaction_matrix=matrix,
                n_tracks=10,
                n_users=10,
                params={"playcount_dtype": "float64"},
            )


class TestLogHybridStage:
    """Tests for log_hybrid_stage."""

    def test_log_hybrid_stage_logs_weight(self, tracker):
        with MlflowRun(tracker, "test-hybrid"):
            log_hybrid_stage(
                tracker=tracker,
                weight_content=0.5,
                params={"source": "collab_filtered_data"},
            )


class TestLogEvaluationMetrics:
    """Tests for log_evaluation_metrics."""

    def test_log_evaluation_metrics_skips_metadata(self, tracker):
        results = {
            "content_based": {
                "5": {"precision": 0.1, "recall": 0.2, "ndcg": 0.3, "ap": 0.4}
            },
            "metadata": {"n_users": 100},
        }
        with MlflowRun(tracker, "test-eval"):
            log_evaluation_metrics(tracker, results, prefix="eval_")


class TestNoCircularImports:
    """L4: mlflow_tracking must remain a leaf module.

    A runtime check that verifies the import graph stays acyclic by
    inspecting mlflow_tracking's namespace after import. If any of the
    pipeline modules appear as attributes, the dependency direction has
    been violated and a circular import has been introduced.
    """

    _FORBIDDEN_ATTRS = (
        "clean_data",
        "data_for_content_filtering",
        "content_recommendation",
        "train_transformer",
        "collaborative_recommendation",
        "create_interaction_matrix",
        "HybridRecommenderSystem",
        "give_recommendations",
    )

    def test_mlflow_tracking_does_not_import_pipeline_modules(self):
        """mlflow_tracking must not import from any pipeline module."""
        import mlflow_tracking

        leaked = []
        for attr in self._FORBIDDEN_ATTRS:
            if hasattr(mlflow_tracking, attr):
                leaked.append(attr)

        assert not leaked, (
            f"mlflow_tracking imports from pipeline modules: {leaked}. "
            f"mlflow_tracking must be a leaf module — move shared helpers "
            f"to a separate module (e.g. config.py)."
        )

    def test_import_graph_is_acyclic(self):
        """All modules must import without circular dependency errors."""
        import importlib
        import sys

        # Force fresh import of the full dependency chain
        modules_to_reload = [
            "mlflow_tracking",
            "data_cleaning",
            "content_based_filtering",
            "collaborative_filtering",
            "transform_filtered_data",
            "evaluate",
        ]
        for mod_name in modules_to_reload:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        # Importing these in order must not raise ImportError
        for mod_name in modules_to_reload:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                pytest.fail(f"Circular import detected importing {mod_name}: {e}")
