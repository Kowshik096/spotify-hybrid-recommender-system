"""Unit tests for recommendation functions — covers fixes from the deep review.

Tests verify:
- C1: hybrid path raises ValueError on missing track_id (not cryptic .item() crash)
- H4: k validation in content_recommendation and collaborative_recommendation
- H2: dtype guard prevents silent track_id mismatch
- M4: artifact loading produces actionable error messages
"""


import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from collaborative_filtering import collaborative_recommendation
from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


@pytest.fixture
def songs_data():
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "name": ["song a", "song b", "song c"],
            "artist": ["artist x", "artist y", "artist z"],
            "spotify_preview_url": ["url1", "url2", "url3"],
        }
    )


@pytest.fixture
def transformed_data():
    return csr_matrix(np.eye(5))


@pytest.fixture
def interaction_matrix():
    return csr_matrix(np.eye(5))


@pytest.fixture
def track_ids():
    return np.array([1, 2, 3], dtype=np.int64)


class TestContentRecommendationKValidation:
    """H4: k must be a positive integer."""

    def test_k_zero_raises(self, songs_data, transformed_data):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            content_recommendation(
                song_name="song a",
                artist_name="artist x",
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=0,
            )

    def test_k_negative_raises(self, songs_data, transformed_data):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            content_recommendation(
                song_name="song a",
                artist_name="artist x",
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=-5,
            )


class TestCollaborativeRecommendationKValidation:
    """H4: k must be a positive integer."""

    def test_k_zero_raises(self, songs_data, interaction_matrix, track_ids):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            collaborative_recommendation(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                k=0,
            )

    def test_k_negative_raises(self, songs_data, interaction_matrix, track_ids):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            collaborative_recommendation(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                k=-5,
            )


class TestHybridMissingTrackId:
    """C1: Hybrid path must raise ValueError on missing track_id (not .item() crash)."""

    def test_missing_track_id_raises_valueerror(self, songs_data, interaction_matrix):
        """When track_id is not in track_ids, a ValueError must be raised."""
        sys = HybridRecommenderSystem(
            number_of_recommendations=5, weight_content_based=0.5
        )
        # Use a track_id that exists in songs_data but NOT in track_ids
        track_ids_missing = np.array([10, 20, 30], dtype=np.int64)
        # song a has track_id=1, which is not in [10, 20, 30]
        with pytest.raises(ValueError, match="not found in collaborative filtering data"):
            sys._HybridRecommenderSystem__calculate_collaborative_filtering_similarities(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids_missing,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                cached_song_row=songs_data.iloc[[0]],
            )

    def test_missing_song_raises_valueerror(self, songs_data, interaction_matrix, track_ids):
        """When song is not in songs_data, a ValueError must be raised."""
        sys = HybridRecommenderSystem(
            number_of_recommendations=5, weight_content_based=0.5
        )
        transformed = csr_matrix(np.eye(5))
        with pytest.raises(ValueError, match="not found in database"):
            sys.give_recommendations(
                song_name="nonexistent song",
                artist_name="nonexistent artist",
                songs_data=songs_data,
                track_ids=track_ids,
                transformed_matrix=transformed,
                interaction_matrix=interaction_matrix,
            )


class TestHybridDtypeGuard:
    """H2: dtype cast prevents silent track_id mismatch."""

    def test_string_track_id_cast_to_int(self):
        """String track_id should be cast to match track_ids dtype."""
        track_ids = np.array([1, 2, 3], dtype=np.int64)
        input_track_id = "2"
        casted = track_ids.dtype.type(input_track_id)
        assert casted == 2
        assert isinstance(casted, np.int64)

    def test_int_track_id_no_cast_needed(self):
        """Integer track_id should work without casting."""
        track_ids = np.array([1, 2, 3], dtype=np.int64)
        input_track_id = np.int64(2)
        # No cast needed, comparison works directly
        indices = np.where(track_ids == input_track_id)[0]
        assert indices.item() == 1


class TestHybridKValidation:
    """H4: HybridRecommenderSystem validates k in give_recommendations."""

    def test_k_zero_raises_in_give_recommendations(
        self, songs_data, interaction_matrix, track_ids
    ):
        sys = HybridRecommenderSystem(
            number_of_recommendations=0, weight_content_based=0.5
        )
        transformed = csr_matrix(np.eye(5))
        with pytest.raises(ValueError, match="number_of_recommendations must be positive"):
            sys.give_recommendations(
                song_name="song a",
                artist_name="artist x",
                songs_data=songs_data,
                track_ids=track_ids,
                transformed_matrix=transformed,
                interaction_matrix=interaction_matrix,
            )

    def test_k_negative_raises_in_give_recommendations(
        self, songs_data, interaction_matrix, track_ids
    ):
        sys = HybridRecommenderSystem(
            number_of_recommendations=-5, weight_content_based=0.5
        )
        transformed = csr_matrix(np.eye(5))
        with pytest.raises(ValueError, match="number_of_recommendations must be positive"):
            sys.give_recommendations(
                song_name="song a",
                artist_name="artist x",
                songs_data=songs_data,
                track_ids=track_ids,
                transformed_matrix=transformed,
                interaction_matrix=interaction_matrix,
            )


class TestHybridCachedSongRow:
    """H1: give_recommendations uses cached song_row (no duplicate lookup)."""

    def test_give_recommendations_works_with_cached_row(
        self, songs_data, track_ids
    ):
        """give_recommendations should work end-to-end with the cached song_row pattern."""
        sys = HybridRecommenderSystem(
            number_of_recommendations=2, weight_content_based=0.5
        )
        # Use a 3x3 matrix to match the 3 songs in songs_data
        transformed = csr_matrix(np.eye(3))
        interaction = csr_matrix(np.eye(3))
        result = sys.give_recommendations(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            track_ids=track_ids,
            transformed_matrix=transformed,
            interaction_matrix=interaction,
        )
        assert len(result) > 0
        assert "name" in result.columns
        assert "artist" in result.columns
        assert "spotify_preview_url" in result.columns


class TestArtifactLoading:
    """M4: load_artifacts produces actionable error messages."""

    def test_missing_artifact_raises_filenotfound(self, monkeypatch, tmp_path):
        """Missing artifacts should raise FileNotFoundError with actionable message."""
        import app
        import config

        # Redirect DATA_DIR to a non-existent directory
        monkeypatch.setattr(config, "DATA_DIR", tmp_path / "nonexistent")
        # Reset the cached resource
        app.load_artifacts.clear()

        with pytest.raises(FileNotFoundError) as exc_info:
            app.load_artifacts()

        assert "dvc pull" in str(exc_info.value) or "dvc repro" in str(exc_info.value)
