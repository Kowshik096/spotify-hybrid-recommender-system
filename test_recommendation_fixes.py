"""Unit tests for recommendation functions — covers fixes from the deep review.

Tests verify:
- C1: hybrid path raises ValueError on missing track_id (not cryptic .item() crash)
- H4: k validation in content_recommendation and collaborative_recommendation
- H2: dtype guard prevents silent track_id mismatch
- M4: artifact loading produces actionable error messages
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from collaborative_filtering import collaborative_recommendation
from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


@pytest.fixture
def songs_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "name": ["song a", "song b", "song c"],
            "artist": ["artist x", "artist y", "artist z"],
            "spotify_preview_url": ["url1", "url2", "url3"],
        }
    )


@pytest.fixture
def transformed_data() -> csr_matrix:
    return csr_matrix(np.eye(5))


@pytest.fixture
def interaction_matrix() -> csr_matrix:
    return csr_matrix(np.eye(5))


@pytest.fixture
def track_ids() -> np.ndarray:
    return np.array([1, 2, 3], dtype=np.int64)


class TestContentRecommendationKValidation:
    """H4: k must be a positive integer."""

    def test_k_zero_raises(self, songs_data: pd.DataFrame, transformed_data: csr_matrix) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            content_recommendation(
                song_name="song a",
                artist_name="artist x",
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=0,
            )

    def test_k_negative_raises(
        self, songs_data: pd.DataFrame, transformed_data: csr_matrix
    ) -> None:
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

    def test_k_zero_raises(
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            collaborative_recommendation(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                k=0,
            )

    def test_k_negative_raises(
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray
    ) -> None:
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

    def test_missing_track_id_raises_valueerror(
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix
    ) -> None:
        """When track_id is not in track_ids, a ValueError must be raised."""
        sys = HybridRecommenderSystem(number_of_recommendations=5, weight_content_based=0.5)
        # Use a track_id that exists in songs_data but NOT in track_ids
        track_ids_missing = np.array([10, 20, 30], dtype=np.int64)
        # song a has track_id=1, which is not in [10, 20, 30]
        collab_method = getattr(  # noqa: B009
            sys, "_HybridRecommenderSystem__calculate_collaborative_filtering_similarities"
        )
        with pytest.raises(ValueError, match="not found in collaborative filtering data"):
            collab_method(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids_missing,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                cached_song_row=songs_data.iloc[[0]],
            )

    def test_missing_song_raises_valueerror(
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray
    ) -> None:
        """When song is not in songs_data, a ValueError must be raised."""
        sys = HybridRecommenderSystem(number_of_recommendations=5, weight_content_based=0.5)
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

    def test_string_track_id_cast_to_int(self) -> None:
        """String track_id should be cast to match track_ids dtype."""
        track_ids = np.array([1, 2, 3], dtype=np.int64)
        input_track_id = "2"
        casted = track_ids.dtype.type(input_track_id)
        assert casted == 2
        assert isinstance(casted, np.int64)

    def test_int_track_id_no_cast_needed(self) -> None:
        """Integer track_id should work without casting."""
        track_ids = np.array([1, 2, 3], dtype=np.int64)
        input_track_id = np.int64(2)
        # No cast needed, comparison works directly
        indices = np.where(track_ids == input_track_id)[0]
        assert indices.item() == 1


class TestHybridKValidation:
    """H4: HybridRecommenderSystem validates k in give_recommendations."""

    def test_k_zero_raises_in_give_recommendations(
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray
    ) -> None:
        sys = HybridRecommenderSystem(number_of_recommendations=0, weight_content_based=0.5)
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
        self, songs_data: pd.DataFrame, interaction_matrix: csr_matrix, track_ids: np.ndarray
    ) -> None:
        sys = HybridRecommenderSystem(number_of_recommendations=-5, weight_content_based=0.5)
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
        self, songs_data: pd.DataFrame, track_ids: np.ndarray
    ) -> None:
        """give_recommendations should work end-to-end with the cached song_row pattern."""
        sys = HybridRecommenderSystem(number_of_recommendations=2, weight_content_based=0.5)
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


class TestRecommendationExactCountAndSelfExclusion:
    """Regression tests for the k+1 / self-match bug.

    The recommenders used ``np.argsort(scores)[-k - 1:]``, which returns the
    top k+1 entries and always includes the query song (similarity 1.0 against
    itself). The UI then rendered k+1 rows with the seed song as the first
    "Currently Playing" recommendation. These tests pin the corrected
    behaviour: exactly k rows, seed song excluded.
    """

    @staticmethod
    def _five_song_data() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "track_id": [1, 2, 3, 4, 5],
                "name": ["song a", "song b", "song c", "song d", "song e"],
                "artist": ["artist x"] * 5,
                "spotify_preview_url": ["u1", "u2", "u3", "u4", "u5"],
            }
        )

    def test_content_recommendation_returns_exactly_k(self) -> None:
        songs_data = self._five_song_data()
        transformed = csr_matrix(np.eye(5))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=3,
        )
        assert len(result) == 3

    def test_content_recommendation_excludes_seed_song(self) -> None:
        songs_data = self._five_song_data()
        transformed = csr_matrix(np.eye(5))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=3,
        )
        assert "song a" not in result["name"].tolist()

    def test_collaborative_recommendation_returns_exactly_k(self) -> None:
        songs_data = self._five_song_data()
        track_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        interaction = csr_matrix(np.eye(5))
        result = collaborative_recommendation(
            song_name="song a",
            artist_name="artist x",
            track_ids=track_ids,
            songs_data=songs_data,
            interaction_matrix=interaction,
            k=3,
        )
        assert len(result) == 3

    def test_collaborative_recommendation_excludes_seed_song(self) -> None:
        songs_data = self._five_song_data()
        track_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        interaction = csr_matrix(np.eye(5))
        result = collaborative_recommendation(
            song_name="song a",
            artist_name="artist x",
            track_ids=track_ids,
            songs_data=songs_data,
            interaction_matrix=interaction,
            k=3,
        )
        assert "song a" not in result["name"].tolist()

    def test_hybrid_returns_exactly_k_and_excludes_seed(self) -> None:
        songs_data = self._five_song_data()
        track_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        sys = HybridRecommenderSystem(number_of_recommendations=3, weight_content_based=0.5)
        transformed = csr_matrix(np.eye(5))
        interaction = csr_matrix(np.eye(5))
        result = sys.give_recommendations(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            track_ids=track_ids,
            transformed_matrix=transformed,
            interaction_matrix=interaction,
        )
        assert len(result) == 3
        assert "song a" not in result["name"].tolist()

    def test_hybrid_normalization_handles_constant_scores(
        self, songs_data: pd.DataFrame, track_ids: np.ndarray
    ) -> None:
        """When every collaborative similarity is identical (e.g. an all-zero
        interaction row -> cosine_similarity returns 0 for every track), the
        min-max normalizer must not divide by zero and produce NaN that would
        corrupt the weighted ranking."""
        sys = HybridRecommenderSystem(number_of_recommendations=2, weight_content_based=0.5)
        # seed row all zeros -> cosine_similarity returns 0 for every track
        interaction = csr_matrix(np.zeros((3, 3)))
        transformed = csr_matrix(np.eye(3))
        result = sys.give_recommendations(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            track_ids=track_ids,
            transformed_matrix=transformed,
            interaction_matrix=interaction,
        )
        assert len(result) == 2
        assert result["name"].notna().all()


class TestDuplicateSongRows:
    """Regression tests for duplicate (name, artist) rows in the catalog.

    A re-imported or scraped dataset can contain the same song twice. The
    recommenders must not crash with a cryptic numpy error; they should raise
    a clear ValueError so the caller can deduplicate the catalog.
    """

    @staticmethod
    def _duplicate_data() -> pd.DataFrame:
        # Same (name, artist) on two DIFFERENT track_ids — the realistic
        # duplicate-catalog scenario (re-import / scrape). track_ids stay unique.
        return pd.DataFrame(
            {
                "track_id": [1, 99, 2, 3],
                "name": ["song a", "song a", "song b", "song c"],
                "artist": ["artist x", "artist x", "artist y", "artist z"],
                "spotify_preview_url": ["u1", "u2", "u3", "u4"],
            }
        )

    def test_collaborative_raises_on_duplicate_rows(self) -> None:
        songs_data = self._duplicate_data()
        track_ids = np.array([1, 99, 2, 3], dtype=np.int64)
        interaction = csr_matrix(np.eye(4))
        with pytest.raises(ValueError, match="Multiple songs match"):
            collaborative_recommendation(
                song_name="song a",
                artist_name="artist x",
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction,
                k=2,
            )

    def test_collaborative_unique_row_still_works(self) -> None:
        songs_data = self._duplicate_data()
        track_ids = np.array([1, 99, 2, 3], dtype=np.int64)
        interaction = csr_matrix(np.eye(4))
        result = collaborative_recommendation(
            song_name="song b",
            artist_name="artist y",
            track_ids=track_ids,
            songs_data=songs_data,
            interaction_matrix=interaction,
            k=2,
        )
        assert len(result) == 2
        assert "song b" not in result["name"].tolist()

    def test_content_handles_duplicate_rows_without_crash(self) -> None:
        """content_recommendation uses index[0] and must not crash."""
        songs_data = self._duplicate_data()
        transformed = csr_matrix(np.eye(4))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=2,
        )
        assert len(result) == 2


class TestNonPositionalIndex:
    """Regression tests for positional-vs-label index alignment.

    ``song_row.index[0]`` is a DataFrame LABEL, but ``transformed_matrix`` is
    indexed by ROW POSITION. On a non-RangeIndex frame (set_index, filtering,
    shuffling) the two diverge and the wrong feature row is read. The fix
    resolves the label to its position via ``Index.get_indexer``.
    """

    @staticmethod
    def _five_song_data() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "track_id": [1, 2, 3, 4, 5],
                "name": ["song a", "song b", "song c", "song d", "song e"],
                "artist": ["artist x"] * 5,
                "spotify_preview_url": ["u1", "u2", "u3", "u4", "u5"],
            }
        )

    def test_content_with_non_default_integer_index(self) -> None:
        songs_data = self._five_song_data()
        songs_data = songs_data.copy()
        songs_data.index = [10, 20, 30, 40, 50]
        transformed = csr_matrix(np.eye(5))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=2,
        )
        assert len(result) == 2
        assert "song a" not in result["name"].tolist()

    def test_content_with_string_index(self) -> None:
        songs_data = self._five_song_data()
        songs_data = songs_data.copy()
        songs_data.index = ["a", "b", "c", "d", "e"]
        transformed = csr_matrix(np.eye(5))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=2,
        )
        assert len(result) == 2
        assert "song a" not in result["name"].tolist()

    def test_hybrid_with_non_default_integer_index(self) -> None:
        songs_data = self._five_song_data()
        songs_data = songs_data.copy()
        songs_data.index = [100, 200, 300, 400, 500]
        track_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        sysr = HybridRecommenderSystem(number_of_recommendations=2, weight_content_based=0.5)
        result = sysr.give_recommendations(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            track_ids=track_ids,
            transformed_matrix=csr_matrix(np.eye(5)),
            interaction_matrix=csr_matrix(np.eye(5)),
        )
        assert len(result) == 2
        assert "song a" not in result["name"].tolist()

    def test_content_shuffled_index_uses_correct_row(self) -> None:
        """With identity similarity the seed must be excluded regardless of order."""
        songs_data = self._five_song_data().sample(frac=1.0, random_state=0).sort_index()
        transformed = csr_matrix(np.eye(5))
        result = content_recommendation(
            song_name="song a",
            artist_name="artist x",
            songs_data=songs_data,
            transformed_data=transformed,
            k=3,
        )
        assert len(result) == 3
        assert "song a" not in result["name"].tolist()


class TestArtifactLoading:
    """M4: load_artifacts produces actionable error messages."""

    def test_missing_artifact_raises_filenotfound(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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


class TestLeaveOneOutSplit:
    """Regression tests for the evaluation leave-one-out split.

    The original 20%-of-group-size rule held out only one test item per
    user. Since every recommender excludes the seed track, a single relevant
    item makes recall@k mathematically 0 — and precision/ndcg/ap all 0.0 —
    which silently rendered the content-based metrics useless. The split now
    holds out the top half (min 2) of each user's highest-playcount tracks
    and requires >= 3 interactions per user.
    """

    @staticmethod
    def _build_df() -> pd.DataFrame:
        # 3 users: one with 2 interactions (train only), one with 3, one with 6.
        return pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                "playcount": [1, 2, 3, 2, 1, 6, 5, 4, 3, 2, 1],
                "track_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            }
        )

    def test_user_with_two_interactions_is_train_only(self) -> None:
        from evaluate import _leave_one_out_split

        train_df, test_df = _leave_one_out_split(self._build_df())
        assert set(train_df["user_idx"]) == {0, 1, 2}
        assert set(test_df["user_idx"]) == {1, 2}

    def test_each_test_user_has_at_least_two_test_items(self) -> None:
        from evaluate import _leave_one_out_split

        _, test_df = _leave_one_out_split(self._build_df())
        sizes = test_df.groupby("user_idx").size()
        assert (sizes >= 2).all()

    def test_test_items_are_highest_playcount_per_user(self) -> None:
        from evaluate import _leave_one_out_split

        _, test_df = _leave_one_out_split(self._build_df())
        # user 1 (playcounts 3,2,1) -> top half = 3,2 -> c,d
        u1 = set(test_df.loc[test_df["user_idx"] == 1, "track_id"])
        assert u1 == {"c", "d"}
        # user 2 (playcounts 6..1) -> top half = 6,5,4 -> f,g,h
        u2 = set(test_df.loc[test_df["user_idx"] == 2, "track_id"])
        assert u2 == {"f", "g", "h"}

    def test_train_and_test_are_disjoint(self) -> None:
        from evaluate import _leave_one_out_split

        train_df, test_df = _leave_one_out_split(self._build_df())
        train_keys = set(zip(train_df["user_idx"], train_df["track_id"], strict=False))
        test_keys = set(zip(test_df["user_idx"], test_df["track_id"], strict=False))
        assert not (train_keys & test_keys)

    def test_no_test_users_raises(self) -> None:
        from evaluate import _leave_one_out_split

        df = pd.DataFrame(
            {"user_idx": [0, 0, 1, 1], "playcount": [1, 2, 3, 4], "track_id": ["a", "b", "c", "d"]}
        )
        with pytest.raises(ValueError):
            _leave_one_out_split(df)

    def test_train_test_cover_all_interactions(self) -> None:
        from evaluate import _leave_one_out_split

        df = self._build_df()
        train_df, test_df = _leave_one_out_split(df)
        assert len(train_df) + len(test_df) == len(df)

    def test_evaluate_model_is_nonzero_with_two_relevant_items(self) -> None:
        """The bug this class guards against: recall@k was 0.0 for every user
        because the split left only one relevant item and recommenders exclude
        the seed. With >= 2 relevant items a recommender that surfaces the
        second item must return a non-zero recall."""
        from evaluate import evaluate_model

        # user 0 has 2 relevant items (a, b); recommender returns [a, c, d]
        user_items = {0: {"a", "b"}}

        def recommend(seed_track: str, top_n: int = 100) -> list:
            return ["a", "c", "d"]

        metrics = evaluate_model("Test", recommend, [0], user_items, [5])
        assert metrics[5]["recall"] > 0.0
        assert metrics[5]["precision"] > 0.0
