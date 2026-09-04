import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based

    def __calculate_content_based_similarities(
        self,
        song_name: str,
        artist_name: str,
        songs_data: pd.DataFrame,
        transformed_matrix: csr_matrix,
        cached_song_row: pd.DataFrame,
    ) -> NDArray:
        if cached_song_row.empty:
            raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")
        if len(cached_song_row) > 1:
            raise ValueError(
                f"Multiple songs match '{song_name}' by '{artist_name}' "
                f"({len(cached_song_row)} rows). The catalog must be "
                f"deduplicated on 'track_id' before recommending."
            )
        # Resolve the matched label to its POSITIONAL index. ``transformed_matrix``
        # is indexed by row position, so using ``cached_song_row.index[0]``
        # (a label) directly would read the wrong feature row on any
        # non-RangeIndex frame (e.g. after set_index / filtering / shuffling).
        song_index = int(songs_data.index.get_indexer([cached_song_row.index[0]])[0])
        if song_index < 0:
            raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")
        # generate the input vector
        input_vector = transformed_matrix[song_index].reshape(1, -1)
        # calculate similarity scores
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        return np.asarray(content_similarity_scores, dtype=np.float64)

    def __calculate_collaborative_filtering_similarities(
        self,
        song_name: str,
        artist_name: str,
        track_ids: NDArray,
        songs_data: pd.DataFrame,
        interaction_matrix: csr_matrix,
        cached_song_row: pd.DataFrame,
    ) -> NDArray:
        if cached_song_row.empty:
            raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")
        # track_id of input song
        input_track_id = cached_song_row["track_id"].values.item()
        # Cast to match track_ids dtype to avoid silent mismatch
        if track_ids.dtype.kind in ("i", "u", "f"):
            input_track_id = track_ids.dtype.type(input_track_id)
        # index value of track_id
        track_indices = np.where(track_ids == input_track_id)[0]
        if track_indices.size == 0:
            raise ValueError(
                f"Track ID '{input_track_id}' not found in collaborative filtering data"
            )
        ind = track_indices.item()
        # fetch the input vector
        input_array = interaction_matrix[ind]
        # get similarity scores
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return np.asarray(collaborative_similarity_scores, dtype=np.float64)

    def __normalize_similarities(self, similarity_scores: NDArray) -> NDArray:
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        # Guard against divide-by-zero when every score is identical (e.g. an
        # all-zero interaction row or a single-song matrix) — without the
        # epsilon the result would be NaN and poison the weighted ranking.
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum + 1e-8)
        return np.asarray(normalized_scores, dtype=np.float64)

    def __weighted_combination(
        self, content_based_scores: NDArray, collaborative_filtering_scores: NDArray
    ) -> NDArray:
        weighted_scores = (self.weight_content_based * content_based_scores) + (
            self.weight_collaborative * collaborative_filtering_scores
        )
        return weighted_scores

    def give_recommendations(
        self,
        song_name: str,
        artist_name: str,
        songs_data: pd.DataFrame,
        track_ids: NDArray,
        transformed_matrix: csr_matrix,
        interaction_matrix: csr_matrix,
    ) -> pd.DataFrame:
        if self.number_of_recommendations <= 0:
            raise ValueError(
                f"number_of_recommendations must be positive, got {self.number_of_recommendations}"
            )

        # Look up the song once — both similarity methods need the same row
        song_row = songs_data.loc[
            (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
        ]
        if song_row.empty:
            raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")

        # calculate content based similarities
        content_based_similarities = self.__calculate_content_based_similarities(
            song_name=song_name,
            artist_name=artist_name,
            songs_data=songs_data,
            transformed_matrix=transformed_matrix,
            cached_song_row=song_row,
        )

        # calculate collaborative filtering similarities
        collaborative_filtering_similarities = (
            self.__calculate_collaborative_filtering_similarities(
                song_name=song_name,
                artist_name=artist_name,
                track_ids=track_ids,
                songs_data=songs_data,
                interaction_matrix=interaction_matrix,
                cached_song_row=song_row,
            )
        )

        # normalize content based similarities
        normalized_content_based_similarities = self.__normalize_similarities(
            content_based_similarities
        )

        # normalize collaborative filtering similarities
        normalized_collaborative_filtering_similarities = self.__normalize_similarities(
            collaborative_filtering_similarities
        )

        # weighted combination of similarities
        weighted_scores = self.__weighted_combination(
            content_based_scores=normalized_content_based_similarities,
            collaborative_filtering_scores=normalized_collaborative_filtering_similarities,
        )

        # Rank every song by descending weighted similarity. The query song
        # always scores 1.0 against itself, so it would otherwise sit at the
        # top of the list; exclude it and return exactly k recommendations.
        # (The historical ``[-k - 1:]`` slice returned k+1 entries including
        # the seed — an off-by-one that surfaced as k+1 results in the UI.)
        seed_track_id = song_row["track_id"].values.item()
        if track_ids.dtype.kind in ("i", "u", "f"):
            seed_track_id = track_ids.dtype.type(seed_track_id)

        ranked_indices = np.argsort(weighted_scores.ravel())[::-1]
        ranked_track_ids = track_ids[ranked_indices]
        ranked_scores = weighted_scores.ravel()[ranked_indices]

        # drop the query song itself, then keep the top k
        keep = ranked_track_ids != seed_track_id
        recommendation_track_ids = ranked_track_ids[keep][: self.number_of_recommendations]
        top_scores = ranked_scores[keep][: self.number_of_recommendations]

        # get the songs from data
        scores_df = pd.DataFrame(
            {"track_id": recommendation_track_ids.tolist(), "score": top_scores.tolist()}
        )
        top_k_songs = (
            songs_data.loc[songs_data["track_id"].isin(recommendation_track_ids)]
            .merge(scores_df, on="track_id")
            .sort_values(by="score", ascending=False)
            .drop(columns=["track_id", "score"])
            .reset_index(drop=True)
        )

        return top_k_songs
