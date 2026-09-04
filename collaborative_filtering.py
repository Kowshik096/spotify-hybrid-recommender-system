from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

from mlflow_tracking import MLflowTracker, get_mlflow_context, log_collaborative_stage

# set paths (resolved relative to project root, not CWD)
_BASE = Path(__file__).parent
track_ids_save_path = str(_BASE / "data" / "track_ids.npy")
filtered_data_save_path = str(_BASE / "data" / "collab_filtered_data.csv")
interaction_matrix_save_path = str(_BASE / "data" / "interaction_matrix.npz")
songs_data_path = str(_BASE / "data" / "cleaned_data.csv")
user_listening_history_data_path = str(_BASE / "data" / "User Listening History.csv")


def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
    """
    Filter the songs data for the given track ids
    """
    # filter data based on track_ids
    filtered_data = songs_data[songs_data["track_id"].isin(track_ids)].copy()
    # sort the data by track id
    filtered_data.sort_values(by="track_id", inplace=True)
    # rest index
    filtered_data.reset_index(drop=True, inplace=True)
    # save the data
    save_pandas_data_to_csv(filtered_data, save_df_path)

    return filtered_data


def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the data to a csv file
    """
    data.to_csv(file_path, index=False)


def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
    """
    Save the sparse matrix to a npz file
    """
    save_npz(file_path, matrix)


def create_interaction_matrix(
    history_data: dd.DataFrame, track_ids_save_path: str, save_matrix_path: str
) -> csr_matrix:
    # make a copy of data
    df = history_data.copy()

    # convert the playcount column to float
    df["playcount"] = df["playcount"].astype(np.float64)

    # convert string column to categorical
    df = df.categorize(columns=["user_id", "track_id"])

    # Convert user_id and track_id to numeric indices
    user_mapping = df["user_id"].cat.codes
    track_mapping = df["track_id"].cat.codes

    # get the list of track_ids
    track_ids = df["track_id"].cat.categories.values

    # save the categories
    np.save(track_ids_save_path, track_ids, allow_pickle=True)

    # add the index columns to the dataframe
    df = df.assign(user_idx=user_mapping, track_idx=track_mapping)

    # create the interaction matrix
    interaction_matrix = df.groupby(["track_idx", "user_idx"])["playcount"].sum().reset_index()

    # compute the matrix
    interaction_matrix = interaction_matrix.compute()

    # get the indices to form sparse matrix
    row_indices = interaction_matrix["track_idx"]
    col_indices = interaction_matrix["user_idx"]
    values = interaction_matrix["playcount"]

    # get the shape of sparse matrix
    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()

    # create the sparse matrix
    interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))

    # save the sparse matrix
    save_sparse_matrix(interaction_matrix, save_matrix_path)


def collaborative_recommendation(
    song_name: str,
    artist_name: str,
    track_ids: np.ndarray,
    songs_data: pd.DataFrame,
    interaction_matrix: csr_matrix,
    k: int = 5,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    # lowercase the song name
    song_name = song_name.lower()

    # lowercase the artist name
    artist_name = artist_name.lower()

    # fetch the row from songs data
    song_row = songs_data.loc[
        (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
    ]
    if song_row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")
    if len(song_row) > 1:
        raise ValueError(
            f"Multiple songs match '{song_name}' by '{artist_name}' "
            f"({len(song_row)} rows). The catalog must be deduplicated on "
            f"'track_id' before recommending."
        )

    # track_id of input song
    input_track_id = song_row["track_id"].values.item()
    # Cast to match track_ids dtype so a string track_id read from CSV does
    # not silently fail to match an integer track_ids array (H2).
    if track_ids.dtype.kind in ("i", "u", "f"):
        input_track_id = track_ids.dtype.type(input_track_id)

    # index value of track_id
    track_indices = np.where(track_ids == input_track_id)[0]
    if track_indices.size == 0:
        raise ValueError(f"Track ID '{input_track_id}' not found in collaborative filtering data")
    ind = track_indices.item()

    # fetch the input vector
    input_array = interaction_matrix[ind]

    # get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    # Rank every track by descending similarity. The seed track scores 1.0
    # against itself, so it must be excluded and exactly k returned.
    # (The previous ``[-k - 1:]`` slice returned k+1 entries including the
    # seed — an off-by-one visible as k+1 rows in the UI.)
    ranked_indices = np.argsort(similarity_scores.ravel())[::-1]
    ranked_track_ids = track_ids[ranked_indices]
    keep = ranked_track_ids != input_track_id
    recommendation_track_ids = ranked_track_ids[keep][:k]

    # get top scores
    top_scores = np.sort(similarity_scores.ravel())[::-1][keep][:k]

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


def main(use_mlflow: bool = False, tracking_uri: str | None = None) -> None:
    tracker = None
    if use_mlflow:
        tracker = MLflowTracker("spotify-hybrid-recsys", tracking_uri=tracking_uri)

    with get_mlflow_context(
        tracker, "collaborative_filtering", {"stage": "collaborative_filtering"}
    ):
        # load the history data
        user_data = dd.read_csv(user_listening_history_data_path)

        # get the unique track ids
        unique_track_ids = user_data.loc[:, "track_id"].unique().compute()
        unique_track_ids = unique_track_ids.tolist()

        # filter the songs data
        songs_data = pd.read_csv(songs_data_path)
        filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)

        # create the interaction matrix
        interaction_matrix = create_interaction_matrix(
            user_data, track_ids_save_path, interaction_matrix_save_path
        )

        if tracker:
            log_collaborative_stage(
                tracker=tracker,
                interaction_matrix=interaction_matrix,
                n_tracks=interaction_matrix.shape[0],
                n_users=interaction_matrix.shape[1],
                params={
                    "playcount_dtype": "float64",
                    "aggregation": "sum",
                    "categorize_columns": ["user_id", "track_id"],
                },
            )


if __name__ == "__main__":
    main()
