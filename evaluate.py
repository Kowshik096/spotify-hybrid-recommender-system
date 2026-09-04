"""
Model Evaluation Script for Hybrid Recommender System.

Computes Precision@K, Recall@K, NDCG@K, MAP@K for:
- Content-Based Filtering
- Item-Item Collaborative Filtering
- Hybrid Recommender

Note: This is a template demonstrating evaluation methodology.
Full evaluation on 962K users requires distributed computing (Spark/Dask/GPU).
Run with: python evaluate.py --sample 100  (requires significant compute)
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import load
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from mlflow_tracking import MlflowRun, MLflowTracker, NullContext, log_evaluation_metrics

_BASE = Path(__file__).parent


def load_artifacts() -> tuple[pd.DataFrame, Any, np.ndarray, pd.DataFrame, csr_matrix, csr_matrix]:
    songs_data = pd.read_csv(str(_BASE / "data" / "cleaned_data.csv"))
    transformed_data = load_npz(str(_BASE / "data" / "transformed_data.npz"))
    track_ids = load(str(_BASE / "data" / "track_ids.npy"), allow_pickle=True)
    filtered_data = pd.read_csv(str(_BASE / "data" / "collab_filtered_data.csv"))
    interaction_matrix = load_npz(str(_BASE / "data" / "interaction_matrix.npz"))
    transformed_hybrid_data = load_npz(str(_BASE / "data" / "transformed_hybrid_data.npz"))
    return (
        songs_data,
        transformed_data,
        track_ids,
        filtered_data,
        interaction_matrix,
        transformed_hybrid_data,
    )


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def ap_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def evaluate_model(
    name: str,
    get_recommendations_fn: Callable[[Any], list],
    test_users: list,
    user_items: dict,
    k_values: list,
) -> dict:
    """Generic evaluation function for any recommender model."""
    metrics: dict[int, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "ndcg": [], "ap": []} for k in k_values
    }

    for user_idx in tqdm(test_users, desc=f"{name} Evaluation"):
        relevant_tracks = user_items.get(user_idx, set())
        if not relevant_tracks:
            continue

        seed_track = next(iter(relevant_tracks))
        recommended = get_recommendations_fn(seed_track)

        for k in k_values:
            metrics[k]["precision"].append(precision_at_k(recommended, relevant_tracks, k))
            metrics[k]["recall"].append(recall_at_k(recommended, relevant_tracks, k))
            metrics[k]["ndcg"].append(ndcg_at_k(recommended, relevant_tracks, k))
            metrics[k]["ap"].append(ap_at_k(recommended, relevant_tracks, k))

    return {k: {m: float(np.mean(v)) for m, v in vals.items()} for k, vals in metrics.items()}


def create_content_recommender(
    songs_data: pd.DataFrame, transformed_data: Any
) -> Callable[[str], list]:
    """Factory for content-based recommender function.

    ``transformed_data`` is indexed by row POSITION in ``songs_data`` — the
    content pipeline transforms the catalog in its stored order. The
    collaborative ``track_to_idx`` (track_id -> position in the sorted
    ``track_ids`` array) is a *different ordering*: the collab index points
    at whichever track happens to sit at that position in the catalog, not
    the seed track. Using it here reads the wrong feature row and silently
    corrupts the content-based metrics, so we build the catalog index from
    ``songs_data`` itself.
    """
    # track_id -> positional index in songs_data (== transformed_data rows)
    track_to_catalog_idx = {str(tid): int(i) for i, tid in enumerate(songs_data["track_id"])}

    def recommend(seed_track: str, top_n: int = 100) -> list:
        idx = track_to_catalog_idx.get(str(seed_track))
        if idx is None:
            return []
        sims = cosine_similarity(transformed_data[idx].reshape(1, -1), transformed_data).ravel()
        top_indices = np.argsort(sims)[::-1]
        return [
            songs_data.iloc[i]["track_id"]
            for i in top_indices
            if songs_data.iloc[i]["track_id"] != seed_track
        ][:top_n]

    return recommend


def create_collab_recommender(
    interaction_matrix: Any, track_ids: np.ndarray, track_to_idx: dict
) -> Callable[[str], list]:
    """Factory for collaborative filtering recommender function."""

    def recommend(seed_track: str, top_n: int = 100) -> list:
        if seed_track not in track_to_idx:
            return []
        idx = track_to_idx[seed_track]
        sims = cosine_similarity(interaction_matrix[idx].reshape(1, -1), interaction_matrix).ravel()
        top_indices = np.argsort(sims)[::-1]
        return [track_ids[i] for i in top_indices if track_ids[i] != seed_track][:top_n]

    return recommend


def create_hybrid_recommender(
    songs_data: pd.DataFrame,
    filtered_data: pd.DataFrame,
    transformed_hybrid_data: Any,
    track_ids: np.ndarray,
    interaction_matrix: Any,
    track_to_idx: dict,
    weight_content: float = 0.5,
) -> Callable[[str], list]:
    """Factory for hybrid recommender function."""
    filtered_track_to_idx = {row["track_id"]: i for i, row in filtered_data.iterrows()}

    def recommend(seed_track: str, top_n: int = 100) -> list:
        if seed_track not in track_to_idx or seed_track not in filtered_track_to_idx:
            return []
        idx_c = filtered_track_to_idx[seed_track]
        idx_cf = track_to_idx[seed_track]

        content_sims = cosine_similarity(
            transformed_hybrid_data[idx_c].reshape(1, -1), transformed_hybrid_data
        ).ravel()
        collab_sims = cosine_similarity(
            interaction_matrix[idx_cf].reshape(1, -1), interaction_matrix
        ).ravel()

        content_sims = (content_sims - content_sims.min()) / (
            content_sims.max() - content_sims.min() + 1e-8
        )
        collab_sims = (collab_sims - collab_sims.min()) / (
            collab_sims.max() - collab_sims.min() + 1e-8
        )

        hybrid_sims = weight_content * content_sims + (1 - weight_content) * collab_sims
        top_indices = np.argsort(hybrid_sims)[::-1]
        return [track_ids[i] for i in top_indices if track_ids[i] != seed_track][:top_n]

    return recommend


def _leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions into train/test (leave-one-out style).

    For each user, the top half (at least two) of their highest-playcount
    tracks are held out as the test set; the remainder is train. Users with
    fewer than three interactions are train-only.

    The split is vectorized over the 962K users — a previous
    ``for _, group in df.groupby("user_idx")`` loop iterated a Python-level
    group per user and took >15 minutes.

    Returns:
        (train_df, test_df) — both without the helper columns.

    Raises:
        ValueError: if no test users can be produced (e.g. every user has
            fewer than 3 interactions).
    """
    df = df.sort_values(["user_idx", "playcount"], ascending=[True, False])
    df["rank"] = df.groupby("user_idx").cumcount()
    df["group_size"] = df.groupby("user_idx")["user_idx"].transform("size")
    # Hold out the top half of each user's highest-playcount tracks (at least
    # two) as the test set. The previous 20%-of-group-size rule left only one
    # test item per user, which made every metric 0.0: the seed track is
    # excluded from recommendations, so with a single relevant item recall is
    # mathematically 0 and precision/ndcg/ap are all 0.
    df["is_test"] = (df["group_size"] >= 3) & (df["rank"] < np.maximum(2, df["group_size"] // 2))
    train_df = df.loc[~df["is_test"]].drop(columns=["rank", "group_size", "is_test"])
    test_df = df.loc[df["is_test"]].drop(columns=["rank", "group_size", "is_test"])
    if test_df["user_idx"].nunique() == 0:
        raise ValueError(
            "No test users found after the leave-one-out split. The interaction "
            "matrix must contain users with >= 3 interactions to build a test set."
        )
    return train_df, test_df


def _run_evaluation(sample_users: int, tracker: Any, k_values: list[int] | None = None) -> None:
    """Run the evaluation logic."""
    if k_values is None:
        k_values = [5, 10, 20]

    # Load artifacts
    (
        songs_data,
        transformed_data,
        track_ids,
        filtered_data,
        interaction_matrix,
        transformed_hybrid_data,
    ) = load_artifacts()

    # Build user-item mapping from interaction matrix
    # Use scipy.sparse found arrays directly — avoids constructing a dense
    # intermediate and keeps memory footprint proportional to nnz.
    n_tracks, total_users = interaction_matrix.shape
    rows, cols = interaction_matrix.nonzero()
    values = interaction_matrix.data
    df = pd.DataFrame({"track_idx": rows, "user_idx": cols, "playcount": values})
    df["track_id"] = df["track_idx"].map(lambda i: track_ids[i])

    train_df, test_df = _leave_one_out_split(df)

    # Sample test users for demo (full eval needs distributed compute)
    test_users = test_df["user_idx"].unique()
    np.random.seed(42)
    sampled_users = np.random.choice(test_users, min(sample_users, len(test_users)), replace=False)
    sampled_users_list = sampled_users.tolist()
    test_df = test_df[test_df["user_idx"].isin(sampled_users_list)]

    user_items = test_df.groupby("user_idx")["track_id"].apply(set).to_dict()
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    # Create recommenders
    content_rec = create_content_recommender(songs_data, transformed_data)
    collab_rec = create_collab_recommender(interaction_matrix, track_ids, track_to_idx)
    hybrid_rec = create_hybrid_recommender(
        songs_data,
        filtered_data,
        transformed_hybrid_data,
        track_ids,
        interaction_matrix,
        track_to_idx,
        weight_content=0.5,
    )

    # Evaluate
    content_metrics = evaluate_model(
        "Content-Based", content_rec, sampled_users_list, user_items, k_values
    )
    collab_metrics = evaluate_model(
        "Collaborative", collab_rec, sampled_users_list, user_items, k_values
    )
    hybrid_metrics = evaluate_model(
        "Hybrid (w=0.5)", hybrid_rec, sampled_users_list, user_items, k_values
    )

    results = {
        "content_based": content_metrics,
        "collaborative": collab_metrics,
        "hybrid": hybrid_metrics,
        "metadata": {
            "n_users_total": total_users,
            "n_items": len(track_ids),
            "n_train_interactions": len(train_df),
            "n_test_interactions": len(test_df),
            "k_values": k_values,
            "sampled_users": len(sampled_users),
            "is_synthetic": False,
            "note": (
                "Real measurements on a random sample of users. Leave-one-out "
                "split holds out the top half (min 2) of each user's "
                "highest-playcount tracks; users with <3 interactions are "
                "train-only. Full evaluation on all 962K users requires "
                "distributed compute (Spark/Dask/GPU)."
            ),
        },
    }

    with open(str(_BASE / "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if tracker:
        # Log evaluation params
        tracker.log_params(
            {"sample_users": sample_users, "k_values": k_values, "weight_content": 0.5}
        )
        # Log metrics
        log_evaluation_metrics(tracker, results, prefix="eval_")


def main(
    sample_users: int = 50,
    use_mlflow: bool = False,
    tracking_uri: str | None = None,
    run_name: str | None = None,
) -> None:
    tracker = None
    if use_mlflow:
        tracker = MLflowTracker("spotify-hybrid-recsys", tracking_uri=tracking_uri)

    if tracker:
        with MlflowRun(tracker, run_name or "evaluation", {"stage": "evaluation"}):
            _run_evaluation(sample_users, tracker)
    else:
        with NullContext():
            _run_evaluation(sample_users, tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommender models")
    parser.add_argument(
        "--sample", type=int, default=50, help="Number of users to sample for evaluation"
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--tracking-uri", type=str, help="MLflow tracking URI")
    parser.add_argument("--run-name", type=str, help="MLflow run name")
    args = parser.parse_args()
    main(
        sample_users=args.sample,
        use_mlflow=args.mlflow,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
    )
