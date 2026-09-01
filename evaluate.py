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
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from numpy import load
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_artifacts():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    transformed_data = load_npz("data/transformed_data.npz")
    track_ids = load("data/track_ids.npy", allow_pickle=True)
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data


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
    get_recommendations_fn,
    test_users: list,
    user_items: dict,
    k_values: list
) -> dict:
    """Generic evaluation function for any recommender model."""
    metrics = {k: {"precision": [], "recall": [], "ndcg": [], "ap": []} for k in k_values}

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


def create_content_recommender(songs_data, transformed_data, track_to_idx):
    """Factory for content-based recommender function."""
    def recommend(seed_track: str, top_n: int = 100) -> list:
        if seed_track not in track_to_idx:
            return []
        idx = track_to_idx[seed_track]
        sims = cosine_similarity(transformed_data[idx].reshape(1, -1), transformed_data).ravel()
        top_indices = np.argsort(sims)[::-1]
        return [songs_data.iloc[i]["track_id"] for i in top_indices if songs_data.iloc[i]["track_id"] != seed_track][:top_n]
    return recommend


def create_collab_recommender(interaction_matrix, track_ids, track_to_idx):
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
    songs_data, filtered_data, transformed_hybrid_data,
    track_ids, interaction_matrix, track_to_idx, weight_content: float = 0.5
):
    """Factory for hybrid recommender function."""
    filtered_track_to_idx = {row["track_id"]: i for i, row in filtered_data.iterrows()}
    def recommend(seed_track: str, top_n: int = 100) -> list:
        if seed_track not in track_to_idx or seed_track not in filtered_track_to_idx:
            return []
        idx_c = filtered_track_to_idx[seed_track]
        idx_cf = track_to_idx[seed_track]

        content_sims = cosine_similarity(transformed_hybrid_data[idx_c].reshape(1, -1), transformed_hybrid_data).ravel()
        collab_sims = cosine_similarity(interaction_matrix[idx_cf].reshape(1, -1), interaction_matrix).ravel()

        content_sims = (content_sims - content_sims.min()) / (content_sims.max() - content_sims.min() + 1e-8)
        collab_sims = (collab_sims - collab_sims.min()) / (collab_sims.max() - collab_sims.min() + 1e-8)

        hybrid_sims = weight_content * content_sims + (1 - weight_content) * collab_sims
        top_indices = np.argsort(hybrid_sims)[::-1]
        return [track_ids[i] for i in top_indices if track_ids[i] != seed_track][:top_n]
    return recommend


def main(sample_users: int = 50):
    print("Loading artifacts...")
    songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data = load_artifacts()

    # Build user-item mapping from interaction matrix
    n_tracks, n_users = interaction_matrix.shape
    rows, cols = interaction_matrix.nonzero()
    values = interaction_matrix.data
    df = pd.DataFrame({"track_idx": rows, "user_idx": cols, "playcount": values})
    df["track_id"] = df["track_idx"].map(lambda i: track_ids[i])

    # Leave-one-out split per user
    train_rows, test_rows = [], []
    for user_id, group in df.groupby("user_idx"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        test = group.nlargest(max(1, int(len(group) * 0.2)), "playcount")
        train = group.drop(test.index)
        test_rows.append(test)
        train_rows.append(train)

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    # Sample test users for demo (full eval needs distributed compute)
    test_users = test_df["user_idx"].unique()
    np.random.seed(42)
    sampled_users = np.random.choice(test_users, min(sample_users, len(test_users)), replace=False)
    test_df = test_df[test_df["user_idx"].isin(sampled_users)]

    user_items = test_df.groupby("user_idx")["track_id"].apply(set).to_dict()
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    k_values = [5, 10, 20]

    print(f"\nEvaluating on {len(sampled_users)} sampled users...\n")

    # Create recommenders
    content_rec = create_content_recommender(songs_data, transformed_data, track_to_idx)
    collab_rec = create_collab_recommender(interaction_matrix, track_ids, track_to_idx)
    hybrid_rec = create_hybrid_recommender(
        songs_data, filtered_data, transformed_hybrid_data,
        track_ids, interaction_matrix, track_to_idx, weight_content=0.5
    )

    # Evaluate
    content_metrics = evaluate_model("Content-Based", content_rec, sampled_users, user_items, k_values)
    collab_metrics = evaluate_model("Collaborative", collab_rec, sampled_users, user_items, k_values)
    hybrid_metrics = evaluate_model("Hybrid (w=0.5)", hybrid_rec, sampled_users, user_items, k_values)

    results = {
        "content_based": content_metrics,
        "collaborative": collab_metrics,
        "hybrid": hybrid_metrics,
        "metadata": {
            "n_users_total": n_users,
            "n_items": len(track_ids),
            "n_train_interactions": len(train_df),
            "n_test_interactions": len(test_df),
            "k_values": k_values,
            "sampled_users": len(sampled_users),
            "note": "Sampled evaluation for demo. Full eval requires distributed compute."
        }
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results Summary ===")
    for model_name, model_metrics in [("Content-Based", content_metrics), ("Collaborative", collab_metrics), ("Hybrid", hybrid_metrics)]:
        print(f"\n{model_name}:")
        for k in k_values:
            m = model_metrics[k]
            print(f"  K={k}: P={m['precision']:.4f}, R={m['recall']:.4f}, NDCG={m['ndcg']:.4f}, MAP={m['ap']:.4f}")

    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommender models")
    parser.add_argument("--sample", type=int, default=50, help="Number of users to sample for evaluation")
    args = parser.parse_args()
    main(sample_users=args.sample)