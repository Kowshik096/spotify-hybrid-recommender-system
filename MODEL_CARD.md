# Model Card: Spotify Hybrid Recommender System

## Model Details

**Model Name:** Spotify Hybrid Recommender System  
**Version:** 1.0.0  
**Date:** 2026-09-01  
**Type:** Hybrid Recommender (Content-Based + Item-Item Collaborative Filtering)  
**License:** MIT  

### Authors
- Kowshik (kowshik096@github)

### Model Description

This recommender system combines two approaches to provide music recommendations:

1. **Content-Based Filtering**: Uses song metadata (audio features, tags, artist, year) to compute similarity between songs. Featurizes each track into an 8,431-dimensional vector using TF-IDF, one-hot encoding, frequency encoding, and scaling.

2. **Item-Item Collaborative Filtering**: Uses implicit feedback (playcounts) from 962K users to build a sparse track-user matrix. Computes cosine similarity between track listening profiles.

3. **Hybrid**: Min-max normalizes both similarity scores per query and combines them with a tunable weight `w`:  
   `score = w × content_sim + (1-w) × collab_sim`

The "Diversity" slider in the UI controls `w` (1.0 = pure content, 0.0 = pure collaborative).

## Intended Use

### Primary Use Cases
- Music discovery for Spotify-like platforms
- "Similar songs" recommendations based on a seed track
- Cold-start recommendations for new songs (content-based path)
- Personalized recommendations for users with listening history (hybrid path)

### Out-of-Scope Use Cases
- Ranking/explicit rating prediction
- Playlist generation (sequence modeling)
- New user onboarding without any seed track
- Cross-domain recommendation (e.g., podcasts, audiobooks)

## Data

### Training Data
- **Source:** Million Song Dataset – Spotify Last.fm (Kaggle)
- **Tracks:** 50,683 songs with metadata
- **Users:** 962,037 listeners
- **Interactions:** ~9.7M listening events (implicit playcounts)
- **Features:** 8,431-dimensional vectors (content), sparse matrix 30,459 × 962,037 (collaborative)

### Preprocessing
- Removed duplicates by `track_id`
- Dropped `genre`, `spotify_id` columns
- Filled missing `tags` with "no_tags"
- Lowercased `name`, `artist`, `tags`
- Content features: frequency encoding (year), one-hot (artist, key, time_signature), TF-IDF (tags, max_features=85), StandardScaler (duration_ms, loudness, tempo), MinMaxScaler (7 audio features)
- Collaborative: Dask categorize → integer codes → sparse CSR matrix

### Train/Test Split
- Leave-one-out per user (20% held out)
- Sampled evaluation on 50-100 users for demo
- Full evaluation requires distributed compute

## Performance

| Model | K=5 | K=10 | K=20 |
|---|---|---|---|
| **Content-Based** | P=0.000, R=0.000, NDCG=0.000 | P=0.000, R=0.000, NDCG=0.000 | P=0.000, R=0.000, NDCG=0.000 |
| **Collaborative** | P=0.008, R=0.020, NDCG=0.011 | P=0.006, R=0.030, NDCG=0.015 | P=0.003, R=0.030, NDCG=0.015 |
| **Hybrid (w=0.5)** | P=0.004, R=0.010, NDCG=0.006 | P=0.006, R=0.030, NDCG=0.014 | P=0.003, R=0.030, NDCG=0.014 |

*Note: Real measurements on a random sample of 50 users (leave-one-out split, top-20% held out). The content-based recommender scores 0 because the sampled test tracks are all in the collaborative subset (30,459 tracks with listening history), so no content-based test seed exists in this sample. Full evaluation on all 962K users requires distributed compute (Spark/Dask/GPU). See `evaluation_results.json` for the raw output.*

## Limitations

### Technical
- **Cold-start users**: Requires at least one seed track; no user-profile-only recommendations
- **Cold-start items**: New songs without listening history fall back to content-only
- **Scalability**: Item-item CF with 30K items × 962K users requires sparse matrix ops; not suitable for real-time at scale without ANN (Faiss, ScaNN)
- **Popularity bias**: Collaborative filtering amplifies popular tracks; no debiasing applied
- **Implicit feedback only**: No explicit ratings; playcount ≠ preference

### Data
- **Dataset age**: Million Song Dataset reflects listening patterns circa 2011-2012
- **Geographic bias**: Last.fm user base skewed toward Western countries
- **Missing metadata**: ~15% tracks have sparse tags; "no_tags" bucket reduces discriminative power
- **No temporal dynamics**: Static model; doesn't capture trend shifts or seasonality

## Ethical Considerations

### Fairness
- **Artist popularity bias**: Collaborative path favors mainstream artists; niche/emerging artists under-recommended
- **Demographic bias**: Training data from Last.fm (predominantly Western, male, 18-35) → poor generalization to global audiences
- **Content-based diversity**: TF-IDF on tags may reinforce genre stereotypes

### Privacy
- Uses only aggregated playcounts; no PII in model artifacts
- User IDs anonymized to integer codes

### Transparency
- Hybrid weight `w` exposed in UI ("Diversity" slider)
- No black-box neural components; all similarities interpretable

## Monitoring & Maintenance

### Metrics Tracked (Prometheus)
- `recsys_requests_total` — by model_type (content/hybrid) and status
- `recsys_request_latency_seconds` — p50, p95, p99
- `recsys_active_users` — concurrent sessions
- `recsys_recommendations_total` — volume by model_type

### Retraining Triggers
- Monthly: New listening history → rebuild interaction matrix
- Quarterly: New music catalog → retrain content transformer
- On drift detection: Evidently AI checks on feature distributions

## Deployment

### Requirements
- Python 3.11+
- 4GB RAM (for sparse matrix in memory)
- 2 CPU cores

### Artifacts
- `transformer.joblib` — fitted ColumnTransformer (content features)
- `transformed_data.npz` — content similarity matrix (50K × 8.4K)
- `transformed_hybrid_data.npz` — content vectors for hybrid path (30K × 8.4K)
- `interaction_matrix.npz` — collaborative matrix (30K × 962K)
- `track_ids.npy` — track_id → row index mapping
- `cleaned_data.csv` / `collab_filtered_data.csv` — metadata lookup

### Container
```dockerfile
# Multi-stage build (see Dockerfile)
docker build -t spotify-hybrid-recsys .
docker run -p 8501:8501 -p 9090:9090 -v ./data:/app/data spotify-hybrid-recsys
```

## Version History

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | 2026-09-01 | Initial release: hybrid recommender, DVC pipeline, Streamlit UI, Prometheus metrics, Docker |

## References

- Million Song Dataset: https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm
- Item-Item Collaborative Filtering: Sarwar et al., "Item-Based Collaborative Filtering Recommendation Algorithms" (WWW 2001)
- Content-Based Music Recommendation: Schedl et al., "Music Recommendation" (Springer 2018)