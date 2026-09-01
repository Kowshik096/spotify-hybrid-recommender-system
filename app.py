import os
import threading
import time
from difflib import get_close_matches
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import streamlit as st
from numpy import load, ndarray
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from scipy.sparse import csr_matrix, load_npz

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem

# Prometheus metrics
REQUEST_COUNT = Counter(
    "recsys_requests_total", "Total recommendation requests", ["model_type", "status"]
)
REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds", "Request latency in seconds", ["model_type"]
)
ACTIVE_USERS = Gauge("recsys_active_users", "Number of active users")
RECOMMENDATION_COUNT = Counter(
    "recsys_recommendations_total", "Total recommendations served", ["model_type"]
)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP server in background thread."""
    server = HTTPServer(("", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@st.cache_resource
def load_artifacts() -> tuple[
    pd.DataFrame, csr_matrix, ndarray, pd.DataFrame, csr_matrix, csr_matrix
]:
    """Load all pre-computed artifacts required by the recommender.

    Raises FileNotFoundError with an actionable message if any artifact
    is missing, distinguishing 'not generated yet' from 'file corrupted'.
    """
    artifacts = {
        "data/cleaned_data.csv": "song catalog (run: dvc repro)",
        "data/transformed_data.npz": "content-based similarity matrix (run: dvc repro)",
        "data/track_ids.npy": "track ID mapping (run: dvc repro)",
        "data/collab_filtered_data.csv": "filtered songs with listening history (run: dvc repro)",
        "data/interaction_matrix.npz": "collaborative filtering matrix (run: dvc repro)",
        "data/transformed_hybrid_data.npz": "hybrid content vectors (run: dvc repro)",
    }

    missing = [path for path in artifacts if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} artifact(s): {', '.join(missing)}. "
            f"Run `dvc pull` (or `dvc repro` if DVC remote is not configured) "
            f"to generate them from the raw data."
        )

    # load the data
    songs_data = pd.read_csv("data/cleaned_data.csv")

    # load the transformed data
    transformed_data = load_npz("data/transformed_data.npz")

    # load the track ids
    track_ids = load("data/track_ids.npy", allow_pickle=True)

    # load the filtered songs data
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")

    # load the interaction matrix
    interaction_matrix = load_npz("data/interaction_matrix.npz")

    # load the transformed hybrid data
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")

    return (
        songs_data,
        transformed_data,
        track_ids,
        filtered_data,
        interaction_matrix,
        transformed_hybrid_data,
    )


@st.cache_resource
def build_song_index(songs_data: pd.DataFrame) -> dict[str, tuple[str, str, str]]:
    # map lowercase "name — artist" display strings to (name, artist) pairs
    mapping = {}
    for song_name, artist_name in zip(songs_data["name"], songs_data["artist"], strict=False):
        display = f"{song_name} — {artist_name}"
        key = display.lower()
        if key not in mapping:
            mapping[key] = (song_name, artist_name, display)
    return mapping


def fuzzy_song_matches(
    query: str, mapping: dict[str, tuple[str, str, str]], limit: int = 10, cutoff: float = 0.4
) -> dict[str, tuple[str, str]]:
    # return {display_string: (song_name, artist_name)} for the closest matches
    query = query.strip().lower()
    if not query:
        return {}
    matches = get_close_matches(query, list(mapping.keys()), n=limit, cutoff=cutoff)
    return {mapping[key][2]: (mapping[key][0], mapping[key][1]) for key in matches}


# Title
st.title("Welcome to the Spotify Song Recommender!")

# Subheader
st.write("### Enter the name of a song and the recommender will suggest similar songs 🎵🎧")

# Start Prometheus metrics server (port from env or default)
METRICS_PORT = int(os.environ.get("METRICS_PORT", 9090))
if "metrics_server" not in st.session_state:
    try:
        st.session_state.metrics_server = start_metrics_server(METRICS_PORT)
    except OSError:
        # Port already in use (e.g. multiple Streamlit workers) — skip metrics
        st.session_state.metrics_server = None

# Load artifacts (cached, runs once per session)
(
    songs_data,
    transformed_data,
    track_ids,
    filtered_data,
    interaction_matrix,
    transformed_hybrid_data,
) = load_artifacts()

# Text Input with fuzzy search
song_query = st.text_input("Enter a song name:", placeholder="e.g. crazy in love")

song_mapping = build_song_index(songs_data)
matches = fuzzy_song_matches(song_query, song_mapping)

song_name, artist_name = "", ""
if song_query.strip() and not matches:
    st.info(f"No songs found matching '{song_query}'. Try a different spelling.")
elif matches:
    selection = st.selectbox("Did you mean:", list(matches.keys()))
    song_name, artist_name = matches[selection]
    st.write("You selected:", f"**{song_name.title()}** by **{artist_name.title()}**")

# k recommndations
k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

filtering_type = None
if song_name:
    if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
        # type of filtering
        filtering_type = "Hybrid Recommender System"

        # diversity slider
        diversity = st.slider(
            label="Diversity in Recommendations", min_value=1, max_value=9, value=5, step=1
        )

        content_based_weight = 1 - (diversity / 10)

        # plot a bar graph
        chart_data = pd.DataFrame(
            {"type": ["Personalized", "Diverse"], "ratio": [10 - diversity, diversity]}
        )

        st.bar_chart(chart_data, x="type", y="ratio")

    else:
        # type of filtering
        filtering_type = "Content-Based Filtering"

# Button
if filtering_type == "Content-Based Filtering":
    if st.button("Get Recommendations"):
        if ((songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)).any():
            st.write("Recommendations for", f"**{song_name}** by **{artist_name}**")
            ACTIVE_USERS.inc()
            start_time = time.time()
            try:
                recommendations = content_recommendation(
                    song_name=song_name,
                    artist_name=artist_name,
                    songs_data=songs_data,
                    transformed_data=transformed_data,
                    k=k,
                )
                REQUEST_LATENCY.labels(model_type="content_based").observe(time.time() - start_time)
                REQUEST_COUNT.labels(model_type="content_based", status="success").inc()
                RECOMMENDATION_COUNT.labels(model_type="content_based").inc(len(recommendations))

                # Display Recommendations
                for ind, recommendation in recommendations.iterrows():
                    rec_song_name = recommendation["name"].title()
                    rec_artist_name = recommendation["artist"].title()

                    if ind == 0:
                        st.markdown("## Currently Playing")
                        st.markdown(f"#### **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")
                    elif ind == 1:
                        st.markdown("### Next Up 🎵")
                        st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")
                    else:
                        st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation["spotify_preview_url"])
                        st.write("---")
            except ValueError as e:
                REQUEST_COUNT.labels(model_type="content_based", status="error").inc()
                st.error(str(e))
            except Exception as e:
                REQUEST_COUNT.labels(model_type="content_based", status="error").inc()
                st.error(f"An error occurred while generating recommendations: {str(e)}")
            finally:
                ACTIVE_USERS.dec()
        else:
            st.write(
                f"Sorry, we couldn't find {song_name} in our database. Please try another song."
            )

elif filtering_type == "Hybrid Recommender System":
    if st.button("Get Recommendations"):
        st.write("Recommendations for", f"**{song_name}** by **{artist_name}**")
        recommender = HybridRecommenderSystem(
            number_of_recommendations=k, weight_content_based=content_based_weight
        )

        # get the recommendations
        ACTIVE_USERS.inc()
        start_time = time.time()
        try:
            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                track_ids=track_ids,
                interaction_matrix=interaction_matrix,
            )
            REQUEST_LATENCY.labels(model_type="hybrid").observe(time.time() - start_time)
            REQUEST_COUNT.labels(model_type="hybrid", status="success").inc()
            RECOMMENDATION_COUNT.labels(model_type="hybrid").inc(len(recommendations))
            # Display Recommendations
            for ind, recommendation in recommendations.iterrows():
                rec_song_name = recommendation["name"].title()
                rec_artist_name = recommendation["artist"].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")
                elif ind == 1:
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")
                else:
                    st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation["spotify_preview_url"])
                    st.write("---")
        except ValueError as e:
            REQUEST_COUNT.labels(model_type="hybrid", status="error").inc()
            st.error(str(e))
        except Exception as e:
            REQUEST_COUNT.labels(model_type="hybrid", status="error").inc()
            st.error(f"An error occurred while generating recommendations: {str(e)}")
        finally:
            ACTIVE_USERS.dec()
