import streamlit as st
from difflib import get_close_matches
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem


@st.cache_resource
def load_artifacts():
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

    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data


@st.cache_resource
def build_song_index(songs_data):
    # map lowercase "name — artist" display strings to (name, artist) pairs
    mapping = {}
    for song_name, artist_name in zip(songs_data["name"], songs_data["artist"]):
        display = f"{song_name} — {artist_name}"
        key = display.lower()
        if key not in mapping:
            mapping[key] = (song_name, artist_name, display)
    return mapping


def fuzzy_song_matches(query, mapping, limit=10, cutoff=0.4):
    # return {display_string: (song_name, artist_name)} for the closest matches
    query = query.strip().lower()
    if not query:
        return {}
    matches = get_close_matches(query, list(mapping.keys()), n=limit, cutoff=cutoff)
    return {mapping[key][2]: (mapping[key][0], mapping[key][1]) for key in matches}


# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Load artifacts (cached, runs once per session)
songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid_data = load_artifacts()

# Text Input with fuzzy search
song_query = st.text_input('Enter a song name:', placeholder='e.g. crazy in love')

song_mapping = build_song_index(songs_data)
matches = fuzzy_song_matches(song_query, song_mapping)

song_name, artist_name = "", ""
if song_query.strip() and not matches:
    st.info(f"No songs found matching '{song_query}'. Try a different spelling.")
elif matches:
    selection = st.selectbox('Did you mean:', list(matches.keys()))
    song_name, artist_name = matches[selection]
    st.write('You selected:', f"**{song_name.title()}** by **{artist_name.title()}**")

# k recommndations
k = st.selectbox('How many recommendations do you want?', [5,10,15,20], index=1)

filtering_type = None
if song_name:
    if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
        # type of filtering
        filtering_type = "Hybrid Recommender System"

        # diversity slider
        diversity = st.slider(label="Diversity in Recommendations",
                            min_value=1,
                            max_value=9,
                            value=5,
                            step=1)

        content_based_weight = 1 - (diversity / 10)

        # plot a bar graph
        chart_data = pd.DataFrame({
            "type" : ["Personalized", "Diverse"],
            "ratio": [10 - diversity, diversity]
        })

        st.bar_chart(chart_data,x="type",y="ratio")

    else:
        # type of filtering
        filtering_type = 'Content-Based Filtering'

# Button
if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((songs_data["name"] == song_name) & (songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
            recommendations = content_recommendation(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=songs_data,
                                                     transformed_data=transformed_data,
                                                     k=k)

            # Display Recommendations
            for ind , recommendation in recommendations.iterrows():
                rec_song_name = recommendation['name'].title()
                rec_artist_name = recommendation['artist'].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")

elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):
        st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")
        recommender = HybridRecommenderSystem(
                                            number_of_recommendations= k,
                                            weight_content_based= content_based_weight
                                            )

        # get the recommendations
        recommendations = recommender.give_recommendations(song_name= song_name,
                                                        artist_name= artist_name,
                                                        songs_data= filtered_data,
                                                        transformed_matrix= transformed_hybrid_data,
                                                        track_ids= track_ids,
                                                        interaction_matrix= interaction_matrix)
        # Display Recommendations
        for ind , recommendation in recommendations.iterrows():
            rec_song_name = recommendation['name'].title()
            rec_artist_name = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"#### **{rec_song_name}** by **{rec_artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            elif ind == 1:
                st.markdown("### Next Up 🎵")
                st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')
            else:
                st.markdown(f"#### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('---')