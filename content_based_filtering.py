import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz, csr_matrix
from mlflow_tracking import MLflowTracker, MlflowRun, log_content_features_stage, NullContext

# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# cols to transform
frequency_enode_cols = ['year']
ohe_cols = ['artist',"time_signature","key"]
tfidf_col = 'tags'
standard_scale_cols = ["duration_ms","loudness","tempo"]
min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]


def train_transformer(data: pd.DataFrame) -> None:
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
    - Frequency Encoding using CountEncoder on specified columns.
    - One-Hot Encoding using OneHotEncoder on specified columns.
    - TF-IDF Vectorization using TfidfVectorizer on a specified column.
    - Standard Scaling using StandardScaler on specified columns.
    - Min-Max Scaling using MinMaxScaler on specified columns.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    Saves:
    transformer.joblib: The trained ColumnTransformer object.
    """
    # transformer 
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_enode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    # fit the transformer
    transformer.fit(data)

    # save the transformer
    joblib.dump(transformer, "transformer.joblib")
    

def transform_data(data: pd.DataFrame) -> csr_matrix:
    """
    Transforms the input data using a pre-trained transformer.
    Args:
        data (pd.DataFrame): The data to be transformed.
    Returns:
        csr_matrix: The transformed data.
    """
    # load the transformer
    transformer = joblib.load("transformer.joblib")
    
    # transform the data
    transformed_data = transformer.transform(data)
    
    return transformed_data


def save_transformed_data(transformed_data: csr_matrix, save_path: str) -> None:
    """
    Save the transformed data to a specified file path.

    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.

    Returns:
    None
    """
    # save the transformed data
    save_npz(save_path, transformed_data)


def calculate_similarity_scores(input_vector: np.ndarray, data: csr_matrix) -> np.ndarray:
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (np.ndarray): The input vector for which similarity scores are to be calculated.
        data (csr_matrix): The dataset against which the similarity scores are to be calculated.
    Returns:
        np.ndarray: An array of similarity scores.
    """
    # calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, data)
    
    return similarity_scores


def content_recommendation(
    song_name: str,
    artist_name: str,
    songs_data: pd.DataFrame,
    transformed_data: csr_matrix,
    k: int = 10
) -> pd.DataFrame:
    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    artist_name (str): The name of the artist of the song.
    songs_data (pd.DataFrame): The DataFrame containing song information.
    transformed_data (csr_matrix): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    pd.DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    # convert song name to lowercase
    song_name = song_name.lower()
    # convert the artist name to lowercase
    artist_name = artist_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    if song_row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found in database")
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list


def main(data_path: str, use_mlflow: bool = False, tracking_uri: str = None) -> None:
    """
    Test the recommendations for a given song using content-based filtering.

    Parameters:
    data_path (str): The path to the CSV file containing the song data.
    use_mlflow (bool): Whether to enable MLflow tracking.
    tracking_uri (str): MLflow tracking server URI.

    Returns:
    None: Prints the top k recommended songs based on content similarity.
    """
    tracker = None
    if use_mlflow:
        tracker = MLflowTracker("spotify-hybrid-recsys", tracking_uri=tracking_uri)

    with MlflowRun(tracker, "content_features", {"stage": "content_features"}) if tracker else NullContext() as t:
        # load the data
        data = pd.read_csv(data_path)
        # clean the data
        data_content_filtering = data_for_content_filtering(data)
        # train the transformer
        train_transformer(data_content_filtering)
        # transform the data
        transformed_data = transform_data(data_content_filtering)
        # save transformed data
        save_transformed_data(transformed_data, "data/transformed_data.npz")

        if tracker:
            # Log transformer params
            transformer = joblib.load("transformer.joblib")
            log_content_features_stage(
                tracker=tracker,
                transformer=transformer,
                transformed_data=transformed_data,
                feature_names=list(transformer.get_feature_names_out()),
                n_samples=transformed_data.shape[0],
                n_features=transformed_data.shape[1],
                params={
                    "frequency_encode_cols": frequency_enode_cols,
                    "ohe_cols": ohe_cols,
                    "tfidf_max_features": 85,
                    "standard_scale_cols": standard_scale_cols,
                    "min_max_scale_cols": min_max_scale_cols
                }
            )


if __name__ == "__main__":
    main(CLEANED_DATA_PATH)