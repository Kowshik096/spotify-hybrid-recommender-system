from pathlib import Path

import pandas as pd

from mlflow_tracking import MLflowTracker, get_mlflow_context, log_data_cleaning_stage

DATA_PATH = str(Path(__file__).parent / "data" / "Music Info.csv")
CLEANED_DATA_PATH = str(Path(__file__).parent / "data" / "cleaned_data.csv")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return (
        data.drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"])
        .fillna({"tags": "no_tags"})
        .assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower(),
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare the data for content based
    filtering by removing unnecessary features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return data.drop(columns=["track_id", "name", "spotify_preview_url"])


def main(data_path: str, use_mlflow: bool = False, tracking_uri: str | None = None) -> None:
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    use_mlflow (bool): Whether to enable MLflow tracking.
    tracking_uri (str): MLflow tracking server URI.
    Returns:
    None
    """
    tracker = None
    if use_mlflow:
        tracker = MLflowTracker("spotify-hybrid-recsys", tracking_uri=tracking_uri)

    with get_mlflow_context(tracker, "data_cleaning", {"stage": "data_cleaning"}):
        # load the data
        data = pd.read_csv(data_path)
        raw_rows = len(data)

        # perform data cleaning
        cleaned_data = clean_data(data)
        cleaned_rows = len(cleaned_data)

        # saved cleaned data
        cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)

        if tracker:
            log_data_cleaning_stage(
                tracker=tracker,
                raw_path=data_path,
                cleaned_path=CLEANED_DATA_PATH,
                raw_rows=raw_rows,
                cleaned_rows=cleaned_rows,
                dropped_columns=["genre", "spotify_id"],
            )


if __name__ == "__main__":
    main(DATA_PATH)
