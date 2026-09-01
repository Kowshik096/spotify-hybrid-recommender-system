from pathlib import Path

import pandas as pd

from content_based_filtering import save_transformed_data, transform_data
from data_cleaning import data_for_content_filtering
from mlflow_tracking import MlflowRun, MLflowTracker, NullContext, log_hybrid_stage

# path of filtered data
_BASE = Path(__file__).parent
filtered_data_path = str(_BASE / "data" / "collab_filtered_data.csv")

# save path
save_path = str(_BASE / "data" / "transformed_hybrid_data.npz")


def main(
    data_path: str, save_path: str, use_mlflow: bool = False, tracking_uri: str | None = None
) -> None:
    tracker = None
    if use_mlflow:
        tracker = MLflowTracker("spotify-hybrid-recsys", tracking_uri=tracking_uri)

    with (
        MlflowRun(tracker, "hybrid_features", {"stage": "hybrid_features"})
        if tracker
        else NullContext()
    ):
        # load the filtered data
        filtered_data = pd.read_csv(data_path)

        # clean the data
        filtered_data_cleaned = data_for_content_filtering(filtered_data)

        # transform the data into matrix
        transformed_data = transform_data(filtered_data_cleaned)

        # save the transformed data
        save_transformed_data(transformed_data, save_path)

        if tracker:
            log_hybrid_stage(
                tracker=tracker,
                weight_content=0.5,  # default, actual weight set at inference
                params={
                    "source": "collab_filtered_data",
                    "transformer": "content_transformer (shared)",
                },
            )


if __name__ == "__main__":
    main(filtered_data_path, save_path)
