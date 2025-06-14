# starter/basic_cleaning/run.py

import argparse
import logging
import wandb
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact...")
    local_path = wandb.use_artifact(args.input_artifact).file()

    logger.info("Reading input data...")
    df = pd.read_parquet(local_path)

    logger.info("Dropping columns with high missingness...")
    df = df.drop(columns=["title", "song_name"])

    logger.info("Imputing missing numeric values...")
    numeric_columns = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    logger.info("Filtering out rows outside proper geolocation boundaries...")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned dataset...")
    df.to_parquet("clean_data.parquet", index=False)

    logger.info("Uploading cleaned dataset as artifact...")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_data.parquet")
    run.log_artifact(artifact)

    logger.info("Finished basic cleaning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw data")

    parser.add_argument("--input_artifact", type=str, help="Input raw data artifact", required=True)
    parser.add_argument("--output_artifact", type=str, help="Output cleaned data artifact", required=True)
    parser.add_argument("--output_type", type=str, help="Type of artifact (usually 'cleaned_data')", required=True)
    parser.add_argument("--output_description", type=str, help="Description for the cleaned artifact", required=True)

    args = parser.parse_args()

    go(args)