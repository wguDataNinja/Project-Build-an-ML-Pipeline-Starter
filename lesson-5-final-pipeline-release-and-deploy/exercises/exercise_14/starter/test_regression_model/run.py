# filename: run.py

import argparse
import logging
import os
import pandas as pd
import mlflow
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="test_model")

    logger.info("Downloading model artifact...")
    artifact = run.use_artifact(args.mlflow_model)
    model_path = artifact.download()
    model = mlflow.sklearn.load_model(model_path)

    logger.info("Downloading test dataset...")
    test_artifact = run.use_artifact(args.test_artifact)
    test_path = test_artifact.file()
    df_test = pd.read_csv(test_path)

    logger.info("Preparing test data...")
    y_test = df_test.pop("genre")
    preds = model.predict(df_test)

    logger.info("Scoring model...")
    accuracy = (preds == y_test).mean()

    run.summary["test_accuracy"] = accuracy
    logger.info(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mlflow_model", type=str, required=True, help="MLflow model artifact (prod alias)")
    parser.add_argument("--test_artifact", type=str, required=True, help="Test dataset artifact")

    args = parser.parse_args()

    go(args)