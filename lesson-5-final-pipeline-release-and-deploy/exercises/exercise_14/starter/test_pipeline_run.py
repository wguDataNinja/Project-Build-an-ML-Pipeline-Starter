import mlflow
import os
import subprocess

# Use this to fully bypass shell parsing problems
def test_mlflow_pipeline():
    print("Running MLflow programmatically...")

    try:
        mlflow.projects.run(
            uri=".",   # run from current directory
            entry_point="main",
            parameters={"steps": "download"},
            env_manager="local"
        )
    except Exception as e:
        print(f"MLflow run failed: {e}")

if __name__ == "__main__":
    test_mlflow_pipeline()