# main.py

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import copy  

# Make sure WANDB_API_KEY is passed to all subprocesses
if "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_API_KEY"] = "d334422fef54b08f4d719fac8abc2e824eeae389"

@hydra.main(config_path=os.path.dirname(__file__), config_name="config", version_base=None)
def go(config: DictConfig):
    print(OmegaConf.to_yaml(config)) 

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()
    steps_to_execute = config["main"]["execute_steps"]
    if isinstance(steps_to_execute, str):
        steps_to_execute = steps_to_execute.split(",")
    else:
        steps_to_execute = list(steps_to_execute)

    if "download" in steps_to_execute:
        env_vars = copy.deepcopy(os.environ)
        env_vars["WANDB_PROJECT"] = config["main"]["project_name"]
        env_vars["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"]
            },
            env_manager="local"
        )

    if "basic_cleaning" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": config["basic_cleaning"]["input_artifact"],
                "output_artifact": config["basic_cleaning"]["output_artifact"],
                "output_type": config["basic_cleaning"]["output_type"],
                "output_description": config["basic_cleaning"]["output_description"],
            },
            env_manager="local",
        )

    if "preprocess" in steps_to_execute:
        pass

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": f"{config['main']['project_name']}/{config['basic_cleaning']['output_artifact']}:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            },
            env_manager="local"
        )

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": config["segregate"]["input_artifact"],
                "artifact_root": config["segregate"]["artifact_root"],
                "artifact_type": config["segregate"]["artifact_type"],
                "test_size": config["segregate"]["test_size"],
                "random_state": config["segregate"]["random_state"],
                "stratify": config["segregate"]["stratify"],
            },
            env_manager="local",
        )

    if "random_forest" in steps_to_execute:
        model_config = os.path.abspath("random_forest_config.yml")
        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": config["random_forest"]["train_data"],
                "model_config": model_config,
                "export_artifact": config["random_forest"]["export_artifact"],
                "random_seed": config["random_forest"]["random_seed"],
                "val_size": config["random_forest"]["val_size"],
                "stratify": config["random_forest"]["stratify"],
            },
            env_manager="local",
        )

    if "evaluate" in steps_to_execute:
        pass

    if "test_regression_model" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "test_regression_model"),
            "main",
            parameters={
                "mlflow_model": "model_export:prod",
                "test_artifact": "trainval_data_test.csv:latest"
            },
            env_manager="local",
        )

if __name__ == "__main__":
    go()