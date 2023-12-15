from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client

import mlflow

client = Client()

experiment_tracker = client.active_stack.experiment_tracker

if __name__ == "__main__":
    train_pipeline()
    print(
        "Now run \n "
        f"""    mlflow ui --backend-store-uri "{experiment_tracker.get_tracking_uri()}"\n"""
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    
    print(experiment_tracker)
    experiment = mlflow.get_experiment("0")
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")


