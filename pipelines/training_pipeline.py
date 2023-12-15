from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False,settings={"docker": docker_settings})
def train_pipeline(): 
    """
    Args:
        ingest_data: function
        clean_data: function
        model_train: function
        evaluation: function
    Returns:
        mse: float
        rmse: float
    """
    
    loaded_data = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(loaded_data)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
    return mse, rmse

