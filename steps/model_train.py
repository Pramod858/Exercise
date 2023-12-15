import logging

import mlflow
import pandas as pd
from model.model_dev import RandomForestModel, KNNModel, XGBoostModel, SVMModel, LinearRegressionModel, HyperparameterTuner
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig

client = Client()

# Check if there's an active stack
if client.active_stack:
    # Get the experiment tracker from the active stack
    experiment_tracker = client.active_stack.experiment_tracker
else:
    # If there's no active stack, you might want to handle this case appropriately
    # For example, you can log a warning and set experiment_tracker to None or some default value
    logging.warning("No active stack found.")
    experiment_tracker = None


@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def train_model(x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "knn":
            mlflow.sklearn.autolog()
            model = KNNModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "svm":
            mlflow.sklearn.autolog()
            model = SVMModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e
