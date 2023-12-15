import logging
import mlflow
import pandas as pd
from model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from typing import Tuple
# from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

# Initialize the Client
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

# @enable_mlflow
@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def evaluation(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:

        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e
