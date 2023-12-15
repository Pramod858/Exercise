import logging
import pandas as pd
from typing import Tuple
from zenml import step
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy, DataOutlierHandlingStrategy, DataCatToNumeric
from typing_extensions import Annotated

@step
def clean_data(data: pd.DataFrame,) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        if data is None:
            raise ValueError("Input data is None. Check the previous step.")
        
        outlier_stratergy = DataOutlierHandlingStrategy()
        data_cleaner = DataCleaning(data, outlier_stratergy)
        data_cleaned = data_cleaner.handle_data()

        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data_cleaned, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        cat_to_numeric_strategy = DataCatToNumeric()
        data_cleaner = DataCleaning(preprocessed_data, cat_to_numeric_strategy)
        data_numeric = data_cleaner.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(data_numeric, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
