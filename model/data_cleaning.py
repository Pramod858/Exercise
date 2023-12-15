import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, Fills missing values with median average values, and converts the data type to int.
        """
        try:
            data.drop(
                [
                    "street",
                    "date",
                    "country",
                    "yr_renovated",
                ],
                axis=1, inplace=True
            )
            data[['floors','bathrooms','bedrooms']] = data[['floors','bathrooms','bedrooms']].astype("int")

            data['price'].replace(0,np.nan,inplace=True)
            #Once the values are replaced to nan let's fill them with mean
            data['price'].fillna(value=data["price"].mean(), inplace=True)

            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataCatToNumeric(DataStrategy):
    """
    Data CatoNumeric strategy which convert categorical to numeric.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert categorical fetures to numeric in the data.
        """
        try:
            data['city'], _ = pd.factorize(data['city'])
            data['statezip'], _ = pd.factorize(data['statezip']) # function returns factorized df with list of classes
            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("price", axis=1)
            y = data["price"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataOutlierHandlingStrategy(DataStrategy):
    """
    Data outlier handling strategy which replaces outliers with NaN and fills NaN with means.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers with NaN and fill NaN with means.
        """
        try:
            feature_with_outlier = ["price", "sqft_lot", "sqft_basement"]
            for feature in feature_with_outlier:
                self.replace_outliers_with_nan_iqr(data, feature)

            # Get means for features with outliers
            feature_means = data[feature_with_outlier].mean()

            # Replace NaNs with means
            data.fillna(feature_means, inplace=True)

            return data
        except Exception as e:
            logging.error(e)
            raise e

    @staticmethod
    def replace_outliers_with_nan_iqr(df, feature, inplace=True):
        desired_feature = df[feature]

        q1, q3 = desired_feature.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        indices = (
            desired_feature[
                (desired_feature > upper_bound) | (desired_feature < lower_bound)
            ]
        ).index

        if not inplace:
            return desired_feature.replace(desired_feature[indices].values, np.nan)
        return desired_feature.replace(desired_feature[indices].values, np.nan, inplace=True)



class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
