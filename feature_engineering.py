import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots


def data_preprocessing(data: pd.DataFrame) -> tuple:
    """
    :param data: dataframe with data
    :return: 4 dataframes X_train, X_test, y_train, y_test
    """
    data["GENDER"] = data["GENDER"].replace({"M": 0, "F": 1})
    data["LUNG_CANCER"] = data["LUNG_CANCER"].replace({"NO": 0, "YES": 1})
    X = data.drop(columns=["LUNG_CANCER"])
    y = data["LUNG_CANCER"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
