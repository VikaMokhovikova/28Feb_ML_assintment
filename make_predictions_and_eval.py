from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from pandas import Series


def make_prediction(clf: RandomForestClassifier, X_test) -> Series:
    """
    A function to make predictions for random forest classifier


    :param clf: A pretrained random forest classifier
    :return: predictions result of this classifier
    """
    return clf.predict(X_test)


def eval_prediction(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred)
