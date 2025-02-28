from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas import DataFrame


def train_model(X_train: DataFrame, y_train: DataFrame) -> RandomForestClassifier:
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    return clf
