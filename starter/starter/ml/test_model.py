import os

import pytest
import pandas as pd

from .data import process_data
from .model import train_model
from .. import DATA_DIR


@pytest.fixture
def df():
    return pd.read_csv(os.path.join(os.path.abspath(DATA_DIR), "census.csv"))


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def X(df, cat_features):
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return X


@pytest.fixture
def y(df, cat_features):
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return y


def test_X(X):
    assert X.shape == (32561, 108)


def test_y(y):
    assert y.shape == (32561,)


def test_train_model(df, X, y):
    assert df.shape == (32561, 15)
    model = train_model(X_train=X, y_train=y)
    assert len(model.classes_) == 2
