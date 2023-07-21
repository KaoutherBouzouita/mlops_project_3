import os
import pickle

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import process_data
from .model import train_model, compute_model_metrics, inference


@pytest.fixture
def df():
    return pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', "census.csv")))


def df_train(df):
    train, test = train_test_split(df, test_size=0.20, random_state=23)
    return train


def df_test(df):
    train, test = train_test_split(df, test_size=0.20, random_state=23)
    return test


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
def X_train(df, cat_features):
    X, y, encoder, lb = process_data(df_train, categorical_features=cat_features, label="salary", training=True)
    return X


@pytest.fixture
def y_train(df, cat_features):
    X, y, encoder, lb = process_data(df_train, categorical_features=cat_features, label="salary", training=True)
    return y


@pytest.fixture
def encoder(df, cat_features):
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return encoder


@pytest.fixture
def lb(df, cat_features):
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return lb


@pytest.fixture
def X_test(df, cat_features, encoder, lb):
    X_test, y_test, encoder, lb = process_data(df_test, categorical_features=cat_features, label="salary", training=False,
                                               encoder=encoder, lb=lb)
    return X_test


@pytest.fixture
def y_test(df, cat_features, encoder, lb):
    X_test, y_test, encoder, lb = process_data(df_test, categorical_features=cat_features, label="salary", training=False,
                                               encoder=encoder, lb=lb)
    return y_test


@pytest.fixture
def model():
    return pickle.load(open(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..',
                                                          'model', 'model.sav')), 'rb'))


@pytest.fixture
def y_pred(model, X_test):
    return inference(model, X_test)


def test_X(X_train):
    assert X_train.shape == (32561, 108)


def test_y(y_train):
    assert y_train.shape == (32561,)


def test_train_model(df, X_train, y_train):
    assert df.shape == (32561, 15)
    model = train_model(X_train=X_train, y_train=y_train)
    assert len(model.classes_) == 2


