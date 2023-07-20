import os
import pickle

import pytest
import pandas as pd

from .data import process_data
from .model import train_model, compute_model_metrics, inference


@pytest.fixture
def df():
    return pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', "census.csv")))


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
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return X


@pytest.fixture
def y_train(df, cat_features):
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
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
    X_test, y_test, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=False,
                                               encoder=encoder, lb=lb)
    return X_test


@pytest.fixture
def y_test(df, cat_features, encoder, lb):
    X_test, y_test, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=False,
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


def test_slice_averages(cat_features, X_test, y_test, df, model):
    """ Function for calculating descriptive stats on slices of the Adult dataset."""
    for cat in cat_features:  # cat in ['workclass', 'education', 'marital-status', 'occupation',...]
        for value in df[cat].unique():
            # Features values for the selected value
            y_test_cat = y_test[df[df[cat] == value].index]
            X_test_cat = X_test[df[df[cat] == value].index]
            # Run inference
            y_pred_cat = inference(model, X_test_cat)
            # Model metrics for each category
            precision, recall, fbeta = compute_model_metrics(y_test_cat, y_pred_cat)

            print(f"Feature: {cat} - Value {value}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Fbeta: {fbeta:.4f}")
        print()
