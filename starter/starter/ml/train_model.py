# Script to train machine learning model.
import os.path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pickle
from sklearn.tree import DecisionTreeClassifier

import starter.starter.ml.data as data
import starter.starter.ml.model as model


def slice_averages(cat_features, df_train, df_test):
    """ Function for calculating descriptive stats on slices of the Adult dataset."""

    if os.path.exists("slice_output.txt"):
        os.remove("slice_output.txt")

    for cat in cat_features:  # cat in ['workclass', 'education', 'marital-status', 'occupation',...]
        for value in df_test[cat].unique():
            # X and y datasets for category cat
            y_train_cat = df_train[df_train[cat] == value][["salary"]]
            X_train_cat = df_train[df_train[cat] == value].drop(["salary"], axis=1)
            y_test_cat = df_test[df_test[cat] == value][["salary"]]
            X_test_cat = df_test[df_test[cat] == value].drop(["salary"], axis=1)

            # Construct DFs for category cat
            df_train_cat = X_train_cat
            df_train_cat["salary"] = y_train_cat
            df_test_cat = X_test_cat
            df_test_cat["salary"] = y_test_cat

            # Proces the test data with the process_data function.
            X_train_cat_preproc, y_train_cat_preproc, encoder, lb = data.process_data(
                df_train_cat, categorical_features=cat_features, label="salary", training=True
            )

            # Proces the test data with the process_data function.
            X_test_cat_preproc, y_test_cat_preproc, encoder, lb = data.process_data(
                df_test_cat, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )

            # Train model
            model_cat = model.train_model(X_train_cat_preproc, y_train_cat_preproc)

            # Run inference
            y_pred_cat = model.inference(model_cat, X_test_cat_preproc)
            # Model metrics for each category
            precision, recall, fbeta = model.compute_model_metrics(y_test_cat_preproc, y_pred_cat)

            with open("slice_output.txt", "a") as file:
                file.write("______________________________________" + "\n")
                file.write(f"Feature: {cat} - Value {value}" + "\n")
                file.write("______________________________________" + "\n")
                file.write(f"Precision: {precision:.4f}" + "\n")
                file.write(f"Recall: {recall:.4f}" + "\n")
                file.write(f"Fbeta: {fbeta:.4f}" + "\n")
                file.write("______________________________________" + "\n")


def load_data():
    """Add code to load in the data."""
    return pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__), '../..', 'data', "census.csv")))


def split_data(data: pd.DataFrame, random_state: bool):
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    if random_state:
        return train_test_split(data, test_size=0.20, random_state=23)
    return train_test_split(data, test_size=0.20)


def go():
    input_data = load_data()
    train, test = split_data(input_data, True)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Process the train data with the process_data function.
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    with open('encoder.pkl', 'wb') as file:
        pickle.dump(encoder, file)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = data.process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Train a model.
    trained_model = model.train_model(X_train, y_train)

    # Save the trained model
    model.save_model(trained_model)

    # Run inference
    y_pred = model.inference(trained_model, X_test)

    print(f"Y test: {y_test}")
    print(f"Y pred: {y_pred}")

    # Compute model metrics
    precision, recall, fbeta = model.compute_model_metrics(y_test, y_pred)

    print("Overall Precision: ", precision)
    print("Overall Recall: ", recall)
    print("Overall Fbeta: ", fbeta)

    slice_averages(cat_features, train, test)


go()
