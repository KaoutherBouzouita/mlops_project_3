# Script to train machine learning model.
import os.path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.data import process_data
import pickle
from sklearn.tree import DecisionTreeClassifier

from ml.model import train_model, inference, compute_model_metrics



def slice_averages(cat_features, X_test, y_test, df, model):
    """ Function for calculating descriptive stats on slices of the Adult dataset."""
    for cat in cat_features:  # cat in ['workclass', 'education', 'marital-status', 'occupation',...]
        for value in df[cat].unique():
            # Features values for the selected value
            y_test_cat = y_test[df[df[df[cat] == value].index]["salary"]]
            X_test_cat = X_test[df[df[df[cat] == value].index].drop["salary"]]
            # Run inference
            y_pred_cat = inference(model, X_test_cat)
            # Model metrics for each category
            precision, recall, fbeta = compute_model_metrics(y_test_cat, y_pred_cat)

            print(f"Feature: {cat} - Value {value}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Fbeta: {fbeta:.4f}")
        print()


# Add code to load in the data.
data = pd.read_csv(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', "census.csv")))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train a model.
model = train_model(X_train, y_train)

# Save the trained model
filename = "../model/model.sav"
pickle.dump(model, open(filename, 'wb'))

# Run inference
y_pred = inference(model, X_test)

print(f"Y test: {y_test}")
print(f"Y pred: {y_pred}")

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
print("Fbeta: ", fbeta)

slice_averages(cat_features, X_test, y_test, data, model)
