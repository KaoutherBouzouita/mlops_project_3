# Script to train machine learning model.
import os.path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.data import process_data
import pickle
from sklearn.tree import DecisionTreeClassifier

from . import DATA_DIR
from starter.starter.ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv(os.path.join(os.path.abspath(DATA_DIR), "census.csv"))

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

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

