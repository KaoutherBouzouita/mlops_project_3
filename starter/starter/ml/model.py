import os
import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def save_model(model):
    """Save trained model"""
    filename = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../model/model.sav"))
    pickle.dump(model, open(filename, 'wb'))


def load_model(model_name):
    return pickle.load(open(os.path.realpath(os.path.join(os.path.dirname(__file__), '../..',
                                                          'model', f'{model_name}.sav')), 'rb'))


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    print("X:", X)
    print("X shape:", X.shape)
    preds = model.predict(X)
    print("Preds: ", preds)
    return preds
