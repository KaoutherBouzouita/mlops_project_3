import pickle
from typing import Any

import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import inference, load_model


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


app = FastAPI()


@app.get("/")
async def greetings():
    return {"greetings": "Welcome to the 3rd MLOPS project !"}


@app.post("/run_inference/{model}")
async def run_inference(model: Any, X: Person):
    model = load_model(model_name=f"{model}.sav")
    return inference(model, X)


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    # Input data (single sample)
    X = {
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "India"
    }

    X = pd.DataFrame([X])

    # Extract numerical features
    numerical_features = ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    X_numerical = np.array(X[numerical_features])

    # Extract categorical features
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    X_categorical = np.array(X[categorical_features])

    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    X_categorical_df = pd.DataFrame(X_categorical, columns=categorical_features)

    # Transform X_categorical using the encoder
    X_categorical_encoded = encoder.transform(X_categorical_df)

    X = np.concatenate((X_numerical, X_categorical_encoded), axis=1)

    y = inference(model=load_model("model"), X=X)
    print("Salary belongs to category: ", y)
