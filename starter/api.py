import pickle
from typing import Any

import numpy as np
import pandas as pd
from munch import DefaultMunch

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from starter.ml.model import load_model, inference


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


app = FastAPI()


@app.get("/")
async def greetings():
    return {"greetings": "Welcome to the 3rd MLOPS project !"}


@app.post("/run_inference/")
def run_inference(X_input: Person):
    # Input data (single sample)
    X_init = {
        "age": X_input.age,
        "workclass": X_input.workclass,
        "fnlgt": X_input.fnlgt,
        "education": X_input.education,
        "education_num": X_input.education_num,
        "marital_status": X_input.marital_status,
        "occupation": X_input.occupation,
        "relationship": X_input.relationship,
        "race": X_input.race,
        "sex": X_input.sex,
        "capital_gain": X_input.capital_gain,
        "capital_loss": X_input.capital_loss,
        "hours_per_week": X_input.hours_per_week,
        "native_country": X_input.native_country
    }

    X_data = pd.DataFrame.from_dict(DefaultMunch.fromDict(X_init), orient='index')
    X_data = X_data.transpose()

    # Extract numerical features
    numerical_features = ['age', 'fnlgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    X_numerical = np.array(X_data[numerical_features])

    # Extract categorical features
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                            'native_country']
    X_categorical = np.array(X_data[categorical_features])

    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)

    X_categorical_df = pd.DataFrame(X_categorical, columns=categorical_features)

    # Transform X_categorical using the encoder
    X_categorical_encoded = encoder.transform(X_categorical_df)

    X_data = np.concatenate((X_numerical, X_categorical_encoded), axis=1)

    y = inference(model=load_model("model"), X=X_data)

    return {"salary": str(y)}


if __name__ == "__main__":
    X = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 120000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United States"
    }

    # response = requests.post("http://localhost:8000/run_inference/", json.dumps(X))

    # print(response.json())

    # print(response.status_code)

    run_inference(X_input=DefaultMunch.fromDict(X))
