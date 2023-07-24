from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_greetings():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greetings": "Welcome to the 3rd MLOPS project !"}


def test_run_inference_low_salary():
    response = client.post("/run_inference/", json={
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
    })

    assert response.status_code == 200
    assert response.json() == {"salary": "[0]"}


def test_run_inference_high_salary():
    response = client.post("/run_inference/", json={
        "age": 37,
        "workclass": "emp-inc",
        "fnlgt": 257295,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 75,
        "native_country": "Thailand"
    })

    assert response.status_code == 200
    assert response.json() == {"salary": "[1]"}
