import json

import requests

input_data = {
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

response = requests.post("https://mlops-s8p2.onrender.com/run_inference/", data=json.dumps((input_data)))

print("Status code: ", response.status_code)
print("Salary category: ", response.json())
