from fastapi import FastAPI
from fastapi.testclient import TestClient
import json
from inference import app
client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg":"Введите /health или /predict"}


def test_predict_wrong_order():
    response = client.post(
        "/predict/",
        json={
            "data": [59.0, 1.0, 0.0, 178.0, 270.0, 0.0, 2.0, 145.0, 0.0, 4.2, 2.0, 0.0, 2.0],
            "features": ['sex', 'cp', 'trestbps', 'chol', 'age', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        },
    )

    assert response.status_code == 400


def test_predict_wrong_feature_number():
    response = client.post(
        "/predict/",
        json={
            "data": [59.0, 1.0, 0.0, 178.0, 270.0, 0.0, 2.0, 145.0, 0.0, 4.2, 2.0, 0.0, 2.0],
            "features": ['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        },
    )
    assert response.status_code == 400


def test_predict_wrong_data_len():
    response = client.post(
        "/predict/",
        json={
            "data": [1.0, 0.0, 178.0, 270.0, 0.0, 2.0, 145.0, 0.0, 4.2, 2.0, 0.0, 2.0],
            "features": ['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        },
    )
    assert response.status_code == 400


def test_predict_wrong_features():
    response = client.post(
        "/predict/",
        json={
            "data": [59.0, 1.0, 0.0, 178.0, 270.0, 0.0, 2.0, 145.0, 0.0, 4.2, 2.0, 0.0, 2.0],
            "features": ['age', 'se234x', 'cp352134', 'trestbps2', 'c23hol', 'fbssdafasd', 'asad3252restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        },
    )
    assert response.status_code == 400