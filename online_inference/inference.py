import json
import pickle
from http.client import HTTPException
from typing import List, Optional
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel,validator
import uvicorn
from loguru import logger
from sklearn.pipeline import Pipeline
import os
import pandas as pd

logger.add("logs/inference_info.log")


class Response(BaseModel):
    condition: bool


class HeartDecease(BaseModel):
    data: List[List[float]]
    features: List[str]

    @validator('features')
    def check_features(cls, features):
        correct_features = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        if len(features) != len(correct_features):
            raise HTTPException(status_code=400, detail="features len is not right")
        for i, feature in enumerate(correct_features):
            if features[i] != feature:
                raise HTTPException(
                    status_code=400, detail="features order is not right"
                )
        return features

    @validator('data')
    def data_check(cls, data):
        for row in data:
            if len(row) != 13:
                raise HTTPException(status_code=400, detail="data len is not 13")

            for i, value in enumerate(row):
                if not isinstance(value, float):
                    raise HTTPException(
                        status_code=400, detail="data is not in correct type"
                    )
        return data


model: Optional[Pipeline] = None
app = FastAPI()


@app.get("/")
def main():
    return {"msg":"Введите /health или /predict"}


def load_object(path: str):
    with open(path, 'rb') as model_pkl:
        model = pickle.load(model_pkl)
        return model


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL",'models/logregmodel.pkl')
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health", status_code=200)
def health() -> json:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return {"model": "ready"}


def make_predict(data, features,model):
    data = np.array(data)

    logger.info(data)

    predicts = model.predict(data)

    logger.info(f"predicts: {predicts}")

    return [Response(condition=bool(cond)) for id, cond in enumerate(predicts)]


@app.post("/predict/", response_model=List[Response])
def predict(request: HeartDecease):
    return make_predict(request.data, request.features, model)





if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=os.getenv("PORT", 8080))


