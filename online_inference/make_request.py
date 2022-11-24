import numpy as np
import pandas as pd
import requests
from loguru import logger
logger.add("logs/make_request.log")
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    del data["condition"]

    request_features = list(data.columns)
    for i in range(10):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_features)
        print(request_data)
        logger.info({"data": [request_data], "features": request_features})
        response = requests.post(
            "http://0.0.0.0:8080/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())