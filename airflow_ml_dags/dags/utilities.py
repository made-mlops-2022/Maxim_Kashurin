import os
from datetime import timedelta

from docker.types import Mount

VAL_SIZE = 0.25
METRICS_DIR_NAME = "/data/metrics/{{ ds }}"
GENERATE_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
TRANSFORMER_DIR_NAME = "/data/transformer_model/{{ ds }}"
MODEL_DIR_NAME = "/data/models/{{ ds }}"
PREDICTIONS_DIR_NAME = "/data/predictions/{{ ds }}"

MOUNT_OBJ = [Mount(
    source="/home/maxim/Технопарк/MLOps/airflow_ml_dags/data",
    target="/data",
    type='bind'
    )]

default_args = {
    "owner": "maxkashh",
    "email": ["kashurin2001@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def wait_file(file_name: str) -> bool:
    return os.path.exists(file_name)
