from ml_project.train_pipeline import *
import os
import pytest
import pandas as pd
from typing import List, Tuple
from catboost import CatBoostClassifier
@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def target_col():
    return "SalePrice"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "MSZoning",
        "Neighborhood",
        "RoofStyle",
        "MasVnrType",
        "BsmtQual",
        "BsmtExposure",
        "HeatingQC",
        "CentralAir",
        "KitchenQual",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "PavedDrive",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "OverallQual",
        "MSSubClass",
        "OverallCond",
        "GrLivArea",
        "GarageCars",
        "1stFlrSF",
        "Fireplaces",
        "BsmtFullBath",
        "YrSold",
        "YearRemodAdd",
        "LotFrontage",
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["YrSold"]

def test_load_dataset():
    model_path='../configs/models.yaml'
    dataset_path='../configs/CatBoostConfig.yaml'
    name='CatBoostClassifier'
    params=read_training_pipeline_params(model_path,dataset_path,name)
    params.input_data_path='../data/heart_cleveland_upload.csv'
    p=PreprocessingModel(params)
    data = p.read_data('../data/heart_cleveland_upload.csv')
    assert len(data) > 10

    assert params.feature_params.target_col in data.keys()


def test_split_dataset(tmpdir, dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    model_path = '../configs/models.yaml'
    dataset_path = '../configs/CatBoostConfig.yaml'
    name = 'CatBoostClassifier'
    params = read_training_pipeline_params(model_path, dataset_path, name)
    params.input_data_path = '../data/heart_cleveland_upload.csv'
    p = PreprocessingModel(params)
    data = p.read_data('../data/heart_cleveland_upload.csv')
    train, val = p.split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10

@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(

        numerical_features=numerical_features,
        target_col="condition",
    )
    model_path = '../configs/models.yaml'
    dataset_path='../configs/CatBoostConfig.yaml'
    name=CatBoostClassifier
    params=read_training_pipeline_params(model_path,dataset_path,name)
    params.input_data_path='../data/heart_cleveland_upload.csv'
    p=PreprocessingModel(params)
    data = p.read_data('../data/heart_cleveland_upload.csv')
    tf = TransformModel(params)
    transformer = tf.build_transformer(params)
    transformer.fit(data)
    features = tf.make_features(transformer, data)
    target = tf.extract_target(data, params)
    return features, target





