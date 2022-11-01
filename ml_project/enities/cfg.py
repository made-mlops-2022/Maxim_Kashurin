from dataclasses import dataclass, field
from typing import List, Optional
from marshmallow_dataclass import class_schema
import yaml
import os

@dataclass()
class FeatureParams:
    """Feature params"""
    numerical_features: List[str]
    target_col: Optional[str]


@dataclass()
class SplittingParams:
    """Spliting params"""
    val_size: float = field(default=0.2)
    random_state: int = field(default=13)


@dataclass()
class TrainingParamsOfCatBoostClassifier:
    """Train params"""
    model_type: str = field(default="CatBoostClassifier")
    random_state: int = field(default=255)
    depth: int = field(default=255)
    iterations: int = field(default=255)
    learning_rate: float = field(default=255)


@dataclass()
class TrainingParamsOfDecisionTreeClassifier:
    """Train params"""
    model_type: str = field(default="DecisionTreeClassifier")
    max_depth: int = field(default=255)
    random_state: int = field(default=255)


@dataclass()
class TrainingPipelineParamsOfDecisionTreeClassifier:
    """Params of yaml file"""
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParamsOfDecisionTreeClassifier
    # downloading_params: Optional[DownloadParams] = None
    # use_mlflow: bool = False
    # mlflow_uri: str = "http://18.156.5.226/"
    # mlflow_experiment: str = "inference_demo"


@dataclass()
class TrainingPipelineParamsOfCatBoostClassifier:
    """Params of yaml file"""
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParamsOfCatBoostClassifier
    # downloading_params: Optional[DownloadParams] = None
    # use_mlflow: bool = False
    # mlflow_uri: str = "http://18.156.5.226/"
    # mlflow_experiment: str = "inference_demo"


@dataclass()
class TrainingPipeUnion:
    """Names of models"""
    modelDTC: str
    modelCBC: str


TrainingPipeUnionSchema = class_schema(TrainingPipeUnion)

from loguru import logger
logger.add("read_info.log")
def read_training_pipeline_params(path_of_model: str, path: str, name: str) -> TrainingPipelineParamsOfDecisionTreeClassifier:
    """Creating schema of yaml file of model"""

    res = None
    with open(path_of_model, "r") as input_stream:

        schema = TrainingPipeUnionSchema()
        res = schema.load(yaml.safe_load(input_stream))
    if name == res.modelDTC:
        with open(path, "r") as input_stream2:
            TrainingPipelineParamsOfDecisionTreeClassifierSchema = class_schema(
                TrainingPipelineParamsOfDecisionTreeClassifier)
            schemaDTC = TrainingPipelineParamsOfDecisionTreeClassifierSchema()
            return schemaDTC.load(yaml.safe_load(input_stream2))
    else:
        with open(path, "r") as input_stream3:
            TrainingPipelineParamsOfCatBoostClassifierSchema = class_schema(TrainingPipelineParamsOfCatBoostClassifier)
            schemaDTC = TrainingPipelineParamsOfCatBoostClassifierSchema()
            schemaCBC = TrainingPipelineParamsOfCatBoostClassifierSchema()
            return schemaCBC.load(yaml.safe_load(input_stream3))
    raise Exception
