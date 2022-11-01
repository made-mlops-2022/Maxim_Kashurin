


from typing import Dict, Union
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from ml_project.classes.Transform import TransformModel
from ml_project.enities.cfg import TrainingPipelineParamsOfDecisionTreeClassifier
from ml_project.enities.cfg import TrainingParamsOfCatBoostClassifier

from loguru import logger
logger.add("logs/Train_info.log")

class TrainModel(TransformModel):

    """Обучение модели"""
    SklearnClassificationModel = Union[DecisionTreeClassifier, CatBoostClassifier]
    @staticmethod
    def train_model(
            features: pd.DataFrame, target: pd.Series, train_params: Union[TrainingPipelineParamsOfDecisionTreeClassifier,TrainingParamsOfCatBoostClassifier]
    ) -> SklearnClassificationModel:
        if train_params.model_type == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(
                random_state=train_params.random_state
            )
        elif train_params.model_type == "CatBoostClassifier":
            model = CatBoostClassifier(depth=train_params.depth, iterations=train_params.iterations,
                                       learning_rate=train_params.learning_rate)
        else:
            raise NotImplementedError()
        model.fit(features, target)
        return model
    @staticmethod
    def create_inference_pipeline(
            model: SklearnClassificationModel, transformer: ColumnTransformer
    ) -> Pipeline:
        return Pipeline([("feature_part", transformer), ("model_part", model)])

    @staticmethod
    def predict_model(
            model: Pipeline, features: pd.DataFrame, use_log_trick: bool = True
    ) -> np.ndarray:
        predicts = model.predict(features)

        return predicts
    @staticmethod
    def evaluate_model(
            predicts: np.ndarray, target: pd.Series
    ) -> Dict[str, float]:

        return {
            "Accuracy": accuracy_score(target, predicts),
            "Precision": precision_score(target, predicts),
            "Recall": recall_score(target, predicts),
            "F1 Score": f1_score(target, predicts),

        }


    def __init__(self,cfg):
        setattr(TrainModel, 'model', self.train_model(self.train_features, self.train_target, cfg.train_params))
        logger.info(f'Обучили модель')
        setattr(TrainModel, 'inference_pipeline', self.create_inference_pipeline(self.model, self.transformer))
        logger.info(f'Сделали пайплайн')
        setattr(TrainModel, 'predicts',self.predict_model(TrainModel.inference_pipeline,self.val_df,))
        logger.info(f'Реальные метки :{self.val_target.values}')
        logger.info(f'Предсказания :{self.predicts}')
        setattr(TrainModel, 'metrics', self.evaluate_model(TrainModel.predicts,self.val_target,))
        logger.info(f'Метрики :{self.metrics}')


