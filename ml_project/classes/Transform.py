import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml_project.classes.Preprocess import PreprocessingModel
from ml_project.enities.cfg import FeatureParams
from loguru import logger
logger.add("logs/Transform_info.log")
class TransformModel(PreprocessingModel):
    """Нормализация данных"""

    @staticmethod
    def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
        num_pipeline = TransformModel.build_numerical_pipeline()
        return pd.DataFrame(num_pipeline.fit_transform(numerical_df))

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        num_pipeline = Pipeline(
            [("scale", StandardScaler()), ]
        )
        return num_pipeline

    @staticmethod
    def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
        return transformer.transform(df)

    @staticmethod
    def build_transformer(params: FeatureParams) -> ColumnTransformer:
        transformer = ColumnTransformer(
            [

                (
                    "numerical_pipeline",
                    TransformModel.build_numerical_pipeline(),
                    params.numerical_features,
                ),
            ]
        )
        return transformer


    def __init__(self, cfg):
        setattr(TransformModel, 'transformer', self.build_transformer(cfg.feature_params))
        logger.info(f'Данные до преобразования {self.train_df.head(5)}')
        TransformModel.transformer.fit(self.train_df)
        setattr(TransformModel, 'train_features', self.make_features(self.transformer, self.train_df))
        logger.info(f'Данные после преобразования {self.train_features}')


