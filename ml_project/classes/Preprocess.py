import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, NoReturn
from ml_project.enities.cfg import SplittingParams
from ml_project.enities.cfg import FeatureParams
import sys
from loguru import logger
logger.add("logs/Prepoc_info.log")



class PreprocessingModel(object):
    """Препроцессинг """
    @staticmethod
    def read_data(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data

    @staticmethod
    def split_train_val_data(
            data: pd.DataFrame, params: SplittingParams
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data, val_data = train_test_split(
            data, test_size=params.val_size, random_state=params.random_state
        )
        return train_data, val_data

    @staticmethod
    def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
        target = df[params.target_col]
        return target



    def __init__(self, cfg):

        setattr(PreprocessingModel,'data',self.read_data(cfg.input_data_path))
        logger.info(f"Данные из датасета {self.read_data(cfg.input_data_path).head(5)}")
        PreprocessingModel.train_df, PreprocessingModel.val_df = self.split_train_val_data(
            PreprocessingModel.data, cfg.splitting_params
        )
        setattr(PreprocessingModel, 'val_target', self.extract_target(PreprocessingModel.val_df, cfg.feature_params))
        logger.info(f"целевая переменная { self.extract_target(PreprocessingModel.val_df, cfg.feature_params).unique()}")
        setattr(PreprocessingModel, 'train_target', self.extract_target(PreprocessingModel.train_df, cfg.feature_params))
        setattr(PreprocessingModel, 'train_df', PreprocessingModel.train_df.drop(cfg.feature_params.target_col, 1))
        logger.info(f"кол-во столбцов после дропа трейна :{len (PreprocessingModel.train_df.columns)}")
        setattr(PreprocessingModel, 'val_df', PreprocessingModel.val_df.drop(cfg.feature_params.target_col, 1))
        logger.info(f"кол-во столбцов после дропа теста :{len (PreprocessingModel.val_df.columns)}")


