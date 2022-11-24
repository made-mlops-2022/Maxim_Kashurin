import click
import logging
import sys
import json
from loguru import logger
logger.add("train_pipeline_info.log")

from ml_project.classes.Preprocess import PreprocessingModel
from ml_project.classes.Transform import TransformModel
from ml_project.classes.Train import TrainModel
from ml_project.classes.Exporting import ExportModel

from ml_project.enities.cfg import *




def run_train_pipeline(mcfg,cfg,name):
    params_from_yaml=read_training_pipeline_params(mcfg,cfg,name)
    preproc =  PreprocessingModel(params_from_yaml)
    tf =  TransformModel(params_from_yaml)
    train =  TrainModel(params_from_yaml)
    export =  ExportModel(params_from_yaml)





@click.command(name="train_pipeline")
@click.argument("model_path")
@click.argument("config_path")
@click.argument("name_model_from_yaml")
def train_pipeline_command(model_path: str,config_path:str,name_model_from_yaml:str):
    logger.info(f"Запустили код с параметрами: model_path :{model_path} cfg_path:{config_path} name_of_model {name_model_from_yaml} ")
    run_train_pipeline(model_path,config_path,name_model_from_yaml)


if __name__ == "__main__":
    train_pipeline_command()