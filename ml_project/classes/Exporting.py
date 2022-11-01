
import pickle
import json
from loguru import logger
logger.add("logs/Export_info.log")
from ml_project.classes.Train import TrainModel


class ExportModel(TrainModel):
    """Экспорт модели и метрик"""
    @staticmethod
    def serialize_model(model: object, output: str) -> str:
        with open(output, "wb") as f:
            pickle.dump(model, f)
        return output
    @staticmethod
    def take_results(path_to_model, metrics):
        return path_to_model, metrics


    def __init__(self,cfg):
        with open(cfg.metric_path, "w") as metric_file:
            json.dump(self.metrics, metric_file)

            logger.info(f"Записали метрики в файл  ")


        self.path_to_model = self.serialize_model(
            self.inference_pipeline, cfg.output_model_path
        )
        logger.info(f"Сериализовали модельку в банку ")
        self.take_results(self.path_to_model, self.metrics)
        logger.info(f"Путь до модельки :{self.path_to_model}")


