from logging import config
from src.textsummarizer.components.data_transformation import DataTransformation
from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()