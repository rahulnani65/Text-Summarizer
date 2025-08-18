from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(">>>>>> stage Model Training started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(">>>>>> stage Model Training completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e