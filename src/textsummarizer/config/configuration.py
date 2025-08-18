from logging import config
from textsummarizer.constants import *
from textsummarizer.utils.common import read_yaml, create_directories
from textsummarizer.entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelEvaluationConfig, ModelTrainerConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Use list instead of list[Path] for ensure compatibility
        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name = config.tokenizer_name
        )

        return data_transformation_config
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=1,  # Fixed value that worked in notebook
            warmup_steps=500,    # Fixed value that worked in notebook
            per_device_train_batch_size=2,  # Reduced batch size
            per_device_eval_batch_size=2,   # Matching eval batch size
            weight_decay=0.01,    # Fixed value that worked in notebook
            logging_steps=10,     # More frequent logging
            eval_strategy="steps",# Correct parameter name
            eval_steps=50,        # More frequent evaluation
            save_steps=50,        # More frequent saving
            gradient_accumulation_steps=4,  # Optimized value
            save_total_limit=3,   # Keep only last 3 checkpoints
            load_best_model_at_end=True    # Load best model after training
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name

        )

        return model_evaluation_config
    
    