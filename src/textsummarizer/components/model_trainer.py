from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textsummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cpu"  # Using CPU to avoid memory issues
        print(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Set special tokens
        tokenizer.pad_token = tokenizer.eos_token

        # Loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        print("Dataset loaded successfully")
        print(f"Train dataset size: {len(dataset_samsum_pt['train'])}")
        print(f"Validation dataset size: {len(dataset_samsum_pt['validation'])}")

        # Function to preprocess data
        def preprocess_function(examples):
            # Tokenize inputs
            inputs = examples["dialogue"]
            targets = examples["summary"]
            
            model_inputs = tokenizer(
                inputs,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Tokenize targets
            labels = tokenizer(
                targets,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Preprocess the datasets
        train_dataset = dataset_samsum_pt["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_samsum_pt["train"].column_names,
            desc="Running tokenizer on train dataset",
        )

        eval_dataset = dataset_samsum_pt["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_samsum_pt["validation"].column_names,
            desc="Running tokenizer on validation dataset",
        )

        print("Datasets preprocessed and ready for training!")
        
        # Training arguments with correct parameter names
        training_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,  # Using the correct config parameter
            learning_rate=5e-5,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,  # Changed to match the config
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_total_limit=self.config.save_total_limit,  # Using the config value
            load_best_model_at_end=self.config.load_best_model_at_end,  # Using the config value
            gradient_checkpointing=True  # Memory optimization
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus,
            padding=True
        )

        # Initialize trainer
        trainer = Trainer(
            model=model_pegasus,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        print("Starting training...")
        try:
            trainer.train()
            print("Training completed successfully!")

            print("Saving model...")
            # Save model
            model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
            print("Model and tokenizer saved successfully!")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e