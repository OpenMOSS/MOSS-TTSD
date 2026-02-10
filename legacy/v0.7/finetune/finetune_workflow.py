import argparse
import os

import finetune
import torch
import yaml
from data_preprocess import process_data

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"  # Download the model from Hugging Face if not specified in the config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Asteroid TTS Instruct Model")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./finetune/finetune_config.yaml",
        help="Path to the finetune workflow configuration file",
    )
    parser.add_argument(
        "-pd",
        "--pass_data_preprocess",
        action="store_true",
        default=False,
        help="Skip data preprocess step and proceed directly to fine-tuning",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(f"Configuration file '{args.config}' does not exist.")
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    if not args.pass_data_preprocess:
        if not config.get("path_to_jsonl"):
            raise ValueError("JSONL file path is required in the configuration.")
        elif not os.path.exists(config["path_to_jsonl"]):
            raise ValueError(f"JSONL file '{config['path_to_jsonl']}' does not exist.")
        if not config.get("path_to_model"):
            config["path_to_model"] = DEFAULT_MODEL_PATH
        elif config["path_to_model"] != DEFAULT_MODEL_PATH and not os.path.exists(
            config["path_to_model"]
        ):
            raise ValueError(f"Model path '{config['path_to_model']}' does not exist.")
        if not config.get("data_output_directory"):
            raise ValueError("Data output directory is required in the configuration.")
        elif not os.path.exists(config["data_output_directory"]):
            os.makedirs(config["data_output_directory"])

        print("Beginning data processing...")
        process_data(
            jsonl=str(config["path_to_jsonl"]),
            model_path=str(config["path_to_model"]),
            output_dir=str(config["data_output_directory"]),
            data_name=str(config.get("data_name", "processed_data")),
            use_normalize=bool(config.get("use_normalize", True)),
        )
        print("Data processing completed.")
    else:
        print("Skipping data preprocess step.")
        # Validate model path for fine-tuning when skipping data preprocess
        if not config.get("path_to_model"):
            config["path_to_model"] = DEFAULT_MODEL_PATH
        elif config["path_to_model"] != DEFAULT_MODEL_PATH and not os.path.exists(
            config["path_to_model"]
        ):
            raise ValueError(f"Model path '{config['path_to_model']}' does not exist.")

    if not config.get("finetuned_model_output"):
        raise ValueError("Finetune output directory is required in the configuration.")
    elif not os.path.exists(config["finetuned_model_output"]):
        os.makedirs(config["finetuned_model_output"])

    training_config = {}
    training_config_file = config.get("training_config_file")
    if training_config_file and os.path.exists(training_config_file):
        with open(training_config_file, "r") as f:
            training_config = yaml.safe_load(f)
    else:
        print(
            "Training config file not found or not specified, using default training configuration."
        )

    # Load LoRA configuration if using LoRA
    lora_cfg = {}
    use_lora = bool(config.get("use_lora", False))
    if use_lora:
        lora_config_file = config.get("lora_config_file", "finetune/lora_config.yaml")
        if lora_config_file and os.path.exists(lora_config_file):
            with open(lora_config_file, "r") as f:
                lora_cfg = yaml.safe_load(f)
            print(f"Loaded LoRA configuration from {lora_config_file}")
        else:
            print(
                "LoRA config file not found or not specified, using default LoRA configuration."
            )

    print("Beginning finetuning...")
    finetune.train(
        model_path=str(config["path_to_model"]),
        data_dir=str(config["data_output_directory"]),
        output_dir=str(config["finetuned_model_output"]),
        training_config=training_config,
        device=str(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ),
        use_lora=use_lora,
        lora_cfg=lora_cfg,
    )
