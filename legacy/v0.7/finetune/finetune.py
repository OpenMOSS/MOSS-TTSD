import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from modeling_asteroid import AsteroidTTSInstruct

# Import peft related modules
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
MAX_CHANNELS = 8


class LazySupervisedDataset(Dataset):
    def __init__(self, data_dir, channels: int, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer, self.channels = tokenizer, channels
        pkls = [
            os.path.join(data_dir, each)
            for each in os.listdir(data_dir)
            if each.endswith(".pkl")
        ]
        self.data = []
        for pkl_file in pkls:
            # Load metas file containing three arrays: [pointers, tokens_lengths, tims_lengths]
            metas = np.load(pkl_file.replace(".pkl", "_metas.npy"))
            pointers = metas[0]  # Extract byte offset position array

            f = open(pkl_file, "rb")
            for start_pointer in pointers:
                f.seek(int(start_pointer))  # Ensure integer type
                self.data.append(pickle.load(f))
            f.close()
        random.shuffle(self.data)
        print(f"Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def truncate_and_shift(self, example: Dict[str, List]) -> Dict[str, np.ndarray]:
        # Read input_ids and labels from data instead of copying input_ids
        input_ids = np.array(example["input_ids"])[:, : self.channels]
        labels = np.array(example["labels"])[:, : self.channels]  # Use labels from data

        seq_len = input_ids.shape[0]
        new_seq_len = seq_len + self.channels - 1

        shifted_input_ids = np.full((new_seq_len, self.channels), 1024)
        shifted_input_ids[:, 0] = np.full(new_seq_len, self.tokenizer.pad_token_id)
        shifted_labels = np.full((new_seq_len, self.channels), -100)

        # Delay Pattern: Shift input_ids and labels
        for i in range(self.channels):
            shifted_input_ids[i : (seq_len + i), i] = input_ids[:, i]
            shifted_labels[i : (seq_len + i), i] = labels[:, i]

        return {
            "input_ids": shifted_input_ids,
            "labels": shifted_labels,
            "attention_mask": np.ones(new_seq_len),
        }

    def __getitem__(self, i) -> Dict[str, np.ndarray]:
        line = self.data[i]

        # Data validation
        if "input_ids" not in line or "labels" not in line:
            raise ValueError(
                f"Data format error: sample {i} missing 'input_ids' or 'labels' field"
            )

        return self.truncate_and_shift(
            line
        )  # Return numpy arrays for consistency with original code


@dataclass
class DataCollatorForSupervisedDataset:
    pad_token_id: int
    max_length: int
    filler_token_id: int = 1024

    def __call__(
        self, instances: List[Dict[str, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        channels = input_ids[0].shape[1]
        max_length = min(max(ids.shape[0] for ids in input_ids), self.max_length)
        padded_input_ids, padded_labels, padded_attns = [], [], []

        for ids, lbls, attn in zip(input_ids, labels, attention_masks):
            seq_len = ids.shape[0]
            if seq_len < max_length:
                pad_len = max_length - seq_len
                input_pad = np.full((pad_len, channels), self.filler_token_id)
                input_pad[:, 0] = self.pad_token_id
                padded_input_ids.append(np.concatenate([ids, input_pad]))
                label_pad = np.full((pad_len, channels), -100)
                padded_labels.append(np.concatenate([lbls, label_pad]))
                attn_pad = np.zeros(pad_len)
                padded_attns.append(np.concatenate([attn, attn_pad]))
            else:
                padded_input_ids.append(ids[:max_length])
                padded_labels.append(lbls[:max_length])
                padded_attns.append(attn[:max_length])

        input_ids = torch.tensor(np.stack(padded_input_ids), dtype=torch.long)
        labels = torch.tensor(np.stack(padded_labels), dtype=torch.long)
        attention_mask = torch.tensor(np.stack(padded_attns), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def train(
    model_path: str,
    data_dir: str,
    output_dir: str,
    training_config: Dict,
    device: str = "cuda",
    use_lora: bool = False,
    lora_cfg: Dict = None,
):
    print("Initializing tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    # Load model with CPU offload support
    model = AsteroidTTSInstruct.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        offload_folder="offload",
        offload_state_dict=True,
    )

    model.set_weights([8, 2, 1, 1, 1, 1, 1, 1])
    model.config.use_cache = False

    # Move model to device
    model.to(torch.device(device))

    # Enable gradient checkpointing first (on base model)
    if training_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled with use_reentrant=False")

    # Configure LoRA parameters if using LoRA
    if use_lora:
        print("Configuring LoRA parameters...")

        # Default LoRA configuration
        default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "use_rslora": True,
        }

        # Merge with user provided configuration
        if lora_cfg:
            default_lora_config.update(lora_cfg)

        print(f"Using LoRA configuration: {default_lora_config}")

        lora_config = LoraConfig(
            r=int(default_lora_config["r"]),
            lora_alpha=int(default_lora_config["lora_alpha"]),
            target_modules=default_lora_config["target_modules"],
            lora_dropout=float(default_lora_config["lora_dropout"]),
            bias=default_lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
            use_rslora=bool(default_lora_config["use_rslora"]),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA configuration completed")

        # Re-enable gradient checkpointing on PEFT model (to ensure compatibility)
        if training_config.get("gradient_checkpointing", True):
            # Call base model's method with gradient_checkpointing_kwargs
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(
                "Re-enabled gradient checkpointing on LoRA base model with use_reentrant=False"
            )

        # Ensure model is in training mode and verify trainable parameters
        model.train()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise ValueError(
                "No trainable parameters! LoRA configuration might be problematic."
            )
        print(f"Number of trainable parameters: {trainable_params:,}")
    else:
        model.train()

    print("Initializing dataloader")
    train_dataset = LazySupervisedDataset(data_dir, MAX_CHANNELS, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, 16000)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(
            training_config.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training_config.get("gradient_accumulation_steps", 1)
        ),
        num_train_epochs=int(training_config.get("num_train_epochs", 50)),
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        bf16=bool(training_config.get("bf16", True)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 10)),
        save_total_limit=int(training_config.get("save_total_limit", 100)),
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 1)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(training_config.get("lr_scheduler_type", "cosine")),
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        gradient_checkpointing=False,  # Already enabled manually on model, don't duplicate
        # Add following parameters to resolve gradient issues
        remove_unused_columns=False,  # Keep all columns
        dataloader_pin_memory=False,  # May help avoid certain CUDA issues
        save_safetensors=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    trainer.train()
    torch.cuda.synchronize()

    # Save model
    if use_lora:
        # If using LoRA, merge LoRA weights to base model first, then save complete model
        print("Merging LoRA weights to base model...")
        merged_model = model.merge_and_unload()

        # Save the merged complete model with updated method
        merged_model.save_pretrained(output_dir, safe_serialization=False)
        print(f"LoRA weights merged and complete model saved to {output_dir}")
    else:
        # If not using LoRA, save complete model
        trainer.save_model(output_dir)
        print(f"Complete model saved to {output_dir}")

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Asteroid TTS Instruct Model")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="finetune/training_config.yaml",
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="finetune/lora_config.yaml",
        help="Path to the LoRA configuration file",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for fine-tuning",
    )

    args = parser.parse_args()
    if not args.model_path:
        args.model_path = MODEL_PATH
    elif not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist.")
    if not args.data_dir:
        raise ValueError("Data directory is required.")
    elif not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory '{args.data_dir}' does not exist.")
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    training_config = {}
    if args.training_config:
        import yaml

        if os.path.exists(args.training_config):
            with open(args.training_config, "r") as f:
                training_config = yaml.safe_load(f)
            print(
                f"Successfully loaded training configuration from {args.training_config}: {training_config}"
            )
        else:
            print(
                f"Warning: Configuration file {args.training_config} does not exist, using default parameters."
            )

    lora_cfg = {}
    if args.lora and args.lora_config:
        import yaml

        if os.path.exists(args.lora_config):
            with open(args.lora_config, "r") as f:
                lora_cfg = yaml.safe_load(f)
            print(
                f"Successfully loaded LoRA configuration from {args.lora_config}: {lora_cfg}"
            )
        else:
            print(
                f"Warning: LoRA configuration file {args.lora_config} does not exist, using default LoRA parameters."
            )

    if args.lora:
        print("Using LoRA fine-tuning mode")
    else:
        print("Using full model fine-tuning mode")

    train(
        args.model_path,
        args.data_dir,
        args.output_dir,
        training_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_lora=args.lora,
        lora_cfg=lora_cfg,
    )
