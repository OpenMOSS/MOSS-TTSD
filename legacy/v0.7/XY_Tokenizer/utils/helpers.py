import glob
import logging
import os
import re
import sys

import debugpy
import numpy as np
import torch
import torchaudio


def count_params_by_module(model_name, model):
    logging.info(f"Counting num_parameters of {model_name}:")

    param_stats = {}
    total_params = 0  # Count total parameters
    total_requires_grad_params = 0  # Count parameters with requires_grad=True
    total_no_grad_params = 0  # Count parameters with requires_grad=False

    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name not in param_stats:
            param_stats[module_name] = {"total": 0, "requires_grad": 0, "no_grad": 0}

        param_num = param.numel()
        param_stats[module_name]["total"] += param_num
        total_params += param_num

        if param.requires_grad:
            param_stats[module_name]["requires_grad"] += param_num
            total_requires_grad_params += param_num
        else:
            param_stats[module_name]["no_grad"] += param_num
            total_no_grad_params += param_num

    # Calculate maximum width for each column
    max_module_name_length = max(len(module) for module in param_stats)
    max_param_length = max(
        len(f"{stats['total'] / 1e6:.2f}M") for stats in param_stats.values()
    )

    # Output parameter statistics for each module
    for module, stats in param_stats.items():
        logging.info(
            f"\t{module:<{max_module_name_length}}: "
            f"Total: {stats['total'] / 1e6:<{max_param_length}.2f}M, "
            f"Requires Grad: {stats['requires_grad'] / 1e6:<{max_param_length}.2f}M, "
            f"No Grad: {stats['no_grad'] / 1e6:<{max_param_length}.2f}M"
        )

    # Output total parameter statistics
    logging.info(f"\tTotal parameters: {total_params / 1e6:.2f}M parameters")
    logging.info(
        f"\tRequires Grad parameters: {total_requires_grad_params / 1e6:.2f}M parameters"
    )
    logging.info(f"\tNo Grad parameters: {total_no_grad_params / 1e6:.2f}M parameters")
    logging.info(f"################################################################")


def load_and_resample_audio(audio_path, target_sample_rate):
    wav, raw_sample_rate = torchaudio.load(audio_path)  # (1, T)   tensor
    if raw_sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(
            wav, raw_sample_rate, target_sample_rate
        )  # tensor
    return wav.squeeze()


def set_logging():
    rank = os.environ.get("RANK", 0)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=f"%(asctime)s [RANK {rank}] (%(module)s:%(lineno)d) %(levelname)s : %(message)s",
    )


def waiting_for_debug(ip, port):
    rank = os.environ.get("RANK", "0")
    debugpy.listen((ip, port))  # Replace localhost with cluster node IP
    logging.info(f"[rank = {rank}] Waiting for debugger attach...")
    debugpy.wait_for_client()
    logging.info(f"[rank = {rank}] Debugger attached")


def load_audio(audio_path, target_sample_rate):
    # Load audio file, wav shape: (channels, time)
    wav, raw_sample_rate = torchaudio.load(audio_path)

    # If multi-channel, convert to mono by averaging across channels
    if wav.shape[0] > 1:
        wav = torch.mean(
            wav, dim=0, keepdim=True
        )  # Average across channels, keep channel dim

    # Resample if necessary
    if raw_sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate)

    # Convert to numpy, add channel dimension, then back to tensor with desired shape
    wav = np.expand_dims(wav.squeeze(0).numpy(), axis=1)  # Shape: (time, 1)
    wav = torch.tensor(wav).reshape(1, 1, -1)  # Shape: (1, 1, time)

    return wav


def save_audio(audio_outpath, audio_out, sample_rate):
    torchaudio.save(
        audio_outpath,
        audio_out,
        sample_rate=sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    logging.info(f"Successfully saved audio at {audio_outpath}")


def find_audio_files(input_dir):
    audio_extensions = ["*.flac", "*.mp3", "*.wav"]
    audios_input = []
    for ext in audio_extensions:
        audios_input.extend(
            glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
        )
    logging.info(f"Found {len(audios_input)} audio files in {input_dir}")
    return sorted(audios_input)


def normalize_text(text):
    # Remove all punctuation (including English and Chinese punctuation)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    # Convert to lowercase (effective for English, no effect on Chinese)
    text = text.lower()
    # Remove extra spaces
    text = " ".join(text.split())
    return text
