import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
from transformers import AutoModel, AutoProcessor

from generation_utils import (
    merge_rank_jsonl_files,
    prepare_sample,
    resolve_sampling_args,
    run_infer_batch,
    streaming_jsonl_reader,
)

FILE_PATH = Path(os.path.abspath(__file__))
PROJECT_ROOT = FILE_PATH.parent

DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTSD-v1.0"
DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


def _load_model_and_processor(args: argparse.Namespace, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        codec_path=args.codec_model_path,
    )
    if getattr(processor, "audio_tokenizer", None) is not None:
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()

    def _load_model_with_attn(attn_implementation: str):
        return AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)

    if device.startswith("cuda"):
        try:
            model = _load_model_with_attn("flash_attention_2")
        except Exception as flash_exc:
            print(
                "[WARN] flash_attention_2 unavailable, fallback to sdpa. "
                f"error={flash_exc}"
            )
            model = _load_model_with_attn("sdpa")
    else:
        model = _load_model_with_attn("sdpa")

    model.eval()
    return model, processor


def _flush_batch(
    batch_data: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]],
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    device: str,
    save_dir: Path,
    out_fp,
    rank: int,
) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    if len(batch_data) == 0:
        return []

    try:
        run_infer_batch(
            batch_data=batch_data,
            model=model,
            processor=processor,
            mode=args.mode,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            save_dir=save_dir,
            out_fp=out_fp,
        )
        return []
    except Exception as exc:
        print(f"[WARN][rank {rank}] batch failed, fallback to per-sample. error={exc}")

    for item in batch_data:
        sample_id = item[0]
        try:
            run_infer_batch(
                batch_data=[item],
                model=model,
                processor=processor,
                mode=args.mode,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                save_dir=save_dir,
                out_fp=out_fp,
            )
        except Exception as sample_exc:
            print(f"[WARN][rank {rank}] sample {sample_id} skipped. error={sample_exc}")
    return []


def _worker_main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    model, processor = _load_model_and_processor(args, device)
    target_sr = int(processor.model_config.sampling_rate)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if world_size == 1:
        output_jsonl_path = save_dir / "output.jsonl"
    else:
        output_jsonl_path = save_dir / f"output_rank_{rank:06d}.jsonl"

    with open(output_jsonl_path, "w", encoding="utf-8") as out_fp:
        batch_data: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
        for line_no, sample in streaming_jsonl_reader(
            args.input_jsonl,
            rank=rank,
            world_size=world_size,
            skip_invalid_json=True,
        ):
            try:
                sample_id, output_record, conversation = prepare_sample(
                    line_no=line_no,
                    raw_sample=sample,
                    mode=args.mode,
                    processor=processor,
                    target_sr=target_sr,
                    text_normalize_enabled=args.text_normalize,
                    sample_rate_normalize_enabled=args.sample_rate_normalize,
                )
            except Exception as exc:
                print(f"[WARN][rank {rank}] line {line_no} skipped. error={exc}")
                continue

            batch_data.append((sample_id, output_record, conversation))
            if len(batch_data) < args.batch_size:
                continue

            batch_data = _flush_batch(
                batch_data=batch_data,
                model=model,
                processor=processor,
                args=args,
                device=device,
                save_dir=save_dir,
                out_fp=out_fp,
                rank=rank,
            )

        batch_data = _flush_batch(
            batch_data=batch_data,
            model=model,
            processor=processor,
            args=args,
            device=device,
            save_dir=save_dir,
            out_fp=out_fp,
            rank=rank,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "generation",
            "continuation",
            "voice_clone",
            "voice_clone_and_continuation",
        ],
        default="generation",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument(
        "--text_normalize",
        action="store_true",
        default=False,
        help="Normalize input text (symbol cleanup, punctuation normalization, speaker tag merge).",
    )
    parser.add_argument(
        "--sample_rate_normalize",
        action="store_true",
        default=False,
        help="Resample prompt audios to the lowest sample rate before encoding.",
    )
    parser.add_argument(
        "--codec_model_path",
        type=str,
        default=str(DEFAULT_CODEC_PATH),
        help="Path to MOSS-Audio-Tokenizer (HuggingFace format).",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = _parse_args()
    resolve_sampling_args(args)

    torch.manual_seed(42)

    if args.batch_size < 1:
        raise ValueError("`batch_size` must be >= 1.")

    args.save_dir = str(Path(args.save_dir).resolve())
    args.input_jsonl = str(Path(args.input_jsonl).resolve())

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    if world_size <= 1:
        _worker_main(rank=0, world_size=1, args=args)
        return

    mp.spawn(
        _worker_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )

    rank_jsonl_paths = [
        save_dir / f"output_rank_{rank:06d}.jsonl" for rank in range(world_size)
    ]
    output_jsonl_path = save_dir / "output.jsonl"
    merge_rank_jsonl_files(
        output_jsonl_path=output_jsonl_path, rank_jsonl_paths=rank_jsonl_paths
    )

    for p in rank_jsonl_paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
