import json
import torch
import torchaudio
import accelerate
import argparse
import os

from generation_utils import load_model, process_batch

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

def main():
    parser = argparse.ArgumentParser(description="TTS inference with Asteroid model")
    parser.add_argument("--jsonl", default="examples/examples.jsonl",help="Path to JSONL file (default: examples/examples.jsonl)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None)")
    parser.add_argument("--output_dir", default="outputs",
                       help="Output directory for generated audio files (default: outputs)")
    parser.add_argument("--summary_file", default=None,
                       help="Path to save summary jsonl file (default: None)")
    parser.add_argument("--use_normalize", action="store_true", default=False,
                       help="Whether to use text normalization (default: False)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                       help="Model data type (default: bf16)")
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2",
                       help="Attention implementation (default: flash_attention_2)")
    parser.add_argument("--silence_duration", type=float, default=0,
                       help="Silence duration between speech prompt and generated speech, which can be used to avoid noise problem at the beginning of generated audio")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_mapping[args.dtype]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {args.dtype} ({torch_dtype})")
    print(f"Using attention implementation: {args.attn_implementation}")
    
    # Load models
    print("Loading models...")
    tokenizer, model, spt = load_model(MODEL_PATH, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH, 
                                      torch_dtype=torch_dtype, attn_implementation=args.attn_implementation)
    spt = spt.to(device)
    model = model.to(device)
    
    # Load the items from the JSONL file
    try:
        with open(args.jsonl, "r") as f:
            items = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(items)} items from {args.jsonl}")
    except FileNotFoundError:
        print(f"Error: JSONL file '{args.jsonl}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        return
    
    # Fix the seed for reproducibility
    if args.seed is not None:
        accelerate.utils.set_seed(args.seed)
        print(f"Set random seed to {args.seed}")
    
    # Process the batch of items
    print("Starting inference...")
    actual_texts_data, audio_results = process_batch(
        batch_items=items,
        tokenizer=tokenizer,
        model=model,
        spt=spt,
        device=device,
        system_prompt=SYSTEM_PROMPT,
        start_idx=0,
        use_normalize=args.use_normalize,
        silence_duration=args.silence_duration
    )
    
    # Save summary if requested
    if args.summary_file:
        summary_data = []
        for item in actual_texts_data:
            summary_data.append({
                "text": item["original_text"],
                "normalized_text": item["normalized_text"],
                "final_text": item["final_text"]
            })
        
        with open(args.summary_file, "w", encoding="utf-8") as f:
            for item in summary_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved summary to {args.summary_file}")
    
    # Save the audio results to files
    saved_count = 0
    for idx, audio_result in enumerate(audio_results):
        if audio_result is not None:
            output_path = os.path.join(args.output_dir, f"output_{idx}.wav")
            torchaudio.save(
                output_path,
                audio_result["audio_data"],
                audio_result["sample_rate"]
            )
            print(f"Saved audio to {output_path}")
            saved_count += 1
        else:
            print(f"Skipping sample {idx} due to generation error")
    
    print(f"Inference completed. Saved {saved_count}/{len(items)} audio files to {args.output_dir}")

if __name__ == "__main__":
    main()