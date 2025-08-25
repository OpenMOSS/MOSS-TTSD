import json
from queue import Queue
import torch
import torchaudio
import accelerate
import argparse
import os
from tqdm import tqdm
import io

from transformers.generation.streamers import BaseStreamer

from generation_utils import (
    load_model, 
    process_jsonl_item, 
    normalize_text, 
    load_audio_data, 
    process_inputs, 
    shifting_inputs, 
    rpadding, 
    find_max_valid_positions,
    process_batch
)
from modeling_asteroid import AsteroidTTSInstruct
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8

class AudioIteratorStreamer(BaseStreamer):

    MAX_TOKEN_LENGTH = 16384
    CHUNK_SIZE = 30
    OVERLAP_SECONDS = 10

    def __init__(
        self, 
        tokenizer,
        model: AsteroidTTSInstruct,
        spt: XY_Tokenizer,
        device,
        use_tqdm: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.spt = spt
        self.device = device
        self.use_tqdm = use_tqdm
        self.speech_offset = model.config.speech_token_range[0]
        self.channels = model.config.channels
        self.duration_code_length = int(self.CHUNK_SIZE * spt.input_sample_rate // spt.encoder_downsample_rate)
        self.overlap_code_length = int(self.OVERLAP_SECONDS * spt.input_sample_rate // spt.encoder_downsample_rate)
        self.valid_code_length = self.duration_code_length - self.overlap_code_length
        self.valid_wav_length = int(self.valid_code_length * spt.decoder_upsample_rate)
        print(f"Speech offset: {self.speech_offset}")
        print(f"Duration code length: {self.duration_code_length}")
        print(f"Overlap code length: {self.overlap_code_length}")
        print(f"Valid code length: {self.valid_code_length}")
        print(f"Valid wav length: {self.valid_wav_length}")

        self.next_tokens_are_prompt = True
        self.token_cache = torch.zeros(self.MAX_TOKEN_LENGTH, self.channels, dtype=torch.long, device=device)
        self.token_cache_length = 0
        self.decoded_idx = 0
        self.audio_queue = Queue()
        self.stop_signal = None
        
        # tqdm related
        self.pbar = None
        if self.use_tqdm:
            self.pbar = tqdm(desc="Processing tokens", unit="token", total=None)

    def decode(self, last_chunk=False):
        duration_to_decode = min(self.duration_code_length, self.token_cache_length - self.decoded_idx - self.channels + 1)

        speech_ids = torch.full((duration_to_decode, self.channels), 0).to(self.device)
        for j in range(self.channels):
            speech_ids[..., j] = self.token_cache[self.decoded_idx + j : self.decoded_idx + j + duration_to_decode, j]
            if j == 0:
                speech_ids[..., j] = speech_ids[..., j] - self.speech_offset

        # Decode generated audio
        with torch.no_grad():
            chunk_codes = speech_ids.permute(1, 0).unsqueeze(1)  # Convert to SPT expected format
            decode_result = self.spt.inference_detokenize(chunk_codes, torch.tensor([duration_to_decode], device=self.device))
            audio_result = decode_result["y"][0].cpu().detach()
            if not last_chunk:
                audio_result = audio_result[:, :self.valid_wav_length]

        self.audio_queue.put(audio_result)
        # will be out of bound of last chunk is not full
        self.decoded_idx += self.valid_code_length

    def put(self, value: torch.LongTensor):
        if self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache[self.token_cache_length, :] = value
        self.token_cache_length += 1
        
        # update tqdm pbar
        if self.use_tqdm and self.pbar is not None:
            self.pbar.set_postfix({
                "cache_len": self.token_cache_length,
                "decoded": self.decoded_idx,
                "cache_usage": f"{self.token_cache_length/self.MAX_TOKEN_LENGTH*100:.1f}%"
            })
            self.pbar.update(1)
        
        if self.token_cache_length >= self.MAX_TOKEN_LENGTH:
            raise ValueError("Token cache is full")

        if self.token_cache_length >= self.decoded_idx + self.duration_code_length + self.channels - 1:
            self.decode(last_chunk=False)
    
    def end(self):
        self.decode(last_chunk=True)
        self.audio_queue.put(self.stop_signal)
        
        # close tqdm
        if self.use_tqdm and self.pbar is not None:
            self.pbar.close()

    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.audio_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

def streamer(
    batch_items, 
    tokenizer,
    model: AsteroidTTSInstruct,
    spt: XY_Tokenizer,
    device,
    system_prompt, 
    use_normalize=False,
    use_tqdm=False,
    ):
    """Process a batch of data items and generate audio, return audio data and metadata"""
    # Prepare batch data
    batch_size = len(batch_items)
    assert batch_size == 1, "Batch size must be 1 for streaming"
    texts = []
    prompts = [system_prompt] * batch_size
    prompt_audios = []
    actual_texts_data = []  # Store actual text data used
    
    # Extract text and audio from each sample
    for i, item in enumerate(batch_items):
        # Use new processing function
        processed_item = process_jsonl_item(item)
        
        text = processed_item["text"]
        prompt_text = processed_item["prompt_text"]
        
        # Merge text, if prompt_text is empty, full_text is just text
        full_text = prompt_text + text if prompt_text else text
        original_full_text = full_text  # Save original text
        
        # Apply text normalization based on parameter
        if use_normalize:
            full_text = normalize_text(full_text)
        
        # Replace speaker tags
        final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")
        texts.append(final_text)
        
        # Save actual text information used
        actual_texts_data.append({
            "index": i,
            "original_text": original_full_text,
            "normalized_text": normalize_text(original_full_text) if use_normalize else None,
            "final_text": final_text,
            "use_normalize": use_normalize
        })
        
        # Get reference audio
        prompt_audios.append(processed_item["prompt_audio"])
    
    # Process inputs
    input_ids_list = []
    for i, (text, prompt, audio_path) in enumerate(zip(texts, prompts, prompt_audios)):
        # Load audio data here
        audio_data = load_audio_data(audio_path) if audio_path else None
        inputs = process_inputs(tokenizer, spt, prompt, text, device, audio_data)
        inputs = shifting_inputs(inputs, tokenizer)
        input_ids_list.append(inputs)
    
    # Pad batch inputs
    input_ids, attention_mask = rpadding(input_ids_list, MAX_CHANNELS, tokenizer)
    
    # Batch generation
    print(f"Starting batch audio generation...")
    
    # Move inputs to GPU
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    streamer = AudioIteratorStreamer(tokenizer, model, spt, device, use_tqdm=use_tqdm)

    from threading import Thread
    thread = Thread(target=model.generate, kwargs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "streamer": streamer,
    })
    thread.start()

    yield from streamer

    thread.join()


def main():
    parser = argparse.ArgumentParser(description="TTS inference with Asteroid model")
    parser.add_argument("--jsonl", default="examples/examples.jsonl", 
                       help="Path to JSONL file (default: examples/examples.jsonl)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None)")
    parser.add_argument("--output_dir", default="outputs/streamer",
                       help="Output directory for generated audio files (default: outputs)")
    parser.add_argument("--use_normalize", action="store_true", default=True,
                       help="Whether to use text normalization (default: True)")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                       help="Model data type (default: bf16)")
    parser.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2",
                       help="Attention implementation (default: flash_attention_2)")
    parser.add_argument("--use_tqdm", action="store_true", default=False,
                       help="Whether to show progress bar using tqdm (default: False)")
    
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
    print("Starting streaming inference...")
    # Create output directory for FLAC files
    flac_output_dir = args.output_dir
    os.makedirs(flac_output_dir, exist_ok=True)
    
    audio_generator = streamer(
        batch_items=[items[0]],
        tokenizer=tokenizer,
        model=model,
        spt=spt,
        device=device,
        system_prompt=SYSTEM_PROMPT,
        use_normalize=args.use_normalize,
        use_tqdm=args.use_tqdm
    )
    
    # Save each FLAC chunk to disk
    full_audio = []
    for i, audio in enumerate(audio_generator):
        chunk_path = os.path.join(flac_output_dir, f'chunk_{i}.flac')
        torchaudio.save(chunk_path, audio, spt.output_sample_rate, format="flac")
        full_audio.append(audio)

    full_audio = torch.cat(full_audio, dim=1)
    full_audio_path = os.path.join(flac_output_dir, 'full_audio.flac')
    torchaudio.save(full_audio_path, full_audio, spt.output_sample_rate, format="flac")
    print(f"Saved full audio to: {full_audio_path}")

if __name__ == "__main__":
    main()