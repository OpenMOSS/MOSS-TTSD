import json
import torch
import numpy as np
import argparse
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation_utils import process_jsonl_item, normalize_text, load_audio_data
from transformers import AutoTokenizer
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"
MAX_CHANNELS = 8
SILENCE_DURATION = 0.0  # Fixed silence duration: 0 seconds

def load_tokenizer(model_path, spt_config_path, spt_checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    spt = XY_Tokenizer.load_from_checkpoint(config_path=spt_config_path, ckpt_path=spt_checkpoint_path)
    spt.eval()
    return tokenizer, spt

def process_inputs(tokenizer, spt, prompt, text, device, audio_data=None, reference_audio=None, main_audio=None, max_channels=8, pad_token=1024):
    # Decompose template into multiple parts
    # 1. Style prompt part
    seg1 = f"<|begin_of_style|>{prompt}<|end_of_style|>\n<|begin_of_text|>"
    inputs1 = np.array(tokenizer.encode(seg1))
    inputs_expanded1 = np.full((len(inputs1), max_channels), pad_token)
    inputs_expanded1[:, 0] = inputs1
    labels1 = np.full(inputs_expanded1.shape, -100)  # Style prompt does not compute loss
    
    # 2. Text part
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    inputs2 = np.array(text_tokens)
    inputs_expanded2 = np.full((len(inputs2), max_channels), pad_token)
    inputs_expanded2[:, 0] = inputs2
    labels2 = np.full(inputs_expanded2.shape, -100)  # Text does not compute loss
    
    # 3. Text end/speech begin part
    seg3 = f"<|end_of_text|>\n<|begin_of_speech|>"
    inputs3 = np.array(tokenizer.encode(seg3))
    inputs_expanded3 = np.full((len(inputs3), max_channels), pad_token)
    inputs_expanded3[:, 0] = inputs3
    labels3 = np.full(inputs_expanded3.shape, -100)  # Start marker does not compute loss

    # 4. Audio processing part
    audio_token = None
    if reference_audio is not None and main_audio is not None:
        # New format: process two audio files separately and then concatenate tokens
        try:
            # Add silence to the end of each audio
            silence_samples = int(SILENCE_DURATION * 16000)
            silence = torch.zeros(1, silence_samples)
            
            # Ensure audio has correct shape [1, samples] 
            if len(reference_audio.shape) == 1:
                reference_audio = reference_audio.unsqueeze(0)
            if len(main_audio.shape) == 1:
                main_audio = main_audio.unsqueeze(0)
            
            # Add silence to each audio
            ref_audio_with_silence = torch.cat([reference_audio, silence], dim=1)
            main_audio_with_silence = torch.cat([main_audio, silence], dim=1)

            with torch.no_grad():
                # Encode two audio files separately
                ref_encode_result = spt.encode([ref_audio_with_silence.squeeze().to(device)])
                main_encode_result = spt.encode([main_audio_with_silence.squeeze().to(device)])
                
                ref_audio_token = ref_encode_result["codes_list"][0].permute(1, 0).cpu().numpy()
                main_audio_token = main_encode_result["codes_list"][0].permute(1, 0).cpu().numpy()
                
                # Concatenate at token level
                audio_token = np.concatenate([ref_audio_token, main_audio_token], axis=0)

        except Exception as e:
            print(f"Error processing two audio files: {e}")
            raise
            
    elif audio_data is not None:
        # Original format: single audio processing
        try:
            wav = audio_data

            # Add fixed silence at the end of audio (using 16k sample rate)
            silence_samples = int(SILENCE_DURATION * 16000)
            silence = torch.zeros(wav.shape[0], silence_samples)
            wav = torch.cat([wav, silence], dim=1)

            with torch.no_grad():
                # Use SPT encoding
                encode_result = spt.encode([wav.squeeze().to(device)])
                audio_token = encode_result["codes_list"][0].permute(1, 0).cpu().numpy()

        except Exception as e:
            print(f"Error processing audio data: {e}")
            raise

    if audio_token is not None:
        # Add offset (only for the first layer)
        audio_token[:, 0] = audio_token[:, 0] + 151665
        
        # Channel count alignment processing
        if audio_token.shape[1] > max_channels:
            audio_token = audio_token[:, :max_channels]
        elif audio_token.shape[1] < max_channels:
            padded = np.full((audio_token.shape[0], max_channels), pad_token)
            padded[:, :audio_token.shape[1]] = audio_token
            audio_token = padded
            
        labels4 = audio_token.copy()  # Audio tokens need to compute loss
    else:
        raise ValueError("No audio data provided")
    
    # 5. Speech end part
    seg5 = "<|end_of_speech|>"
    inputs5 = np.array(tokenizer.encode(seg5))
    inputs_expanded5 = np.full((len(inputs5), max_channels), pad_token)
    inputs_expanded5[:, 0] = inputs5
    labels5 = np.full(inputs_expanded5.shape, -100)
    labels5[:, 0] = inputs_expanded5[:, 0]  # End marker needs to be learned

    # Concatenate all parts
    input_ids = np.concatenate([
        inputs_expanded1,   # Style prompt
        inputs_expanded2,   # Text
        inputs_expanded3,   # Speech start marker
        audio_token,        # Speech tokens (first layer with offset added)
        inputs_expanded5    # End marker
    ])

    labels = np.concatenate([
        labels1,            # Style prompt (no loss computation)
        labels2,            # Text (no loss computation)
        labels3,            # Start marker (no loss computation)
        labels4,            # Speech tokens (compute loss)
        labels5             # End marker (compute loss)
    ])

    # Calculate length information
    total_length = input_ids.shape[0]  # Total token length
    audio_length = audio_token.shape[0]  # Audio token length

    return input_ids, labels, total_length, audio_length

def process_data(
        jsonl: str ,
        model_path: str ,
        output_dir: str ,
        data_name: str = "processd_data",
        use_normalize: bool = True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    tokenizer, spt = load_tokenizer(model_path, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device)
    
    # Load the items from the JSONL file
    try:
        with open(jsonl, "r") as f:
            items = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(items)} items from {jsonl}")
    except FileNotFoundError:
        print(f"Error: JSONL file '{jsonl}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        return

    # Store all processed data and length information
    all_data = []
    offsets = []
    tokens_lengths = []  # Added: store total token length
    tims_lengths = []    # Added: store audio token length
    
    for idx, item in enumerate(items):
        # Support two JSONL formats
        # Format 1: {"file_path": "path/to/audio.wav", "full_transcript": "speech content..."}
        # Format 2: {"reference_audio": "path1", "reference_text": "text1", "audio": "path2", "text": "text2"}
        
        if "file_path" in item and "full_transcript" in item:
            # Original format
            file_path = item["file_path"]
            full_text = item["full_transcript"]
            
            # Check if audio file exists
            if not file_path:
                print(f"Warning: Item {idx} has empty file_path, skipping...")
                continue
                
            if not os.path.exists(file_path):
                print(f"Warning: Audio file not found: {file_path}, skipping item {idx}...")
                continue
                
            try:
                # load_audio_data already includes 16kHz mono conversion functionality
                audio_data = load_audio_data(file_path)
            except Exception as e:
                print(f"Warning: Failed to load audio from {file_path}: {e}, skipping item {idx}...")
                continue
            
            # Apply text normalization based on parameter
            if use_normalize:
                full_text = normalize_text(full_text)
            
            # Replace speaker tags
            final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")
            
            # Process single audio format
            input_id, labels, total_length, audio_length = process_inputs(tokenizer, spt, SYSTEM_PROMPT, final_text, device, audio_data, max_channels=MAX_CHANNELS)
                
        elif "reference_audio" in item and "reference_text" in item and "audio" in item and "text" in item:
            # New format: requires concatenation
            reference_audio_path = item["reference_audio"]
            reference_text = item["reference_text"]
            audio_path = item["audio"]
            text = item["text"]
            
            # Concatenate text
            full_text = reference_text + text
            
            # Check if both audio files exist
            if not reference_audio_path or not audio_path:
                print(f"Warning: Item {idx} has empty audio paths, skipping...")
                continue
                
            if not os.path.exists(reference_audio_path):
                print(f"Warning: Reference audio file not found: {reference_audio_path}, skipping item {idx}...")
                continue
                
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}, skipping item {idx}...")
                continue
            
            try:
                # load_audio_data already includes 16kHz mono conversion functionality
                reference_audio = load_audio_data(reference_audio_path)
                main_audio = load_audio_data(audio_path)
                
                # Apply text normalization based on parameter
                if use_normalize:
                    full_text = normalize_text(full_text)
                
                # Replace speaker tags
                final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")
                
                # Pass two separate audio files to process_inputs
                input_id, labels, total_length, audio_length = process_inputs(tokenizer, spt, SYSTEM_PROMPT, final_text, device, 
                                        reference_audio=reference_audio, main_audio=main_audio, 
                                        max_channels=MAX_CHANNELS)
                
            except Exception as e:
                print(f"Warning: Failed to load audio files: {e}, skipping item {idx}...")
                continue
                
        else:
            print(f"Warning: Item {idx} missing required fields for both supported formats, skipping...")
            continue
        
        # Create data entry containing input_ids and labels
        data_entry = {
            'input_ids': input_id.tolist(),  # shape: [seq_len, 8]
            'labels': labels.tolist()        # shape: [seq_len, 8]
        }
        
        all_data.append(data_entry)
        tokens_lengths.append(total_length)    # Record total length
        tims_lengths.append(audio_length)      # Record audio length

        print(f"Processed item {idx + 1}/{len(items)}: input_ids shape {input_id.shape}, labels shape {labels.shape}, total_len={total_length}, audio_len={audio_length}")
    
    # Save pkl file - serialize one by one
    output_pkl_path = os.path.join(output_dir, f"{data_name}.pkl")
    with open(output_pkl_path, 'wb') as f:
        for data_entry in all_data:
            offsets.append(f.tell())  # Record current position as offset
            pickle.dump(data_entry, f)  # Serialize each data entry separately
    
    # Save metadata file containing three arrays
    output_meta_path = os.path.join(output_dir, f"{data_name}_metas.npy")
    pointers = np.array(offsets)
    tokens = np.array(tokens_lengths)
    tims = np.array(tims_lengths)
    
    # Follow reference code format: stack([pointers, tokens, tims])
    np.save(output_meta_path, np.stack([pointers, tokens, tims]))
    
    print(f"Saved {len(all_data)} processed items to {output_pkl_path}")
    print(f"Saved metadata (pointers, tokens_lengths, tims_lengths) to {output_meta_path}")
    print(f"Total sequences processed: {len(all_data)}")
    print(f"Average total length: {np.mean(tokens_lengths):.1f}, Average audio length: {np.mean(tims_lengths):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS inference with Asteroid model")
    parser.add_argument("--jsonl", type=str, required=True,
                       help="Path to JSONL file")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for generated audio files")
    parser.add_argument("--data_name", default="processed_data",
                       help="Name of the processed data file (default: processed_data)")
    parser.add_argument("--use_normalize", action="store_true", default=False,
                       help="Whether to use text normalization (default: False)")
    
    args = parser.parse_args()
    
    if not args.jsonl:
        raise ValueError("JSONL file path is required.")
    elif not os.path.exists(args.jsonl):
        raise ValueError(f"JSONL file '{args.jsonl}' does not exist.")
    if not args.model_path:
        args.model_path = MODEL_PATH
    elif not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist.")
    if not args.output_dir:
        raise ValueError("Output directory is required.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    process_data(args.jsonl, args.model_path, args.output_dir, args.data_name, args.use_normalize)
