import os
import re

import numpy as np
import torch
import torchaudio

MAX_CHANNELS = 8


def load_model(
    model_path,
    spt_config_path,
    spt_checkpoint_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
):
    from transformers import AutoTokenizer

    from modeling_asteroid import AsteroidTTSInstruct
    from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AsteroidTTSInstruct.from_pretrained(
        model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
    )
    spt = XY_Tokenizer.load_from_checkpoint(
        config_path=spt_config_path, ckpt_path=spt_checkpoint_path
    )

    model.eval()
    spt.eval()
    return tokenizer, model, spt


def process_jsonl_item(item):
    """Process JSONL data items and extract audio and text information according to the new format"""
    base_path = item.get("base_path", "")
    text = item.get("text", "")

    prompt_audio = None
    prompt_text = ""

    # Process prompt audio and text
    if "prompt_audio" in item and "prompt_text" in item:
        print("Using prompt_audio and prompt_text directly from item.")
        # If prompt_audio and prompt_text exist, use them directly
        prompt_audio_val = item["prompt_audio"]
        if prompt_audio_val:  # Only assign if not empty
            prompt_audio = prompt_audio_val
            prompt_text = item["prompt_text"]

            # Only perform path joining when prompt_audio is a string path
            if isinstance(prompt_audio, str) and base_path and prompt_audio:
                prompt_audio = os.path.join(base_path, prompt_audio)
    else:
        # Otherwise, merge speaker1 and speaker2 information
        prompt_audio_speaker1 = item.get("prompt_audio_speaker1", "")
        prompt_text_speaker1 = item.get("prompt_text_speaker1", "")
        prompt_audio_speaker2 = item.get("prompt_audio_speaker2", "")
        prompt_text_speaker2 = item.get("prompt_text_speaker2", "")

        has_speaker1_audio = (
            isinstance(prompt_audio_speaker1, str) and prompt_audio_speaker1
        ) or isinstance(prompt_audio_speaker1, tuple)
        has_speaker2_audio = (
            isinstance(prompt_audio_speaker2, str) and prompt_audio_speaker2
        ) or isinstance(prompt_audio_speaker2, tuple)

        if has_speaker1_audio or has_speaker2_audio:
            print("Using speaker1 and speaker2 information for prompt audio and text.")
            # Process audio: if it's a string path, perform path joining; if it's a tuple, use directly
            if isinstance(prompt_audio_speaker1, str):
                speaker1_audio = (
                    os.path.join(base_path, prompt_audio_speaker1)
                    if base_path and prompt_audio_speaker1
                    else prompt_audio_speaker1
                )
            else:
                speaker1_audio = prompt_audio_speaker1  # Use tuple directly

            if isinstance(prompt_audio_speaker2, str):
                speaker2_audio = (
                    os.path.join(base_path, prompt_audio_speaker2)
                    if base_path and prompt_audio_speaker2
                    else prompt_audio_speaker2
                )
            else:
                speaker2_audio = prompt_audio_speaker2  # Use tuple directly

            prompt_audio = {"speaker1": speaker1_audio, "speaker2": speaker2_audio}

        # Merge text
        temp_prompt_text = ""
        if prompt_text_speaker1:
            temp_prompt_text += f"[S1]{prompt_text_speaker1}"
        if prompt_text_speaker2:
            temp_prompt_text += f"[S2]{prompt_text_speaker2}"
        prompt_text = temp_prompt_text.strip()

    return {"text": text, "prompt_text": prompt_text, "prompt_audio": prompt_audio}


def load_audio_data(prompt_audio, target_sample_rate=16000):
    """Load audio data and return processed audio tensor

    Args:
        prompt_audio: Can be in the following formats:
            - String: audio file path
            - Tuple: (wav, sr) result from torchaudio.load
            - Dict: {"speaker1": path_or_tuple, "speaker2": path_or_tuple}
    """
    if prompt_audio is None:
        return None

    try:
        # Check if prompt_audio is a dictionary (containing speaker1 and speaker2)
        if (
            isinstance(prompt_audio, dict)
            and "speaker1" in prompt_audio
            and "speaker2" in prompt_audio
        ):
            # Process audio from both speakers separately
            wav1, sr1 = _load_single_audio(prompt_audio["speaker1"])
            wav2, sr2 = _load_single_audio(prompt_audio["speaker2"])
            # Merge audio from both speakers
            wav = merge_speaker_audios(wav1, sr1, wav2, sr2, target_sample_rate)
            if wav is None:
                return None
        else:
            # Single audio
            wav, sr = _load_single_audio(prompt_audio)
            # Resample to 16k
            if sr != target_sample_rate:
                wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
            # Ensure mono channel
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)  # Convert multi-channel to mono
            if len(wav.shape) == 1:
                wav = wav.unsqueeze(0)

        return wav
    except Exception as e:
        print(f"Error loading audio data: {e}")
        raise


def _load_single_audio(audio_input):
    """Load single audio, supports file path or (wav, sr) tuple

    Args:
        audio_input: String (file path) or tuple (wav, sr)

    Returns:
        tuple: (wav, sr)
    """
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        # Already a (wav, sr) tuple
        wav, sr = audio_input
        return wav, sr
    elif isinstance(audio_input, str):
        # Is a file path, needs to be loaded
        wav, sr = torchaudio.load(audio_input)
        return wav, sr
    else:
        raise ValueError(f"Unsupported audio input format: {type(audio_input)}")


def merge_speaker_audios(wav1, sr1, wav2, sr2, target_sample_rate=16000):
    """Merge audio data from two speakers"""
    try:
        # Process first audio
        if sr1 != target_sample_rate:
            wav1 = torchaudio.functional.resample(wav1, sr1, target_sample_rate)
        # Ensure mono channel
        if wav1.shape[0] > 1:
            wav1 = wav1.mean(dim=0, keepdim=True)  # Convert multi-channel to mono
        if len(wav1.shape) == 1:
            wav1 = wav1.unsqueeze(0)

        # Process second audio
        if sr2 != target_sample_rate:
            wav2 = torchaudio.functional.resample(wav2, sr2, target_sample_rate)
        # Ensure mono channel
        if wav2.shape[0] > 1:
            wav2 = wav2.mean(dim=0, keepdim=True)  # Convert multi-channel to mono
        if len(wav2.shape) == 1:
            wav2 = wav2.unsqueeze(0)

        # Concatenate audio
        merged_wav = torch.cat([wav1, wav2], dim=1)
        return merged_wav
    except Exception as e:
        print(f"Error merging audio: {e}")
        raise


def process_inputs(
    tokenizer,
    spt,
    prompt,
    text,
    device,
    silence_duration,
    audio_data=None,
    max_channels=8,
    pad_token=1024,
):
    seq = f"<|begin_of_style|>{prompt}<|end_of_style|>\n<|begin_of_text|>{text}<|end_of_text|>\n<|begin_of_speech|>"
    inputs1 = np.array(tokenizer.encode(seq))
    input_ids = np.full((inputs1.shape[0], max_channels), pad_token)
    input_ids[:, 0] = inputs1

    if audio_data is not None:
        try:
            # audio_data should now be a processed audio tensor
            wav = audio_data

            # Add fixed 5-second silence at the end of audio (using 16k sample rate)
            silence_samples = int(silence_duration * 16000)
            silence = torch.zeros(wav.shape[0], silence_samples)
            wav = torch.cat([wav, silence], dim=1)

            with torch.no_grad():
                # Use SPT encoding
                encode_result = spt.encode([wav.squeeze().to(device)])
                audio_token = (
                    encode_result["codes_list"][0].permute(1, 0).cpu().numpy()
                )  # Adjust dimension order

            # similar to DAC encoding adjustment
            audio_token[:, 0] = (
                audio_token[:, 0] + 151665
            )  # Keep this line if offset is needed, otherwise delete
            input_ids = np.concatenate([input_ids, audio_token])
        except Exception as e:
            print(f"Error processing audio data: {e}")
            raise

    return input_ids


def shifting_inputs(input_ids, tokenizer, pad_token=1024, max_channels=8):
    seq_len = input_ids.shape[0]
    new_seq_len = seq_len + max_channels - 1
    shifted_input_ids = np.full((new_seq_len, max_channels), pad_token, dtype=np.int64)
    shifted_input_ids[:, 0] = np.full(
        new_seq_len, tokenizer.pad_token_id, dtype=np.int64
    )
    for i in range(max_channels):
        shifted_input_ids[i : (seq_len + i), i] = input_ids[:, i]
    return shifted_input_ids


def rpadding(input_ids, channels, tokenizer):
    attention_masks = [np.ones(inputs.shape[0]) for inputs in input_ids]
    max_length = max(ids.shape[0] for ids in input_ids)
    padded_input_ids, padded_attns = [], []

    for ids, attn in zip(input_ids, attention_masks):
        pad_len = max_length - ids.shape[0]
        input_pad = np.full((pad_len, channels), 1024)
        input_pad[:, 0] = tokenizer.pad_token_id
        padded_input_ids.append(np.concatenate([input_pad, ids]))
        attn_pad = np.zeros(pad_len)
        padded_attns.append(np.concatenate([attn_pad, attn]))

    input_ids = torch.tensor(np.stack(padded_input_ids))
    attention_mask = torch.tensor(np.stack(padded_attns))

    return input_ids, attention_mask


def find_max_valid_positions(C: torch.Tensor, invalid_value=1024) -> torch.Tensor:
    values = C[:, :, 1]
    mask = values != invalid_value
    reversed_mask = mask.flip(dims=[1])
    reversed_indices = torch.argmax(reversed_mask.int(), dim=1)
    seq_len = C.size(1)
    original_indices = seq_len - 1 - reversed_indices
    has_valid = mask.any(dim=1)
    original_indices = torch.where(has_valid, original_indices, -1)
    return original_indices


def normalize_text(text: str) -> str:
    """
    Normalize multi-speaker script.

    1. Don't preserve line breaks.
    2. Remove brackets for non-speaker tags (if [] doesn't contain S1/S2...Sx format, remove the brackets themselves).
    3. Remove decorative symbols: 【】《》（）『』「」"-“” .
    4. Internal punctuation ！；：、 → ，；only allow ？ and ，。
    5. Multiple 。 keep only the last one, others → ，。
    6. Replace consecutive "哈" (>=2) with "(笑)".
    7. Auto-recognize [S1] / [S2] … tags; if missing, treat as whole segment.
    8. Merge adjacent identical speaker tags.
    """
    # Replace [1], [2] etc. format with [S1], [S2] etc. format
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

    # Remove decorative characters
    remove_chars = "【】《》（）『』「」" '"-_“”～~'

    # Remove brackets for non-speaker tags (keep content, only remove brackets themselves)
    text = re.sub(r"\[(?!S\d+\])([^\]]*)\]", r"\1", text)

    # Use positive lookahead to split text by speaker tags (tags themselves are still preserved)
    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Extract tags
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = m.groups() if m else ("", seg)

        # Remove irrelevant symbols
        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)

        # Handle consecutive "哈" characters: replace 2 or more with "(笑)"
        content = re.sub(r"哈{2,}", "(笑)", content)

        # Handle English laughter (e.g., "haha", "ha ha")
        content = re.sub(r"\b(ha(\s*ha)+)\b", "(laughs)", content, flags=re.IGNORECASE)

        # First handle multi-character punctuation marks
        content = content.replace("——", "，")
        content = content.replace("……", "，")

        # Handle single-character internal punctuation marks
        internal_punct_map = str.maketrans(
            {
                "！": "，",
                "!": ",",
                "；": "，",
                ";": ",",
                "：": "，",
                ":": ",",
                "、": "，",
                "？": "，",
                "?": ",",
            }
        )
        content = content.translate(internal_punct_map)
        content = content.strip()

        # Keep only the final period
        if len(content) > 1:
            last_ch = (
                "。"
                if content[-1] == "，"
                else ("." if content[-1] == "," else content[-1])
            )
            body = content[:-1].replace("。", "，")
            content = body + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    # Merge consecutive same speakers
    merged_lines = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]

    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]

    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

    return "".join(merged_lines).replace("‘", "'").replace("’", "'")


def process_batch(
    batch_items,
    tokenizer,
    model,
    spt,
    device,
    system_prompt,
    start_idx,
    use_normalize=False,
    silence_duration=0,
):
    """Process a batch of data items and generate audio, return audio data and metadata"""
    try:
        # Prepare batch data
        batch_size = len(batch_items)
        texts = []
        prompts = [system_prompt] * batch_size
        prompt_audios = []
        actual_texts_data = []  # Store actual text data used

        print(f"Processing {batch_size} samples starting from index {start_idx}...")

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
            final_text = full_text.replace("[S1]", "<speaker1>").replace(
                "[S2]", "<speaker2>"
            )
            texts.append(final_text)

            # Save actual text information used
            actual_texts_data.append(
                {
                    "index": start_idx + i,
                    "original_text": original_full_text,
                    "normalized_text": (
                        normalize_text(original_full_text) if use_normalize else None
                    ),
                    "final_text": final_text,
                    "use_normalize": use_normalize,
                }
            )

            # Get reference audio
            prompt_audios.append(processed_item["prompt_audio"])

        # Process inputs
        input_ids_list = []
        for i, (text, prompt, audio_path) in enumerate(
            zip(texts, prompts, prompt_audios)
        ):
            # Load audio data here
            audio_data = load_audio_data(audio_path) if audio_path else None
            inputs = process_inputs(
                tokenizer, spt, prompt, text, device, silence_duration, audio_data
            )
            inputs = shifting_inputs(inputs, tokenizer)
            input_ids_list.append(inputs)

        # Pad batch inputs
        input_ids, attention_mask = rpadding(input_ids_list, MAX_CHANNELS, tokenizer)

        # Batch generation
        print(f"Starting batch audio generation...")
        start = input_ids.shape[1] - MAX_CHANNELS + 1

        # Move inputs to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate model outputs
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        print(f"Original outputs shape: {outputs.shape}")
        print(f"Start value: {start}")
        print(f"Shape after slicing: {outputs[:, start:].shape}")
        print(f"MAX_CHANNELS: {MAX_CHANNELS}")
        print(f"Calculated seq_len: {outputs.shape[1] - MAX_CHANNELS + 1}")
        # Process outputs
        outputs = outputs[:, start:]
        seq_len = outputs.shape[1] - MAX_CHANNELS + 1
        speech_ids = torch.full((outputs.shape[0], seq_len, MAX_CHANNELS), 0).to(device)

        # Adjust output format
        for j in range(MAX_CHANNELS):
            speech_ids[..., j] = outputs[:, j : seq_len + j, j]
            if j == 0:
                speech_ids[..., j] = speech_ids[..., j] - 151665

        # Find valid positions for each sample
        li = find_max_valid_positions(speech_ids)

        # Store audio result data
        audio_results = []

        # Process batch sample results individually
        for i in range(batch_size):
            try:
                # Extract valid speech tokens
                end_idx = li[i] + 1
                if end_idx <= 0:
                    print(f"Sample {start_idx + i} has no valid speech tokens")
                    audio_results.append(None)
                    continue

                this_speech_id = speech_ids[i, :end_idx]
                print(
                    f"Speech token shape for sample {start_idx + i}: {this_speech_id.shape}"
                )

                # Decode generated audio
                with torch.no_grad():
                    codes_list = [
                        this_speech_id.permute(1, 0)
                    ]  # Convert to SPT expected format
                    decode_result = spt.decode(codes_list, overlap_seconds=10)
                    audio_result = decode_result["syn_wav_list"][0].cpu().detach()

                    if audio_result.ndim == 1:  # If 1D [samples]
                        audio_result = audio_result.unsqueeze(
                            0
                        )  # Convert to 2D [1, samples]

                # Save audio data instead of file path
                audio_results.append(
                    {
                        "audio_data": audio_result,
                        "sample_rate": spt.output_sample_rate,
                        "index": start_idx + i,
                    }
                )
                print(f"Audio generation completed: sample {start_idx + i}")

            except Exception as e:
                print(f"Error processing sample {start_idx + i}: {str(e)}, skipping...")
                import traceback

                traceback.print_exc()
                audio_results.append(None)

        # Clean up GPU memory
        torch.cuda.empty_cache()

        # Return text data and audio data
        return actual_texts_data, audio_results

    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        raise
