import os
import re

import numpy as np
import torch
import torchaudio

MAX_CHANNELS = 8


def pad_or_truncate_to_seconds(
    wav: torch.Tensor, target_seconds: float, sr: int
) -> torch.Tensor:
    """Pad or truncate a mono waveform to target length in seconds.

    Args:
        wav: (1, T) or (T,) tensor
        target_seconds: target duration in seconds
        sr: sample rate
    Returns:
        (1, T_target) tensor
    """
    if wav.dim() == 2 and wav.shape[0] == 1:
        wav_1d = wav.squeeze(0)
    else:
        wav_1d = wav.reshape(-1)
    target_len = int(round(target_seconds * sr))
    cur_len = wav_1d.shape[-1]
    if cur_len == target_len:
        out = wav_1d
    elif cur_len > target_len:
        out = wav_1d[:target_len]
    else:
        pad_len = target_len - cur_len
        out = torch.cat(
            [wav_1d, torch.zeros(pad_len, dtype=wav_1d.dtype, device=wav_1d.device)],
            dim=-1,
        )
    return out.unsqueeze(0)


def crossfade_concat(
    segments: list, sample_rate: int, crossfade_seconds: float = 0.1
) -> torch.Tensor:
    """Concatenate segments with linear crossfade.

    Args:
        segments: list of (1, T) tensors
        sample_rate: sampling rate
        crossfade_seconds: overlap time for crossfade
    Returns:
        (1, T_total) tensor
    """
    if len(segments) == 0:
        return torch.zeros(1, 0)
    if len(segments) == 1:
        return segments[0]
    out = segments[0]
    cf_len_target = int(round(crossfade_seconds * sample_rate))
    for k in range(1, len(segments)):
        nxt = segments[k]
        if cf_len_target <= 0:
            out = torch.cat([out, nxt], dim=-1)
            continue
        cf_len = min(cf_len_target, out.shape[-1], nxt.shape[-1])
        if cf_len <= 0:
            out = torch.cat([out, nxt], dim=-1)
            continue
        fade_out = torch.linspace(
            1.0, 0.0, steps=cf_len, dtype=out.dtype, device=out.device
        )
        fade_in = torch.linspace(
            0.0, 1.0, steps=cf_len, dtype=nxt.dtype, device=nxt.device
        )
        overlap = out[0, -cf_len:] * fade_out + nxt[0, :cf_len] * fade_in
        out = torch.cat(
            [out[:, :-cf_len], overlap.unsqueeze(0), nxt[:, cf_len:]], dim=-1
        )
    return out


def load_model(
    model_path,
    spt_config_path,
    spt_checkpoint_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
):
    from modeling_asteroid import AsteroidTTSInstruct
    from transformers import AutoTokenizer
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
    """Parse a JSONL item enforcing prompt requirement.

    Only supports Format 1 (separate speaker refs) and Format 2 (shared ref),
    consistent with the updated README. If `base_path` is missing/empty, any
    string paths must be absolute. Text-only input is not supported and will raise.
    """
    base_path = item.get("base_path", "") or ""
    text = item.get("text", "")

    def _resolve_path(p: str) -> str:
        if not isinstance(p, str) or not p:
            return p
        if base_path:
            return os.path.join(base_path, p)
        # base_path missing: require absolute path
        if not os.path.isabs(p):
            raise ValueError(
                "When base_path is omitted, audio paths must be absolute. Got: " + p
            )
        return p

    # Try Format 2 first: shared audio reference
    prompt_audio = None
    prompt_text = ""
    if "prompt_audio" in item:
        prompt_audio_val = item.get("prompt_audio")
        if not prompt_audio_val:
            raise ValueError("Format 2 requires non-empty 'prompt_audio'.")
        if isinstance(prompt_audio_val, str):
            prompt_audio = _resolve_path(prompt_audio_val)
        else:
            # allow tuple form for backward-compatibility
            prompt_audio = prompt_audio_val
        prompt_text = item.get("prompt_text", "")
        return {"text": text, "prompt_text": prompt_text, "prompt_audio": prompt_audio}

    # Try Format 1: separate speaker references
    s1 = item.get("prompt_audio_speaker1", "")
    s2 = item.get("prompt_audio_speaker2", "")
    has_s1 = (isinstance(s1, str) and s1) or isinstance(s1, tuple)
    has_s2 = (isinstance(s2, str) and s2) or isinstance(s2, tuple)

    if has_s1 and has_s2:
        if isinstance(s1, str) and s1:
            s1_resolved = _resolve_path(s1)
        else:
            s1_resolved = s1
        if isinstance(s2, str) and s2:
            s2_resolved = _resolve_path(s2)
        else:
            s2_resolved = s2
        # Build merged prompt audio dict
        prompt_audio = {"speaker1": s1_resolved, "speaker2": s2_resolved}
        # Merge texts
        pt1 = item.get("prompt_text_speaker1", "")
        pt2 = item.get("prompt_text_speaker2", "")
        merged = ""
        if pt1:
            merged += f"[S1]{pt1}"
        if pt2:
            merged += f"[S2]{pt2}"
        prompt_text = merged.strip()
        return {"text": text, "prompt_text": prompt_text, "prompt_audio": prompt_audio}

    # Otherwise, no supported prompt found → reject (text-only unsupported)
    raise ValueError(
        "Input must include prompt (Format 1 or 2). Text-only is not supported."
    )


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
    2. Preserve bracketed segments like [] () <> even when they are not speaker tags.
    3. Remove decorative symbols: 【】《》（）『』「」～~-_.
    4. Internal punctuation ；：、 → ，；keep ？！?.
    5. Multiple 。 keep only the last one, others → ，。
    6. Replace consecutive "哈" (>=2) with "(笑)".
    7. Auto-recognize [S1] / [S2] … tags; if missing, treat as whole segment.
    8. Merge adjacent identical speaker tags.
    """
    # Replace [1], [2] etc. format with [S1], [S2] etc. format
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

    # Remove decorative characters
    remove_chars = "【】《》（）『』「」" '"-_“”～~'

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
        content = re.sub(r"哈{2,}", "[笑]", content)

        # Handle English laughter (e.g., "haha", "ha ha")
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        # First handle multi-character punctuation marks
        content = content.replace("——", "，")
        content = content.replace("……", "，")

        # Handle single-character internal punctuation marks
        internal_punct_map = str.maketrans(
            {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
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

                # Prompt-Augmented Decode (rvq8-style); fall back to original decode if no prompt
                prompt_audio = prompt_audios[i]
                if prompt_audio is None:
                    # Fallback to original decode
                    with torch.no_grad():
                        codes_list = [this_speech_id.permute(1, 0)]
                        decode_result = spt.decode(codes_list, overlap_seconds=10)
                        audio_out = decode_result["syn_wav_list"][0].cpu().detach()
                        if audio_out.ndim == 1:
                            audio_out = audio_out.unsqueeze(0)
                    audio_results.append(
                        {
                            "audio_data": audio_out,
                            "sample_rate": spt.output_sample_rate,
                            "index": start_idx + i,
                        }
                    )
                    print(f"Audio generation completed (orig): sample {start_idx + i}")
                else:
                    # 1) Load prompt at SPT input sr and force to 20s
                    ref_sr_in = (
                        getattr(spt, "input_sample_rate", None)
                        or getattr(spt, "sampling_rate", None)
                        or 24000
                    )
                    ref_wav = load_audio_data(
                        prompt_audio, target_sample_rate=ref_sr_in
                    )
                    if ref_wav is None:
                        # If ref missing, use original decode
                        with torch.no_grad():
                            codes_list = [this_speech_id.permute(1, 0)]
                            decode_result = spt.decode(codes_list, overlap_seconds=10)
                            audio_out = decode_result["syn_wav_list"][0].cpu().detach()
                            if audio_out.ndim == 1:
                                audio_out = audio_out.unsqueeze(0)
                        audio_results.append(
                            {
                                "audio_data": audio_out,
                                "sample_rate": spt.output_sample_rate,
                                "index": start_idx + i,
                            }
                        )
                        print(
                            f"Audio generation completed (orig no-ref): sample {start_idx + i}"
                        )
                    else:
                        # Encode 20s reference to tokens
                        ref_wav_20s = pad_or_truncate_to_seconds(
                            ref_wav, 20.0, ref_sr_in
                        ).to(device)
                        with torch.no_grad():
                            enc = spt.encode([ref_wav_20s.squeeze(0)])
                            ref_codes = (
                                enc["codes_list"][0].to(device).long()
                            )  # (nq, T_ref)

                        # Prepare token-to-sample mapping and windowing params
                        out_sr = (
                            getattr(spt, "output_sample_rate", None)
                            or getattr(spt, "sample_rate", None)
                            or 24000
                        )
                        tokens_per_second = float(ref_sr_in) / float(
                            spt.encoder_downsample_rate
                        )
                        tokens_per_chunk = int(round(10.0 * tokens_per_second))
                        stride_tokens = 85
                        keep_tokens = 85
                        left_ctx_tokens = 20
                        total_tokens = this_speech_id.shape[0]
                        samples_per_token = int(round(out_sr / tokens_per_second))
                        crossfade_seconds = 0.1
                        crossfade_samples = int(round(crossfade_seconds * out_sr))

                        kept_segments = []
                        chunk_idx = 0
                        while True:
                            st_tok = chunk_idx * stride_tokens
                            if st_tok >= total_tokens:
                                break
                            ed_tok = min(st_tok + tokens_per_chunk, total_tokens)
                            gen_chunk = this_speech_id[st_tok:ed_tok]  # (len, C)
                            if gen_chunk.shape[0] == 0:
                                break

                            # Concatenate reference tokens with current window tokens
                            combined_codes = torch.cat(
                                [ref_codes, gen_chunk.permute(1, 0).long()], dim=1
                            ).to(
                                device
                            )  # (nq, T_ref + T_chunk)
                            codes_lengths = torch.tensor(
                                [combined_codes.shape[-1]],
                                dtype=torch.long,
                                device=device,
                            )
                            combined_codes_batched = combined_codes.unsqueeze(
                                1
                            )  # (nq, 1, T)

                            with torch.no_grad():
                                detok = spt.inference_detokenize(
                                    combined_codes_batched, codes_lengths
                                )
                                y = detok["y"][0, 0]  # (T_samples)

                            # Remove 20s reference portion (in samples)
                            ref_samples = int(round(20.0 * out_sr))
                            if y.shape[-1] <= ref_samples:
                                chunk_idx += 1
                                continue
                            chunk_y = y[ref_samples:]

                            # Determine kept region within current window
                            window_len = gen_chunk.shape[0]
                            remains = total_tokens - st_tok
                            is_first = chunk_idx == 0
                            is_last = ed_tok >= total_tokens

                            if is_first:
                                keep_start_tok = 0
                                keep_end_tok = min(
                                    keep_tokens + left_ctx_tokens, window_len
                                )
                            elif is_last and remains < 105:
                                keep_start_tok = (
                                    0 if is_first else min(left_ctx_tokens, window_len)
                                )
                                keep_end_tok = window_len
                            else:
                                keep_start_tok = min(left_ctx_tokens, window_len)
                                keep_end_tok = min(
                                    left_ctx_tokens + keep_tokens, window_len
                                )

                            keep_start_smps = keep_start_tok * samples_per_token
                            keep_end_smps = keep_end_tok * samples_per_token
                            left_margin = 0
                            right_margin = crossfade_samples if not is_last else 0
                            seg_start = max(0, keep_start_smps - left_margin)
                            seg_end = min(
                                chunk_y.shape[-1], keep_end_smps + right_margin
                            )
                            if seg_end > seg_start:
                                kept_segments.append(
                                    chunk_y[seg_start:seg_end]
                                    .detach()
                                    .cpu()
                                    .unsqueeze(0)
                                )

                            chunk_idx += 1

                        # Concatenate with crossfade; if empty, return tiny silence
                        if len(kept_segments) == 0:
                            audio_out = torch.zeros(1, int(0.01 * out_sr))
                        else:
                            audio_out = crossfade_concat(
                                kept_segments,
                                out_sr,
                                crossfade_seconds=crossfade_seconds,
                            )

                        audio_results.append(
                            {
                                "audio_data": audio_out,
                                "sample_rate": out_sr,
                                "index": start_idx + i,
                            }
                        )
                        print(
                            f"Audio generation completed (prompt-aug): sample {start_idx + i}"
                        )

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
