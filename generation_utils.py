import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio


def normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

    remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        m = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = m.groups() if m else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        content = content.replace("——", "，")
        content = content.replace("……", "，")
        content = content.replace("...", "，")
        content = content.replace("⸺", "，")
        content = content.replace("―", "，")
        content = content.replace("—", "，")
        content = content.replace("…", "，")

        internal_punct_map = str.maketrans(
            {"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}
        )
        content = content.translate(internal_punct_map)
        content = content.strip()

        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

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


def streaming_jsonl_reader(
    jsonl_path: str,
    rank: int = 0,
    world_size: int = 1,
    skip_invalid_json: bool = False,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if world_size < 1:
        raise ValueError("`world_size` must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError("`rank` must satisfy 0 <= rank < world_size.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if (line_no - 1) % world_size != rank:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid_json:
                    print(
                        f"[WARN] jsonl line {line_no} skipped: invalid json ({exc.msg})"
                    )
                    continue
                raise ValueError(
                    f"jsonl line {line_no}: invalid json ({exc.msg})"
                ) from exc
            yield line_no, sample


def _to_abs_path_str(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _abspath_record_paths(record: Dict[str, Any]) -> Dict[str, Any]:
    path_key_pattern = re.compile(
        r"^(output_audio|prompt_audio(?:_speaker\d+)?|.*_path)$"
    )
    for key, value in list(record.items()):
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        if path_key_pattern.fullmatch(str(key)):
            record[key] = _to_abs_path_str(value)
    return record


def _resolve_path(maybe_path: str, base_path: Optional[str]) -> str:
    if base_path is None:
        return maybe_path
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    return str(Path(base_path) / p)


def _make_output_record(raw_sample: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    record = dict(raw_sample)
    base_path = record.pop("base_path", None)
    record["id"] = sample_id

    for k, v in list(record.items()):
        if v is None:
            continue
        if re.fullmatch(r"prompt_audio(?:_speaker\d+)?", str(k)) and isinstance(v, str):
            record[k] = _resolve_path(str(v), base_path)
    return _abspath_record_paths(record)


def _write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text

    merged_parts: List[str] = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if m:
            tag, content = m.groups()
            if tag == current_tag:
                merged_parts.append(content)
            else:
                current_tag = tag
                merged_parts.append(f"{tag}{content}")
        else:
            merged_parts.append(seg)
    return "".join(merged_parts)


def _load_generation_config(model_path: str) -> Dict[str, Any]:
    generation_config_path = Path(model_path) / "generation_config.json"
    if not generation_config_path.is_file():
        return {}
    try:
        with open(generation_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read {generation_config_path}: {exc}")
        return {}
    if not isinstance(cfg, dict):
        return {}
    return cfg


def resolve_sampling_args(args: argparse.Namespace) -> None:
    fallback: Dict[str, Any] = {
        "max_new_tokens": 8192,
        "temperature": 1.1,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    generation_cfg = _load_generation_config(args.model_path)

    for key, default_value in fallback.items():
        cli_value = getattr(args, key)
        if cli_value is not None:
            continue
        if key in generation_cfg:
            try:
                if key in {"top_k", "max_new_tokens"}:
                    setattr(args, key, int(generation_cfg[key]))
                else:
                    setattr(args, key, float(generation_cfg[key]))
                continue
            except (TypeError, ValueError):
                pass
        setattr(args, key, default_value)


def _load_mono_wav(wav_path: str) -> Tuple[torch.Tensor, int]:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if wav.ndim != 2:
        raise ValueError(
            f"Expect wav tensor rank=2, got shape={tuple(wav.shape)} from {wav_path}"
        )
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.contiguous(), int(sr)


def _maybe_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(
        waveform=wav,
        orig_freq=orig_sr,
        new_freq=target_sr,
    )


def _preprocess_prompt_wavs(
    loaded_wavs: List[Tuple[torch.Tensor, int]],
    target_sr: int,
    sample_rate_normalize_enabled: bool,
) -> List[torch.Tensor]:
    if sample_rate_normalize_enabled:
        min_sr = min(sr for _, sr in loaded_wavs)
    else:
        min_sr = None

    wav_list: List[torch.Tensor] = []
    for wav, sr in loaded_wavs:
        if sample_rate_normalize_enabled:
            wav = _maybe_resample(wav, sr, int(min_sr))
            sr = int(min_sr)
        wav = _maybe_resample(wav, sr, target_sr)
        wav_list.append(wav)
    return wav_list


def _collect_speaker_fields(
    sample: Dict[str, Any],
    base_path: Optional[str],
) -> Tuple[Dict[int, str], Dict[int, str], List[int]]:
    audio_map: Dict[int, str] = {}
    text_map: Dict[int, str] = {}

    for k, v in sample.items():
        if v is None:
            continue
        key = str(k)
        value = str(v).strip()
        if not value:
            continue

        m_audio = re.fullmatch(r"prompt_audio_speaker(\d+)", key)
        if m_audio:
            speaker_id = int(m_audio.group(1))
            audio_map[speaker_id] = _resolve_path(value, base_path)
            continue

        m_text = re.fullmatch(r"prompt_text_speaker(\d+)", key)
        if m_text:
            speaker_id = int(m_text.group(1))
            text_map[speaker_id] = value

    speaker_ids = sorted(set(audio_map.keys()) & set(text_map.keys()))
    return audio_map, text_map, speaker_ids


def _build_prefixed_text(
    text: str, text_map: Dict[int, str], speaker_ids: List[int]
) -> str:
    parts: List[str] = []
    for speaker_id in speaker_ids:
        cur_text = text_map[speaker_id]
        tag = f"[S{speaker_id}]"
        if not cur_text.lstrip().startswith(tag):
            cur_text = f"{tag}{cur_text}"
        parts.append(cur_text)
    return _merge_consecutive_speaker_tags("".join(parts) + text)


def _encode_concat_prompt_audio(
    processor: Any,
    audio_map: Dict[int, str],
    speaker_ids: List[int],
    target_sr: int,
    sample_rate_normalize_enabled: bool,
) -> torch.Tensor:
    loaded_wavs: List[Tuple[torch.Tensor, int]] = []
    for speaker_id in speaker_ids:
        loaded_wavs.append(_load_mono_wav(audio_map[speaker_id]))

    wav_list = _preprocess_prompt_wavs(
        loaded_wavs=loaded_wavs,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
    )

    wav = torch.cat(wav_list, dim=-1)
    return processor.encode_audios_from_wav([wav], sampling_rate=target_sr)[0]


def _encode_references(
    processor: Any,
    audio_map: Dict[int, str],
    speaker_ids: List[int],
    target_sr: int,
    sample_rate_normalize_enabled: bool,
) -> List[Optional[torch.Tensor]]:
    loaded_wavs: List[Tuple[torch.Tensor, int]] = []
    ordered_ids: List[int] = []
    for speaker_id in speaker_ids:
        loaded_wavs.append(_load_mono_wav(audio_map[speaker_id]))
        ordered_ids.append(speaker_id)

    wav_list = _preprocess_prompt_wavs(
        loaded_wavs=loaded_wavs,
        target_sr=target_sr,
        sample_rate_normalize_enabled=sample_rate_normalize_enabled,
    )

    encoded_list = processor.encode_audios_from_wav(wav_list, sampling_rate=target_sr)
    encoded_map = {
        speaker_id: tokens for speaker_id, tokens in zip(ordered_ids, encoded_list)
    }
    max_speaker_id = max(speaker_ids)
    return [encoded_map.get(speaker_id) for speaker_id in range(1, max_speaker_id + 1)]


def prepare_sample(
    line_no: int,
    raw_sample: Dict[str, Any],
    mode: str,
    processor: Any,
    target_sr: int,
    text_normalize_enabled: bool,
    sample_rate_normalize_enabled: bool,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    sample = dict(raw_sample)
    sample_id = f"{line_no:06d}"
    output_record = _make_output_record(raw_sample, sample_id)

    if sample.get("text") is None:
        raise ValueError(f"jsonl line {line_no}: missing `text`")
    text = str(sample["text"])
    if text_normalize_enabled:
        text = normalize_text(text)

    base_path = sample.get("base_path")
    audio_map, text_map, speaker_ids = _collect_speaker_fields(sample, base_path)

    if mode != "generation" and len(speaker_ids) == 0:
        raise ValueError(
            f"jsonl line {line_no}: mode={mode} requires at least one paired "
            "`prompt_audio_speakerN` + `prompt_text_speakerN`."
        )

    if mode == "generation":
        conversation = [processor.build_user_message(text=text)]
        return sample_id, output_record, conversation

    if mode in ("continuation", "voice_clone_and_continuation"):
        text = _build_prefixed_text(
            text=text, text_map=text_map, speaker_ids=speaker_ids
        )
        if text_normalize_enabled:
            text = normalize_text(text)

    if mode == "continuation":
        prompt_audio = _encode_concat_prompt_audio(
            processor=processor,
            audio_map=audio_map,
            speaker_ids=speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        )
        conversation = [
            processor.build_user_message(text=text),
            processor.build_assistant_message(audio_codes_list=[prompt_audio]),
        ]
        return sample_id, output_record, conversation

    if mode == "voice_clone":
        reference = _encode_references(
            processor=processor,
            audio_map=audio_map,
            speaker_ids=speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        )
        conversation = [processor.build_user_message(text=text, reference=reference)]
        return sample_id, output_record, conversation

    if mode == "voice_clone_and_continuation":
        reference = _encode_references(
            processor=processor,
            audio_map=audio_map,
            speaker_ids=speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        )
        prompt_audio = _encode_concat_prompt_audio(
            processor=processor,
            audio_map=audio_map,
            speaker_ids=speaker_ids,
            target_sr=target_sr,
            sample_rate_normalize_enabled=sample_rate_normalize_enabled,
        )
        conversation = [
            processor.build_user_message(text=text, reference=reference),
            processor.build_assistant_message(audio_codes_list=[prompt_audio]),
        ]
        return sample_id, output_record, conversation

    raise ValueError(f"Unexpected mode: {mode}")


def run_infer_batch(
    batch_data: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]],
    model: Any,
    processor: Any,
    mode: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    save_dir: Path,
    out_fp,
) -> None:
    sample_ids, output_records, conversations = zip(*batch_data)
    input_batch = processor(
        list(conversations),
        mode=(
            "continuation"
            if mode in ("continuation", "voice_clone_and_continuation")
            else "generation"
        ),
    )

    outputs = model.generate(
        input_ids=input_batch["input_ids"].to(device),
        attention_mask=input_batch["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
        audio_temperature=temperature,
        audio_top_p=top_p,
        audio_top_k=top_k,
        audio_repetition_penalty=repetition_penalty,
    )
    messages = processor.decode(outputs)

    sampling_rate = int(processor.model_config.sampling_rate)
    for sample_id, output_record, message in zip(sample_ids, output_records, messages):
        record = dict(output_record)

        if message is None or len(message.audio_codes_list) == 0:
            record["output_audio"] = None
            record["duration"] = 0.0
            record = _abspath_record_paths(record)
            _write_jsonl_line(out_fp, record)
            continue

        wav_segments: List[torch.Tensor] = []
        for wav in message.audio_codes_list:
            if not isinstance(wav, torch.Tensor):
                continue
            wav_segments.append(
                wav.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
            )

        if len(wav_segments) == 0:
            record["output_audio"] = None
            record["duration"] = 0.0
            record = _abspath_record_paths(record)
            _write_jsonl_line(out_fp, record)
            continue

        merged_wav = torch.cat(wav_segments, dim=0)
        save_path = (save_dir / f"{sample_id}.wav").resolve()
        sf.write(str(save_path), merged_wav.numpy(), sampling_rate)

        record["output_audio"] = str(save_path)
        record["duration"] = float(merged_wav.shape[-1] / sampling_rate)
        record = _abspath_record_paths(record)
        _write_jsonl_line(out_fp, record)


def _record_sort_key(record: Dict[str, Any]) -> Tuple[int, str]:
    raw_id = record.get("id", "")
    try:
        numeric_id = int(str(raw_id))
    except (TypeError, ValueError):
        numeric_id = 10**18
    return numeric_id, str(raw_id)


def merge_rank_jsonl_files(
    output_jsonl_path: Path,
    rank_jsonl_paths: List[Path],
) -> None:
    import heapq

    existing_paths = [p for p in rank_jsonl_paths if p.is_file()]
    if len(existing_paths) == 0:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl_path, "w", encoding="utf-8"):
            pass
        return

    fps = [open(p, "r", encoding="utf-8") for p in existing_paths]
    try:
        heap: List[Tuple[Tuple[int, str], int, Dict[str, Any]]] = []
        for idx, fp in enumerate(fps):
            line = fp.readline()
            if not line:
                continue
            obj = json.loads(line)
            heapq.heappush(heap, (_record_sort_key(obj), idx, obj))

        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl_path, "w", encoding="utf-8") as out_fp:
            while heap:
                _, file_idx, obj = heapq.heappop(heap)
                _write_jsonl_line(out_fp, _abspath_record_paths(obj))

                line = fps[file_idx].readline()
                if not line:
                    continue
                next_obj = json.loads(line)
                heapq.heappush(heap, (_record_sort_key(next_obj), file_idx, next_obj))
    finally:
        for fp in fps:
            fp.close()
