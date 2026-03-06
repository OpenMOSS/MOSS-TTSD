import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import aiohttp
import pybase64
import torch
import torchaudio

from generation_utils import normalize_text

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
}


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


def absolutize_prompt_audio(prompt_audio):
    """Convert any string paths within prompt_audio into absolute paths."""
    if prompt_audio is None:
        return None
    if isinstance(prompt_audio, str):
        return os.path.abspath(prompt_audio)
    if isinstance(prompt_audio, dict):
        return {k: absolutize_prompt_audio(v) for k, v in prompt_audio.items()}
    return prompt_audio


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


def get_file_path(output_path: str) -> str:
    """Validate and normalize an output path and return a final file path.

    Behavior:
    - If the input clearly denotes a directory (existing OR ends with a path separator), ensure it exists and append 'output.wav'.
    - If the path does NOT have a filename extension (no dot in the final segment) and does not already exist as a file, treat it as a directory and append 'output.wav'.
    - Otherwise treat it as a file path; ensure its parent directory exists.
    - Performs basic cross‑platform path format validation.

    Validation rules (kept simple, cross-platform safe):
    - Reject empty string.
    - Reject path segments containing Windows reserved characters: < > : " | ? *
      (Path separators '/' '\\' are allowed as separators, not inside a segment.)
    - Reject NUL character ("\0").
    - Reject Windows reserved device names (CON, PRN, AUX, NUL, COM1..COM9, LPT1..LPT9) as a bare segment (optionally followed by an extension) on Windows.

    Args:
        output_path: Raw user provided path (file or directory).

    Returns:
        A string path pointing to the final .wav output file.

    Raises:
        ValueError: If the provided path is not a valid format.
    """
    # Basic empty check
    if not output_path or not output_path.strip():
        raise ValueError("Output path cannot be empty.")

    # Normalize any surrounding quotes or whitespace the user might include
    output_path = output_path.strip().strip('"').strip("'")

    # Fast NUL char check (invalid on all mainstream OSes)
    if "\0" in output_path:
        raise ValueError("Output path contains NUL character, which is invalid.")

    # Determine platform (Windows vs POSIX) – but we apply a safe superset of restrictions
    is_windows = os.name == "nt"

    # Define invalid characters for individual path segments (exclude separators)
    invalid_chars = set('<>:"|?*')  # union of typical Windows-invalid chars

    # Extensions that we treat as explicit file outputs when the path does not yet exist
    known_file_suffixes = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".wma",
        ".aiff",
    }

    # Reserved device names on Windows (case-insensitive)
    reserved_windows_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }

    # We'll inspect each segment (Path.parts includes root/drive separately)
    # Use PurePath via Path for consistent splitting without touching filesystem yet.
    raw_parts = list(Path(output_path).parts)

    # Skip root or drive part when validating (e.g. 'C:\\' or '/')
    def is_drive_or_root(seg: str) -> bool:
        if seg in (os.sep, "/"):
            return True
        # Windows drive like 'C:\\' or 'C:'
        if len(seg) == 2 and seg[1] == ":":
            return True
        return False

    for seg in raw_parts:
        if is_drive_or_root(seg):
            continue
        # Remove trailing separator style artifacts
        seg_clean = seg.rstrip("/\\")
        if not seg_clean:
            continue
        # Check invalid chars
        if any(ch in invalid_chars for ch in seg_clean):
            raise ValueError(
                f"Invalid character found in path segment '{seg_clean}'. Forbidden characters: < > : \" | ? *"
            )
        # Windows reserved names check (case-insensitive, consider base name before dot)
        if is_windows:
            base = seg_clean.split(".")[0].upper()
            if base in reserved_windows_names:
                raise ValueError(
                    f"Segment '{seg_clean}' resolves to reserved Windows device name '{base}'."
                )

    # Interpret directory intent: existing directory OR explicit trailing separator
    # Use Path without resolving symlinks
    path_obj = Path(output_path)
    treat_as_dir = False
    if path_obj.exists() and path_obj.is_dir():
        treat_as_dir = True
    elif output_path.endswith(("/", "\\")):
        # Trailing separator strongly suggests directory even if it does not exist yet
        treat_as_dir = True

    # Additional rules for non-existing paths:
    # * No suffix => directory intent
    # * Unrecognized suffix (e.g., version numbers like .5) => directory intent
    if not treat_as_dir and not path_obj.exists():
        suffix_lower = path_obj.suffix.lower()
        if suffix_lower == "" or suffix_lower not in known_file_suffixes:
            treat_as_dir = True

    if treat_as_dir:
        # Ensure directory exists then append default filename
        path_obj.mkdir(parents=True, exist_ok=True)
        final_path = path_obj / "output.wav"
    else:
        # Treat as file path: ensure parent exists
        parent = path_obj.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        final_path = path_obj

    return str(final_path)


async def send_generate_request(session, url, payload, output_path, idx):
    """Asynchronously generate a single audio file."""
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                content = await response.json()
                if output_path is not None:
                    with open(output_path, "wb") as f:
                        f.write(pybase64.b64decode(content["text"]))
                    print(f"Audio saved to {output_path}")
                meta_info = content.get("meta_info", {})
                print(f"Prompt Tokens: {meta_info.get('prompt_tokens', 'N/A')}")
                print(f"Completion Tokens: {meta_info.get('completion_tokens', 'N/A')}")
                print(f"E2E Latency: {meta_info.get('e2e_latency', 'N/A')}")
                return True, meta_info
            else:
                error_text = await response.text()
                print(f"Error for item {idx}: {response.status}")
                print(error_text)
                return False, None
    except Exception as e:
        print(f"Exception for item {idx}: {e}")
        return False, None


def generate_audio(
    url: str,
    host: str,
    port: str,
    jsonl: str,
    output_dir: str,
    use_normalize: bool,
    silence_duration: float,
):
    """Wrapper function that calls the async implementation."""
    asyncio.run(
        generate_audio_async(
            url,
            host,
            port,
            jsonl,
            output_dir,
            use_normalize,
            silence_duration,
        )
    )


def build_generate_url(url: str, host: str, port: str) -> str:
    """Build a normalized generate endpoint URL from user input."""

    raw_url = (url or "").strip()
    raw_host = (host or "").strip()
    raw_port = (port or "").strip()

    if raw_url:
        base_url = raw_url
    else:
        if not raw_host:
            raise ValueError("Either url or host must be provided.")
        if "://" in raw_host or raw_host.startswith("//"):
            base_url = raw_host
        else:
            base_url = raw_host if not raw_port else f"{raw_host}:{raw_port}"

    if base_url.startswith("//"):
        base_url = "http:" + base_url
    elif "://" not in base_url:
        base_url = "http://" + base_url

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid server URL: {base_url}")

    username = parsed.username or ""
    password = parsed.password or ""
    hostname = parsed.hostname or ""
    resolved_port = parsed.port

    if not raw_url and raw_port and resolved_port is None:
        try:
            resolved_port = int(raw_port)
        except ValueError as exc:
            raise ValueError(f"Invalid port: {raw_port}") from exc

    if not hostname:
        raise ValueError(f"Invalid server URL: {base_url}")

    host_display = hostname
    if ":" in hostname and not hostname.startswith("["):
        host_display = f"[{hostname}]"

    if username:
        auth = username if not password else f"{username}:{password}"
        netloc = f"{auth}@{host_display}"
    else:
        netloc = host_display

    if resolved_port is not None:
        netloc = f"{netloc}:{resolved_port}"

    path = parsed.path.rstrip("/")
    if not path:
        path = "/generate"
    elif not path.endswith("/generate"):
        path = f"{path}/generate"

    return urlunparse((parsed.scheme, netloc, path, parsed.params, parsed.query, ""))


async def generate_audio_async(
    url: str,
    host: str,
    port: str,
    jsonl: str,
    output_dir: str,
    use_normalize: bool,
    silence_duration: float,
):
    try:
        url = build_generate_url(url, host, port)
    except ValueError as e:
        print(f"Failed to prepare request URL: {e}")
        return

    try:
        output_dir = Path(get_file_path(output_dir).removesuffix("output.wav"))
        output_dir = output_dir.resolve()
    except (ValueError, OSError) as e:
        print(f"Failed to prepare output directory: {e}")
        return

    # Load the items from the JSONL file
    try:
        with open(jsonl, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f.readlines()]
        print(f"Loaded {len(items)} items from {jsonl}")
    except FileNotFoundError:
        print(f"Error: JSONL file '{jsonl}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL file: {e}")
        return

    # Create an async HTTP session
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []

        for idx, item in enumerate(items):
            processed_item = process_jsonl_item(item)

            text = str(processed_item["text"])
            prompt_text = str(processed_item.get("prompt_text") or "")

            if use_normalize:
                text = normalize_text(text)
                if prompt_text:
                    prompt_text = normalize_text(prompt_text)

            wav_tensor = load_audio_data(processed_item["prompt_audio"])

            prompt_audio_base64 = None
            if wav_tensor is not None:
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp:
                        torchaudio.save(tmp.name, wav_tensor, sample_rate=16000)
                        tmp.flush()
                        tmp.seek(0)
                        wav_bytes = tmp.read()
                    prompt_audio_base64 = pybase64.b64encode(wav_bytes).decode("utf-8")
                except Exception as e:
                    print(f"Failed to convert wav tensor to base64: {e}")
                    prompt_audio_base64 = None

            if prompt_audio_base64:
                payload = {
                    "text": prompt_text + text,
                    "audio_data": f"data:audio/wav;base64,{prompt_audio_base64}",
                    "sampling_params": sampling_params,
                }
            else:
                payload = {
                    "text": text,
                    "sampling_params": sampling_params,
                }

            output_path = output_dir / f"output_{idx}.wav"

            task = send_generate_request(session, url, payload, output_path, idx)
            tasks.append(task)

        # Execute all tasks concurrently
        print(f"Starting concurrent generation of {len(tasks)} audio files...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful
        print(f"Generation completed: {successful} successful, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS inference with MOSS-TTSD model")
    parser.add_argument("--url", default=None, type=str)
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default="30000", type=str)
    parser.add_argument(
        "--jsonl",
        default="examples/examples.jsonl",
        help="Path to JSONL file (default: examples/examples.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory for generated audio files (default: outputs)",
    )
    parser.add_argument(
        "--use_normalize",
        action="store_true",
        default=False,
        help="Whether to use text normalization (default: False)",
    )
    parser.add_argument("--max_new_tokens", default=20000, type=int)
    parser.add_argument(
        "--silence_duration",
        type=float,
        default=0,
        help="Silence duration between speech prompt and generated speech, which can be used to avoid noise problem at the beginning of generated audio",
    )

    args = parser.parse_args()

    sampling_params["max_new_tokens"] = args.max_new_tokens
    generate_audio(
        args.url,
        args.host,
        args.port,
        args.jsonl,
        args.output_dir,
        args.use_normalize,
        args.silence_duration,
    )
