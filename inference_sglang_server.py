import argparse
import asyncio
import io
import json
import os
from pathlib import Path

import aiohttp
import pybase64
import torchaudio

from generation_utils import load_audio_data, normalize_text, process_jsonl_item

sampling_params = {
    "repetition_penalty": 1.0,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
}


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

    # Additional rule: a non-existing path with no suffix (no extension) is treated as a directory
    if not treat_as_dir:
        if (not path_obj.exists()) and path_obj.suffix == "":
            # No extension and path does not exist => interpret as directory intent
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
                content = await response.read()
                with open(output_path, "wb") as f:
                    f.write(content)
                print(f"Audio saved to {output_path}")
                print(f"Sample Rate: {response.headers.get('sample_rate', 'N/A')}")
                print(f"Prompt Tokens: {response.headers.get('prompt_tokens', 'N/A')}")
                print(
                    f"Completion Tokens: {response.headers.get('completion_tokens', 'N/A')}"
                )
                return True
            else:
                error_text = await response.text()
                print(f"Error for item {idx}: {response.status}")
                print(error_text)
                return False
    except Exception as e:
        print(f"Exception for item {idx}: {e}")
        return False


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


async def generate_audio_async(
    url: str,
    host: str,
    port: str,
    jsonl: str,
    output_dir: str,
    use_normalize: bool,
    silence_duration: float,
):
    if url is None:
        url = "http://" + host + ":" + port + "/generate_audio"
    else:
        url = url.removesuffix("/") + "/generate_audio"

    try:
        output_dir = Path(get_file_path(output_dir).removesuffix("output.wav"))
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

            text = processed_item["text"]
            prompt_text = processed_item["prompt_text"]

            wav_tensor = load_audio_data(processed_item["prompt_audio"])

            prompt_audio_base64 = None
            if wav_tensor is not None:
                try:
                    buf = io.BytesIO()
                    torchaudio.save(buf, wav_tensor, sample_rate=16000, format="wav")
                    wav_bytes = buf.getvalue()
                    prompt_audio_base64 = pybase64.b64encode(wav_bytes).decode("utf-8")
                except Exception as e:
                    print(f"Failed to convert wav tensor to base64: {e}")
                    prompt_audio_base64 = None

            if prompt_audio_base64:
                payload = {
                    "text": text,
                    "prompt_text": prompt_text,
                    "prompt_audio": prompt_audio_base64,
                    "silence_duration": silence_duration,
                    "use_normalize": use_normalize,
                    "sampling_params": sampling_params,
                }
            else:
                payload = {
                    "text": text,
                    "use_normalize": use_normalize,
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
