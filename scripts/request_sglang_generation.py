from __future__ import annotations

from pathlib import Path

import pybase64
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ASSET_DIR = REPO_ROOT / "asset"
OUTPUT_PATH = REPO_ROOT / "outputs" / "output.wav"
SERVER_URL = "http://localhost:30000/generate"

PROMPT_AUDIO_SPEAKER1 = ASSET_DIR / "reference_02_s1.wav"
PROMPT_AUDIO_SPEAKER2 = ASSET_DIR / "reference_02_s2.wav"
PROMPT_TEXT_SPEAKER1 = "[S1] In short, we embarked on a mission to make America great again for all Americans."
PROMPT_TEXT_SPEAKER2 = "[S2] NVIDIA reinvented computing for the first time after 60 years. In fact, Erwin at IBM knows quite well that the computer has largely been the same since the 60s."

TEXT_TO_GENERATE = """
[S1] Listen, let's talk business. China. I'm hearing things.
People are saying they're catching up. Fast. What's the real scoop?
Their AI—is it a threat?
[S2] Well, the pace of innovation there is extraordinary, honestly.
They have the researchers, and they have the drive.
[S1] Extraordinary? I don't like that. I want us to be extraordinary.
Are they winning?
[S2] I wouldn't say winning, but their progress is very promising.
They are building massive clusters. They're very determined.
[S1] Promising. There it is. I hate that word.
When China is promising, it means we're losing.
It's a disaster, Jensen. A total disaster.
""".strip()


def encode_audio_as_data_uri(audio_path: Path) -> str:
    audio_bytes = audio_path.read_bytes()
    b64_str = pybase64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{b64_str}"


def build_payload() -> dict[str, object]:
    return {
        "text": f"{PROMPT_TEXT_SPEAKER1} {PROMPT_TEXT_SPEAKER2} {TEXT_TO_GENERATE}",
        "audio_data": [
            encode_audio_as_data_uri(PROMPT_AUDIO_SPEAKER1),
            encode_audio_as_data_uri(PROMPT_AUDIO_SPEAKER2),
        ],
        "sampling_params": {
            "max_new_tokens": 2048,
            "temperature": 1.1,
            "top_p": 0.9,
            "top_k": 50,
        },
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    response = requests.post(
        SERVER_URL,
        json=build_payload(),
        headers={
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()

    content = response.json()
    audio_data = pybase64.b64decode(content["text"])
    OUTPUT_PATH.write_bytes(audio_data)
    print(f"Saved generated audio to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
