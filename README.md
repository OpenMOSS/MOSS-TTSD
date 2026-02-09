<div align="center">
    <h1>
    MOSS-TTSD: Text to Spoken Dialogue Generation
    </h1>
    <p>
    <img src="asset/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
    <p>
    </p>
    <a href="https://mosi.cn/models/moss-ttsd"><img src="https://img.shields.io/badge/Blog-Read%20More-green" alt="blog"></a>
    <a href="https://mosi.cn/models/moss-ttsd"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-orange" alt="paper"></a>
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20MOSS%20TTSD%20-v1.0-yellow" alt="MOSS-TTSD-v1.0"></a>
    <a href=""><img src="https://img.shields.io/badge/AI Stuidio-Coming%20Soon-blue" alt="AI Studio"></a>
    <a href="https://github.com/"><img src="https://img.shields.io/badge/Python-3.10+-orange" alt="version"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/PyTorch-2.0+-brightgreen" alt="python"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
    <br>

</div>


# MOSS-TTSD🪐

<!-- **MOSS-TTSD** is a long-form spoken dialogue generation model that enables highly expressive multi-party conversational speech synthesis across multiple languages. It supports continuous long-duration generation, flexible multi-speaker dialogue control, and state-of-the-art zero-shot voice cloning with only short reference audio. MOSS-TTSD is designed for real-world long-form content creation, including podcasts, audiobook, sports and esports commentary, dubbing, crosstalk, and entertainment scenarios. （about）-->


## Overview
 <p align="center">
    <img src="asset/ttsd.png" alt="alt text" width="330">
  </p>

MOSS-TTSD is the long-form dialogue specialist within our open-source [MOSS‑TTS Family](https://github.com/OpenMOSS/MOSS-TTS). While foundational models typically prioritize high-fidelity single-speaker synthesis, MOSS-TTSD is architected to bridge the gap between isolated audio samples and cohesive, continuous human interaction.
The model represents a paradigm shift from "text-to-speech" to "script-to-conversation." By prioritizing the flow and emotional nuances of multi-party engagement, MOSS-TTSD transforms static dialogue scripts into dynamic, expressive oral performances. It is designed to serve as a robust backbone for creators and developers who require a seamless transition between distinct speaker personas without sacrificing narrative continuity.
Whether it is capturing the spontaneous energy of a live talk show or the structured complexity of a multilingual drama, MOSS-TTSD provides the stability and expressive depth necessary for professional-grade, long-form content creation in an open-source framework.


## Highlights
- **From Monologue to Dialogue**: Unlike traditional TTS which optimizes for reading, MOSS-TTSD masters the rhythm of conversation. It supports 1 to 5 speakers with flexible control, handling natural turn-taking, overlapping speech patterns, and distinct persona maintenance.
- **Extreme Long-Context Modeling**: moving beyond short-sentence generation, the model is architected for stability over long durations, supporting up to 60 minutes of coherent audio in a single session with consistent identity.
- **Diverse Scenario Adaptation**: fine-tuned for high-variability scenarios including conversational media (AI Podcasts), dynamic commentary (Sports/Esports), and entertainment (Audiobooks, Dubbing, and Crosstalk).
- **Multilingual & Zero-Shot Capabilities**: features state-of-the-art zero-shot voice cloning requiring only short reference audio, with robust cross-lingual performance across major languages including Chinese, English, Japanese, and European languages.


## News 🚀
- **[2026-2-10]** MOSS-TTSD v1.0 is released! MOSS-TTSD v1.0 is officially released! This milestone version redefines long-form synthesis with 60-minute single-session context and support for multi-party interactions. It significantly expands multilingual capabilities and diverse usage scenarios.
 - **[2025-11-01]** MOSS-TTSD v0.7 is released! v0.7 significantly improves audio quality, voice cloning capability, and stability, adds support for 32 kHz high‑quality output, greatly extends single‑pass generation length (960s→1700s).
- **[2025-09-09]** We supported SGLang inference engine to accelerate model inference by up to **16x**.
- **[2025-08-25]** We released the 32khz version of XY-Tokenizer.
- **[2025-08-12]** We add support for streaming inference in MOSS-TTSD v0.5.
- **[2025-07-29]** We provide the SiliconFlow API interface and usage examples for MOSS-TTSD v0.5.
- **[2025-07-16]** We open-source the fine-tuning code for MOSS-TTSD v0.5, supporting full-parameter fine-tuning, LoRA fine-tuning, and multi-node training.
- **[2025-07-04]** MOSS-TTSD v0.5 is released! v0.5 has enhanced the accuracy of timbre switching, voice cloning capability, and model stability.
- **[2025-06-20]** MOSS-TTSD v0 is released! Moreover, we provide a podcast generation pipeline named Podever, which can automatically convert PDF, URL, or long text files into high-quality podcasts.

**Note:** You can still access the previous version, MOSS-TTSD v0.7, in the [legacy v0.7 folder](./legacy/v0.7/README.md).

## Installation

To run MOSS-TTSD, you need to install the required dependencies. You can use pip and conda to set up your environment.

### Using conda

```bash
conda create -n moss_ttsd python=3.12 -y && conda activate moss_ttsd
pip install -r requirements.txt
pip install flash-attn
```

## Usage

### Quick Start

MOSS-TTSD uses a **continuation** workflow: provide reference audio for each speaker, their transcripts as a prefix, and the dialogue text to generate. The model continues in each speaker's identity.

```python
import os
from pathlib import Path
import torch
import soundfile as sf
import torchaudio
from transformers import AutoModel, AutoProcessor

pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTSD-v1.0"
audio_tokenizer_name_or_path = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    codec_path=audio_tokenizer_name_or_path,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)
processor.audio_tokenizer.eval()

attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
# If flash_attention_2 is unavailable on your environment, set this to "sdpa".
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

# --- Inputs ---

prompt_audio_speaker1 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_02_s1.wav"
prompt_audio_speaker2 = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_02_s2.wav"
prompt_text_speaker1 = "[S1] In short, we embarked on a mission to make America great again for all Americans."
prompt_text_speaker2 = "[S2] NVIDIA reinvented computing for the first time after 60 years. In fact, Erwin at IBM knows quite well that the computer has largely been the same since the 60s."

text_to_generate = """
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

# --- Load & resample audio ---

target_sr = int(processor.model_config.sampling_rate)
audio1, sr1 = sf.read(prompt_audio_speaker1, dtype="float32", always_2d=True)
audio2, sr2 = sf.read(prompt_audio_speaker2, dtype="float32", always_2d=True)
wav1 = torch.from_numpy(audio1).transpose(0, 1).contiguous()
wav2 = torch.from_numpy(audio2).transpose(0, 1).contiguous()

if wav1.shape[0] > 1:
    wav1 = wav1.mean(dim=0, keepdim=True)
if wav2.shape[0] > 1:
    wav2 = wav2.mean(dim=0, keepdim=True)
if sr1 != target_sr:
    wav1 = torchaudio.functional.resample(wav1, sr1, target_sr)
if sr2 != target_sr:
    wav2 = torchaudio.functional.resample(wav2, sr2, target_sr)

# --- Build conversation ---

reference_audio_codes = processor.encode_audios_from_wav([wav1, wav2], sampling_rate=target_sr)
concat_prompt_wav = torch.cat([wav1, wav2], dim=-1)
prompt_audio = processor.encode_audios_from_wav([concat_prompt_wav], sampling_rate=target_sr)[0]

full_text = f"{prompt_text_speaker1} {prompt_text_speaker2} {text_to_generate}"

conversations = [
    [
        processor.build_user_message(
            text=full_text,
            reference=reference_audio_codes,
        ),
        processor.build_assistant_message(
            audio_codes_list=[prompt_audio]
        ),
    ],
]

# --- Inference ---

batch_size = 1

save_dir = Path("output")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0
with torch.no_grad():
    for start in range(0, len(conversations), batch_size):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="continuation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2000,
        )

        for message in processor.decode(outputs):
            for seg_idx, audio in enumerate(message.audio_codes_list):
                sf.write(
                    save_dir / f"{sample_idx}_{seg_idx}.wav",
                    audio.detach().cpu().to(torch.float32).numpy(),
                    int(processor.model_config.sampling_rate),
                )
            sample_idx += 1

```
### Batch Inference

You can use the provided inference script for batch inference. The script automatically uses all visible GPUs. You can control GPU visibility via `export CUDA_VISIBLE_DEVICES=<device_ids>`.

```bash
python inference.py \
  --model_path OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --codec_model_path OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --input_jsonl /path/to/input.jsonl \
  --save_dir outputs \
  --mode voice_clone_and_continuation \
  --batch_size 1 \
  --text_normalize
```

Parameters:

- `--model_path`: Path or HuggingFace model ID for MOSS-TTSD.
- `--codec_model_path`: Path or HuggingFace model ID for MOSS-Audio-Tokenizer.
- `--input_jsonl`: Path to the input JSONL file containing dialogue scripts and speaker prompts.
- `--save_dir`: Directory where the generated audio files will be saved.
- `--mode`: Inference mode. Choices: `generation`, `continuation`, `voice_clone`, `voice_clone_and_continuation`. We recommend using `voice_clone_and_continuation` for the best voice cloning experience.
- `--batch_size`: Number of samples per batch (default: `1`).
- `--max_new_tokens`: Maximum number of new tokens to generate. Controls total generated audio length (1s ≈ 12.5 tokens).
- `--temperature`: Sampling temperature (default: `1.1`).
- `--top_p`: Top-p sampling threshold (default: `0.9`).
- `--top_k`: Top-k sampling threshold (default: `50`).
- `--repetition_penalty`: Repetition penalty (default: `1.1`).
- `--text_normalize`: Normalize input text (**recommended to always enable**).
- `--sample_rate_normalize`: Resample prompt audios to the lowest sample rate before encoding (**recommended when using 2 or more speakers**).

#### JSONL Input Format

The input JSONL file should contain one JSON object per line. MOSS-TTSD supports 1 to 5 speakers per dialogue. Use `[S1]`–`[S5]` tags in the `text` field and provide corresponding `prompt_audio_speakerN` / `prompt_text_speakerN` pairs for each speaker:
```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]Speaker 1 dialogue[S2]Speaker 2 dialogue[S3]...[S4]...[S5]...",
  "prompt_audio_speaker1": "path/to/speaker1_audio.wav",
  "prompt_text_speaker1": "Reference text for speaker 1 voice cloning",
  "prompt_audio_speaker2": "path/to/speaker2_audio.wav",
  "prompt_text_speaker2": "Reference text for speaker 2 voice cloning",
  "...": "...",
  "prompt_audio_speaker5": "path/to/speaker5_audio.wav",
  "prompt_text_speaker5": "Reference text for speaker 5 voice cloning"
}
```

## License

MOSS-TTSD is released under the Apache 2.0 license.

## Citation

```
@article{moss2025ttsd,
  title={Text to Spoken Dialogue Generation},
  author={OpenMOSS},
  year={2025}
}
```

## ⚠️ Usage Disclaimer

This project provides an open-source spoken dialogue synthesis model intended for academic research, educational purposes, and legitimate applications such as AI podcast production, assistive technologies, and linguistic research. Users must not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities, and should ensure compliance with local laws and regulations while upholding ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.

<br>

# MOSS-TTS Family

## Overview
MOSS‑TTS Family is an open‑source **speech and sound generation model family** from [MOSI.AI](https://mosi.cn/#hero) and the [OpenMOSS team](https://www.open-moss.com/). It is designed for **high‑fidelity**, **high‑expressiveness**, and **complex real‑world scenarios**, covering stable long‑form speech, multi‑speaker dialogue, voice/character design, environmental sound effects, and real‑time streaming TTS.


## Introduction

<p align="center">
  <img src="./asset/moss_tts_family.jpeg" width="85%" />
</p>

When a single piece of audio needs to **sound like a real person**, **pronounce every word accurately**, **switch speaking styles across content**, **remain stable over tens of minutes**, and **support dialogue, role‑play, and real‑time interaction**, a single TTS model is often not enough. The **MOSS‑TTS Family** breaks the workflow into five production‑ready models that can be used independently or composed into a complete pipeline.

- **MOSS‑TTS**: MOSS-TTS is the flagship, production-ready Text-to-Speech foundation model in the MOSS-TTS Family, built to ship, scale, and deliver real-world voice applications beyond demos. It provides high-fidelity zero-shot voice cloning as the core capability, along with ultra-long speech generation, token-level duration control, multilingual and code-switched synthesis, and fine-grained Pinyin/phoneme pronunciation control. Together, these features make it a robust base model for scalable narration, dubbing, and voice-driven products.
- **MOSS‑TTSD**: MOSS-TTSD is a production-oriented long-form spoken dialogue generation model for creating highly expressive, multi-party conversational audio at scale. It supports continuous long-duration generation, flexible multi-speaker turn-taking control, and zero-shot voice cloning from short reference audio, enabling natural conversations with rich interaction dynamics. It is designed for real-world long-form content such as podcasts, audiobooks, commentary, dubbing, and entertainment dialogue.
- **MOSS‑VoiceGenerator**: MOSS-VoiceGenerator is an open-source voice design system that generates speaker timbres directly from free-form text descriptions, enabling fast creation of voices for characters, personalities, and emotions—without requiring reference audio. It unifies timbre design, style control, and content synthesis in a single instruction-driven model, producing high-fidelity, emotionally expressive speech that feels naturally human. It can be used standalone for creative production, or as a voice design layer that improves integration and usability for downstream TTS systems.
- **MOSS‑SoundEffect**: MOSS-SoundEffect is a high-fidelity sound effect generation model built for real-world content creation, offering strong environmental richness, broad category coverage, and reliable duration controllability. Trained on large-scale, high-quality data, it generates consistent audio from text prompts across natural ambience, urban scenes, creatures, human actions, and music-like clips. It is well suited for film and game production, interactive experiences, and data synthesis pipelines.
- **MOSS‑TTS‑Realtime**: MOSS-TTS-Realtime is a context-aware, multi-turn streaming TTS foundation model designed for real-time voice agents. Unlike conventional TTS that synthesizes replies in isolation, it conditions generation on multi-turn dialogue history—including both textual and acoustic signals from prior user speech—so responses stay coherent, consistent, and natural across turns. With low-latency incremental synthesis and strong voice stability, it enables truly conversational, human-like real-time speech experiences.


## Released Models

| Model | Architecture | Size | Model Card | Hugging Face |
|---|---|---:|---|---|
| **MOSS-TTS** | MossTTSDelay | 8B | [moss_tts_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) |
|  | MossTTSLocal | 1.7B | [moss_tts_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) |
| **MOSS‑TTSD‑V1.0** | MossTTSDelay | 8B | [moss_ttsd_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_ttsd_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) |
| **MOSS‑VoiceGenerator** | MossTTSDelay | 1.7B | [moss_voice_generator_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_voice_generator_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-Voice-Generator) |
| **MOSS‑SoundEffect** | MossTTSDelay | 8B | [moss_sound_effect_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_sound_effect_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) |
| **MOSS‑TTS‑Realtime** | MossTTSRealtime | 1.7B | [moss_tts_realtime_model_card.md](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_realtime_model_card.md) | 🤗 [Huggingface](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) |
