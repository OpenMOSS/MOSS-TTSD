<div align="center">
    <h1>
    MOSS-TTSD：从文本到对话语音生成
    </h1>
    <p>
    <img src="asset/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
    <p>
    </p>
    <a href="https://mosi.cn/models/moss-ttsd"><img src="https://img.shields.io/badge/Blog-Read%20More-green" alt="blog"></a>
    <a href="https://mosi.cn/models/moss-ttsd"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-red" alt="paper"></a>
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20MOSS%20TTSD%20-v1.0-yellow" alt="MOSS-TTSD-v1.0"></a>
     <a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTSD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface%20%20-space-orange" alt="MOSS-TTSD-space"></a>
    <a href=""><img src="https://img.shields.io/badge/AI Stuidio-Coming%20Soon-blue" alt="AI Studio"></a>
    <a href="https://github.com/"><img src="https://img.shields.io/badge/Python-3.10+-orange" alt="version"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/PyTorch-2.0+-brightgreen" alt="python"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
    <br>

</div>


# MOSS-TTSD🪐

[English](README.md) | [简体中文](README_zh.md)

## 概述
 <p align="center">
    <img src="asset/ttsd.png" alt="alt text" width="330">
  </p>

MOSS-TTSD 是开源 [MOSS‑TTS Family](https://github.com/OpenMOSS/MOSS-TTS) 中专注长时多说话人对话生成的模型。相比主要针对高保真单人语音合成的基础模型，MOSS-TTSD 重点解决从离散语音片段到连续、连贯、多人互动语音内容之间的衔接问题。

该模型将任务范式从“text-to-speech”推进到“script-to-conversation”。通过强调多人互动中的节奏、情绪与角色一致性，MOSS-TTSD 能将静态脚本转化为动态、自然、富有表现力的对话语音，适合需要多角色切换且保持叙事连续性的创作者与开发者。

无论是直播访谈的临场感，还是多语言剧情内容的结构化表达，MOSS-TTSD 都能在开源框架下提供面向专业长内容生产所需的稳定性与表现力。


## 亮点
- **从独白到对话**：不同于偏朗读优化的传统 TTS，MOSS-TTSD 更注重对话韵律，可灵活支持 1 到 5 位说话人，处理自然轮替、重叠发言和角色一致性。
- **超长上下文建模**：突破短句生成范式，面向长时段稳定性设计，单次会话最长可支持约 60 分钟一致且连贯的语音生成。
- **多场景适配**：针对高变化场景进行了专项优化，包括 AI 播客、体育/电竞解说、有声书、配音与相声等。
- **多语言与零样本能力**：仅需短参考音频即可进行高质量零样本音色克隆，并在中文、英文、日文及欧洲语种上具备稳定跨语种表现。


## 更新日志 🚀
- **[2026-2-10]** 发布 MOSS-TTSD v1.0。该里程碑版本支持单次 60 分钟上下文与多人交互，大幅扩展了多语言能力和应用场景。
- **[2025-11-01]** 发布 MOSS-TTSD v0.7。显著提升音质、音色克隆能力和稳定性，新增 32 kHz 高音质输出，并将单次生成长度从 960s 提升至 1700s。
- **[2025-09-09]** 支持 SGLang 推理引擎，推理速度最高可提升至 **16x**。
- **[2025-08-25]** 发布 XY-Tokenizer 的 32 kHz 版本。
- **[2025-08-12]** MOSS-TTSD v0.5 新增流式推理支持。
- **[2025-07-29]** 提供 MOSS-TTSD v0.5 的 SiliconFlow API 接口与使用示例。
- **[2025-07-16]** 开源 MOSS-TTSD v0.5 微调代码，支持全参微调、LoRA 微调与多机训练。
- **[2025-07-04]** 发布 MOSS-TTSD v0.5，增强音色切换准确率、音色克隆能力与模型稳定性。
- **[2025-06-20]** 发布 MOSS-TTSD v0，并提供播客生成管线 Podever，可将 PDF、URL 或长文本自动转换为高质量播客。

**说明：** 你仍可在 [legacy v0.7 folder](./legacy/v0.7/README.md) 访问旧版 MOSS-TTSD v0.7。


## 支持的语言

MOSS-TTSD 目前支持 **20 种语言**：

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|---|---|---|---|---|---|---|---|---|
| 中文 | zh | 🇨🇳 | 英语 | en | 🇺🇸 | 德语 | de | 🇩🇪 |
| 西班牙语 | es | 🇪🇸 | 法语 | fr | 🇫🇷 | 日语 | ja | 🇯🇵 |
| 意大利语 | it | 🇮🇹 | 希伯来语 | he | 🇮🇱 | 韩语 | ko | 🇰🇷 |
| 俄语 | ru | 🇷🇺 | 波斯语（法尔西语） | fa | 🇮🇷 | 阿拉伯语 | ar | 🇸🇦 |
| 波兰语 | pl | 🇵🇱 | 葡萄牙语 | pt | 🇵🇹 | 捷克语 | cs | 🇨🇿 |
| 丹麦语 | da | 🇩🇰 | 瑞典语 | sv | 🇸🇪 | 匈牙利语 | hu | 🇭🇺 |
| 希腊语 | el | 🇬🇷 | 土耳其语 | tr | 🇹🇷 |  |  |  |

## 安装

运行 MOSS-TTSD 需要先安装依赖，推荐使用 conda + pip。

### 使用 conda

```bash
conda create -n moss_ttsd python=3.12 -y && conda activate moss_ttsd
pip install -r requirements.txt
pip install flash-attn
```

## 使用方式

### 快速开始

MOSS-TTSD 采用 **continuation** 工作流：为每位说话人提供参考音频与对应转写作为前缀，再提供要生成的对话文本，模型会在各说话人身份上继续生成语音。

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

prompt_audio_speaker1 = "asset/reference_02_s1.wav"
prompt_audio_speaker2 = "asset/reference_02_s2.wav"
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

### 批量推理

你可以使用仓库内置的推理脚本做批量生成。脚本会自动使用所有可见 GPU，可通过 `export CUDA_VISIBLE_DEVICES=<device_ids>` 控制可见设备。

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

参数说明：

- `--model_path`：MOSS-TTSD 的本地路径或 Hugging Face 模型 ID。
- `--codec_model_path`：MOSS-Audio-Tokenizer 的本地路径或 Hugging Face 模型 ID。
- `--input_jsonl`：输入 JSONL 文件路径，包含对话脚本与说话人提示信息。
- `--save_dir`：生成音频的输出目录。
- `--mode`：推理模式，可选 `generation`、`continuation`、`voice_clone`、`voice_clone_and_continuation`，推荐使用 `voice_clone_and_continuation` 以获得更好克隆效果。
- `--batch_size`：批大小，默认 `1`。
- `--max_new_tokens`：最大新生成 token 数，用于控制总音频长度（约 1 秒 ≈ 12.5 token）。
- `--temperature`：采样温度，默认 `1.1`。
- `--top_p`：Top-p 采样阈值，默认 `0.9`。
- `--top_k`：Top-k 采样阈值，默认 `50`。
- `--repetition_penalty`：重复惩罚系数，默认 `1.1`。
- `--text_normalize`：是否进行文本规范化（**建议始终开启**）。
- `--sample_rate_normalize`：编码前将多说话人提示音频重采样到最低采样率（**当说话人数 >= 2 时推荐开启**）。

#### JSONL 输入格式

输入 JSONL 每行一个 JSON 对象。MOSS-TTSD 支持 1 到 5 位说话人。请在 `text` 字段中使用 `[S1]`–`[S5]` 标记，并为每位说话人提供对应的 `prompt_audio_speakerN` / `prompt_text_speakerN`：

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

## 评测
### 客观评测（TTSD-eval）

我们引入了稳健评测框架，使用 **MMS-FA** 进行词级对齐与话语分段，并使用 **wespeaker** 进行说话人嵌入提取，以计算说话人归属准确率（ACC）和说话人相似度（SIM）。评测代码与数据请见 [TTSD-eval](https://github.com/OpenMOSS/TTSD-eval)。

<br>

| Model | ZH - SIM | ZH - ACC | ZH - WER | EN - SIM | EN - ACC | EN - WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Comparison with Open-Source Models** | | | | | | |
| MOSS-TTSD | **0.7949** | **0.9587** | **0.0485** | **0.7326** | **0.9626** | 0.0988 |
| MOSS-TTSD v0.7 | 0.7423 | 0.9391 | 0.0517 | 0.6743 | 0.9266 | 0.1612 |
| Vibevoice 7B | 0.7590 | 0.9222 | 0.0570 | 0.7140 | 0.9554 | **0.0946** |
| Vibevoice 1.5 B | 0.7415 | 0.8798 | 0.0818 | 0.6961 | 0.9353 | 0.1133 |
| FireRedTTS2 | 0.7383 | 0.9022 | 0.0768 | - | - | - |
| Higgs Audio V2 | - | - | - | 0.6860 | 0.9025 | 0.2131 |
| **Comparison with Proprietary Models** | | | | | | |
| Eleven V3 | 0.6970 | 0.9653 | **0.0363** | 0.6730 | 0.9498 | **0.0824** |
| MOSS-TTSD (elevenlabs_voice) | **0.8165** | **0.9736** | 0.0391 | **0.7304** | **0.9565** | 0.1005 |
| | | | | | | |
| gemini-2.5-pro-preview-tts | - | - | - | 0.6786 | 0.9537 | **0.0859** |
| gemini-2.5-flash-preview-tts | - | - | - | 0.7194 | 0.9511 | 0.0871 |
| MOSS-TTSD (gemini_voice) | - | - | - | **0.7893** | **0.9655** | 0.0984 |
| | | | | | | |
| Doubao_Podcast | 0.8034 | 0.9606 | **0.0472** | - | - | - |
| MOSS-TTSD (doubao_voice) | **0.8226** | **0.9630** | 0.0571 | - | - | - |

### 主观评测

针对开源模型，我们让标注员从说话人归属准确性、音色相似性、韵律表现和整体质量四个维度对样本对进行偏好判断；参考 LMSYS Chatbot Arena 方法计算各维度 Elo 分数及置信区间。  
![alt text](./asset/VS_Open-Source_Models.jpg)

针对闭源模型，我们仅比较每对样本的整体偏好并统计胜率。  
![alt text](./asset/VS_Proprietary_Models.png)


## 许可证

MOSS-TTSD 基于 Apache 2.0 协议开源。

## 引用

```
@article{moss2025ttsd,
  title={Text to Spoken Dialogue Generation},
  author={OpenMOSS},
  year={2025}
}
```

## ⚠️ 使用声明

本项目提供开源对话语音合成模型，面向学术研究、教育用途以及 AI 播客制作、辅助技术、语言学研究等合法应用场景。严禁将本模型用于未经授权的声音克隆、冒充、欺诈、诈骗、深度伪造或任何违法用途。使用者应遵守所在地法律法规并遵循伦理规范。开发者不对模型被滥用造成的后果承担责任，并倡导社区在 AI 研究与应用中坚持安全与伦理原则。如对伦理或滥用问题有疑问，请联系我们。

<br>

# MOSS-TTS Family

## 简介

<p align="center">
  <img src="asset/moss_tts_family.jpeg" width="85%" />
</p>

当语音内容需要同时满足 **接近真人自然度**、**高发音准确性**、**多风格切换**、**数十分钟稳定生成**，并支持**多人对话、角色扮演和实时交互**时，单一 TTS 模型往往难以覆盖全部需求。**MOSS‑TTS Family** 将生产流程拆分为 5 个可独立使用、也可自由组合的模型。

- **MOSS‑TTS**：旗舰级生产 TTS 基座模型，核心能力是高保真零样本音色克隆、可控长文本合成、准确发音与多语言/中英混说，适用于规模化旁白、配音和语音产品。
- **MOSS‑TTSD**：面向生产的长对话语音模型，可规模化生成富有表现力的多说话人对话音频，支持长时连续、轮替控制和短参考零样本克隆，适用于播客、有声书、解说、配音和娱乐内容。
- **MOSS‑VoiceGenerator**：开源音色设计模型，无需参考音频即可从自由文本描述直接生成说话人音色，统一了音色设计、风格控制和内容合成，可单独使用或作为下游 TTS 的音色层。
- **MOSS‑SoundEffect**：高保真文本到音效模型，支持广泛类别与时长可控，面向真实内容生产，可稳定生成环境声、城市场景、生物、人类动作和类音乐片段，适用于影视、游戏、交互媒体和数据合成。
- **MOSS‑TTS‑Realtime**：上下文感知的多轮流式 TTS 模型，面向实时语音智能体。通过同时利用文本历史与用户历史声学信息，提供低时延、跨轮一致的语音响应。

## 发布模型

| Model | Architecture | Size | Model Card | Hugging Face | ModelScope |
|---|---|---:|---|---|---|
| **MOSS-TTS** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS) |
|  | `MossTTSLocal` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Local-Transformer) |
| **MOSS‑TTSD‑V1.0** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_ttsd_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTSD-v1.0) |
| **MOSS‑VoiceGenerator** | `MossTTSDelay` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_voice_generator_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-VoiceGenerator) |
| **MOSS‑SoundEffect** | `MossTTSDelay` | 8B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_sound_effect_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-SoundEffect) |
| **MOSS‑TTS‑Realtime** | `MossTTSRealtime` | 1.7B | [![Model Card](https://img.shields.io/badge/Model%20Card-View-blue?logo=markdown)](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md) | [![Hugging Face](https://img.shields.io/badge/Huggingface-Model-orange?logo=huggingface)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-lightgrey?logo=modelscope)](https://modelscope.cn/models/openmoss/MOSS-TTS-Realtime) |
