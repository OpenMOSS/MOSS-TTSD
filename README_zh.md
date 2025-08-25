<div align="center">
    <h1>
    MOSS: 文本到口语对话生成
    </h1>
    <p>
    <img src="asset/OpenMOSS_logo.png" alt="OpenMOSS Logo" width="300">
    <p>
    </p>
    <a href="https://www.open-moss.com/cn/moss-ttsd/"><img src="https://img.shields.io/badge/博客-阅读更多-green" alt="blog"></a>
    <a href="https://www.open-moss.com/en/moss-ttsd/"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-orange" alt="paper"></a>
    <a href="https://huggingface.co/fnlp/MOSS-TTSD-v0.5"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型页-yellow" alt="Hugging Face"></a>
    <a href="https://huggingface.co/spaces/fnlp/MOSS-TTSD"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"></a>
    <a href="https://github.com/"><img src="https://img.shields.io/badge/Python-3.10+-orange" alt="version"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/PyTorch-2.0+-brightgreen" alt="python"></a>
    <a href="https://github.com/OpenMOSS/MOSS-TTSD"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>

# MOSS-TTSD 🪐

[English](README.md) | [简体中文](README_zh.md)

## 概述

MOSS-TTSD（text to spoken dialogue）是一个开源的中英双语口语对话合成模型，可以将包含两位说话人的对话脚本转换为自然、富有表现力的对话语音。MOSS-TTSD支持双说话人零样本音色克隆与长时间单段语音生成，非常适合播客，访谈，聊天等对话场景。
详细模型介绍与演示请见我们的[中文博客](https://www.open-moss.com/cn/moss-ttsd/)和[Blog-en](https://www.open-moss.com/en/moss-ttsd/)。模型权重在 [Hugging Face](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) 提供，并可在 [Spaces 演示](https://huggingface.co/spaces/fnlp/MOSS-TTSD) 在线体验。

## 亮点

- **高表现力对话语音**：基于统一语义-声学神经音频Codec、预训练大语言模型、百万小时TTS数据与约40万小时的真实/合成对话语音数据，MOSS-TTSD能够生成高表现力，高自然度，具有自然对话韵律的拟人对话语音。
- **双说话人零样本声音克隆**：MOSS-TTSD支持零样本双说话人克隆，按脚本精确进行角色/声线切换。只需要提供10到20秒的参考音频片段。
- **中英双语**：MOSS-TTSD支持中英两种语言的高表现力语音生成。
- **长音频生成**：得益于低码率Codec与训练框架优化，MOSS-TTSD在长音频生成场景进行了大量训练（训练最大长度达到960s），能够单次生成超长音频。
- **开源可商用**：当前与后续版本将保持开源，支持免费商用。

## 最新动态 🚀

- **[2025-08-25]** 我们发布了 32khz XY-Tokenizer。
- **[2025-08-12]** 我们支持了 MOSS-TTSD v0.5 的流式推理。
- **[2025-07-29]** 我们提供了 MOSS-TTSD v0.5 的硅基流动API调用接口和调用示例。
- **[2025-07-16]** 我们开源了 MOSS-TTSD v0.5 的微调代码，支持全量微调，LoRA微调和多机训练。
- **[2025-07-04]** 我们发布了 MOSS-TTSD v0.5：显著提升了音色切换准确率、声音克隆能力与稳定性。
- **[2025-06-20]** 我们发布了 MOSS-TTSD v0。此外，我们还提供了播客生成流水线 Podever，可以便捷地将 PDF、URL 或长文本等内容自动转换为高质量播客。

## 安装

运行 MOSS-TTSD 需要安装依赖，可以使用 pip 和 conda 创建环境。

### 使用 conda

```bash
conda create -n moss_ttsd python=3.10 -y && conda activate moss_ttsd
pip install -r requirements.txt
pip install flash-attn
```

### 下载 XY-Tokenizer 权重

首先需要下载 XY-Tokenizer 的Codec模型权重，见 [XY_Tokenizer仓库](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0_32k)。

```bash
mkdir -p XY_Tokenizer/weights
huggingface-cli download fnlp/XY_Tokenizer_TTSD_V0_32k xy_tokenizer.ckpt --local-dir ./XY_Tokenizer/weights/
```

## 使用方法

### 本地推理

我们提供了模型推理脚本用于模型的本地推理。使用前请确保已下载好模型权重及对应的配置文件。

```bash
python inference.py --jsonl examples/examples.jsonl --output_dir outputs --seed 42 --use_normalize --silence_duration 0
```

参数说明：

- `--jsonl`：输入 JSONL 文件路径，包含对话脚本与参考音频
- `--output_dir`：生成音频文件的保存目录
- `--seed`：随机种子
- `--use_normalize`：是否启用文本归一化（建议开启）
- `--dtype`：模型精度（默认 `bf16`）
- `--attn_implementation`：注意力实现（默认 `flash_attention_2`，也支持 `sdpa`、`eager`）
- `--silence_duration`：参考音频与生成音频之间的静默时长（默认 0 秒），当生成音频开头出现杂音时（通常因为生成音频续写了prompt的尾音），请尝试将该参数设置为0.1。

**Windows用户需要将attn_implementation参数设置为sdpa或者eager**

#### JSONL 输入格式

MOSS-TTSD支持多种输入格式：

**格式1：仅文本（不进行声音克隆，使用模型随机音色）**

```json
{
  "text": "[S1]说话人1的内容[S2]说话人2的内容[S1]..."
}
```

**格式2：分别提供两位说话人的参考音频**

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]说话人1的内容[S2]说话人2的内容[S1]...",
  "prompt_audio_speaker1": "path/to/speaker1_audio.wav",
  "prompt_text_speaker1": "说话人1参考音频的对应文本",
  "prompt_audio_speaker2": "path/to/speaker2_audio.wav",
  "prompt_text_speaker2": "说话人2参考音频的对应文本"
}
```

**格式3：共享参考音频（一个参考音频包含两个说话人的内容）**

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]说话人1的内容[S2]说话人2的内容[S1]...",
  "prompt_audio": "path/to/shared_reference_audio.wav",
  "prompt_text": "[S1]用于说话人1的参考文本[S2]用于说话人2的参考文本"
}
```

#### 字段说明

**通用字段：**

- `text`：带 `[S1]`、`[S2]` 说话人标签的对话脚本（必填）
- `base_path`：相对路径的基准目录（可选）

**用于声音克隆（格式2）：**

- `prompt_audio_speaker1/2`：两位说话人的参考音频（可相对 `base_path`）
- `prompt_text_speaker1/2`：对应参考音频的文本，有助于更好匹配音色

**用于共享参考（格式3）：**

- `prompt_audio`：包含两位说话人的共享参考音频（可相对 `base_path`）
- `prompt_text`：对应的参考文本，亦使用 `[S1]`、`[S2]` 区分

#### 说话人标签

- `[S1]`：表示说话人1开始说话
- `[S2]`：表示说话人2开始说话

示例：

```
[S1]你好，今天感觉如何？[S2]挺好的，谢谢关心！[S1]那太好了。
```

**GPU 显存需求**

MOSS-TTSD非常高效，显存需求很低。例如默认 `bf16` 精度生成 600 秒音频时，模型需要的显存低于 7GB。这使得MOSS-TTSD可以运行在大多数消费级GPU上。
我们提供了一个显存估算公式来估计实际的显存开销

$$
y = 0.00172x + 5.8832
$$

其中 $x$ 为生成音频时长（秒），$y$ 为显存占用（GB）。

> 请注意，如果您的prompt（例如 `prompt_audio_speaker1`）比我们的默认示例更长，显存开销会更高。

| 生成时长（秒） | 显存占用（GB） |
| -------------- | -------------- |
| 120            | 6.08           |
| 300            | 6.39           |
| 360            | 6.5            |
| 600            | 6.91           |

### Web UI 使用

你还可以使用以下的命令启动Gradio界面，通过Web UI来使用MOSS-TTSD。

```bash
python gradio_demo.py
```

### 流式推理

我们支持了音频输出的流式推理，`streamer.py` 提供了一个流式音频生成的参考实现。与一次性生成完整音频序列的批量推理不同，这种流式方法会在生成语音 token 的同时，逐步处理并输出音频片段，从而显著缩短首段音频的生成时间。`AudioIteratorStreamer` 类展示了如何实现语音 token 的分片解码，每个片段大约对应 20 秒的音频。


```bash
python streamer.py \
  --jsonl examples/examples.jsonl \
  --output_dir outputs/streamer \
  --dtype bf16 \
  --attn_implementation flash_attention_2 \
  --use_tqdm
```

**Windows用户需要将attn_implementation参数设置为sdpa或者eager**

参数说明：

- `--jsonl`：输入 JSONL 文件路径，包含对话脚本与参考音频（默认 `examples/examples.jsonl`）
- `--seed`：随机种子（可选）
- `--output_dir`：用于保存流式片段和最终音频的目录（默认 `outputs/streamer`）
- `--use_normalize`：是否进行文本归一化（默认 `True`）
- `--dtype`：`bf16`（默认）、`fp16`、`fp32`
- `--attn_implementation`：`flash_attention_2`（默认）、`sdpa`、`eager`
- `--use_tqdm`：显示 token 级进度条

输出说明：

- 流式片段音频：`chunk_0.flac`、`chunk_1.flac` ... 保存在 `--output_dir`
- 拼接后的全量音频：`full_audio.flac` 保存在 `--output_dir`

注意：

- 当前流式仅支持 batch size = 1

### API 使用

#### 批处理工具

我们提供了基于 SiliconFlow API 的Batch处理工具 `use_api.py`，用于并发处理多条对话生成任务。

##### 环境变量

在使用API工具之前，你需要设置用于 API 身份验证的环境变量：

```bash
export SILICONFLOW_API_KEY="your_siliconflow_api_key"
export SILICONFLOW_API_BASE="https://api.siliconflow.cn/v1"
```

##### 运行方式

```bash
python use_api.py --jsonl_file your_data.jsonl --output_dir your_output --max_workers 8
```

##### 参数说明

- `--jsonl_file`：输入 JSONL 文件（默认 `examples/examples.jsonl`）
- `--output_dir`：生成音频输出目录（默认 `api_outputs`）
- `--max_workers`：并发工作器数（默认 8）

##### 输入格式

API调用与本地推理相同，支持以下 JSONL 格式：

**格式1：分别提供两位说话人的参考音频**

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]你好！[S2]嗨，你最近怎么样？[S1]我感觉很棒！",
  "prompt_audio_speaker1": "speaker1_reference.wav",
  "prompt_text_speaker1": "说话人1的参考文本",
  "prompt_audio_speaker2": "speaker2_reference.wav",
  "prompt_text_speaker2": "说话人2的参考文本"
}
```

**格式2：共享参考音频**

```json
{
  "base_path": "/path/to/audio/files",
  "text": "[S1]你好！[S2]嗨，你最近怎么样？[S1]我感觉很棒！",
  "prompt_audio": "shared_reference.wav",
  "prompt_text": "[S1]说话人1参考[S2]说话人2参考"
}
```

##### 输出

API调用会产生：
1. 多个 `output_XXXX.wav` 音频文件
2. `output_results.jsonl` 处理结果索引（含各文件路径）

## 播客生成

我们提供了一个自动化的播客生成工具，可直接分析 URL 或用户上传的 PDF/文本文件，提取内容并生成高质量语音博客。

使用前请确保 `OPENAI_API_KEY` 与 `OPENAI_API_BASE` 环境变量设置正确。我们推荐使用 Gemini API 生成播客脚本，因此 API key 应为 Gemini key，API base 设为 `https://generativelanguage.googleapis.com/v1beta/openai/`。

此外，也可以使用其他模型来生成博客脚本。

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="your_openai_api_base"

# 处理网页
python podcast_generate.py "https://www.open-moss.com/cn/moss-ttsd/"

# 处理 PDF 文件
python podcast_generate.py "examples/Attention Is All You Need.pdf"

# 处理文本文件
python podcast_generate.py "examples/example.txt"

# 自定义输出目录
python podcast_generate.py "your_input" -o "your_output"

# 生成英文播客
python podcast_generate.py "your_input" -l en
```

脚本支持中文（`zh`）与英文（`en`）两种语言，默认中文，可通过 `--language`/`-l` 指定。

## 微调（Fine-Tuning）

我们在 `finetune` 目录内提供了模型微调脚本与数据预处理工具。模型微调可以让模型在固定说话人的数据集上进行训练，提升模型对于特定说话人的音色克隆能力。

### 文件结构

```
MOSS-TTSD/
└── finetune/
    ├── requirements_finetune.txt     # 微调依赖
    ├── finetune_workflow.py          # 一键式微调工作流脚本
    ├── data_preprocess.py            # 数据预处理脚本
    ├── finetune.py                   # 训练脚本
    ├── training_config.yaml          # 训练超参
    └── finetune_config.yaml          # 工作流配置模板
```

### 环境准备

#### 使用 conda

```bash
conda create -n moss_ttsd_finetune python=3.10 -y && conda activate moss_ttsd_finetune
pip install -r finetune/requirements_finetune.txt
pip install flash-attn
```

#### 使用 venv

```bash
python -m venv moss_ttsd_finetune
source moss_ttsd_finetune/bin/activate
pip install -r finetune/requirements_finetune.txt
pip install flash-attn --no-build-isolation
```

### 数据准备

按照前文【使用方法/本地推理/JSONL 输入格式】组织 JSONL 文件，可包含多条符合格式的样例。可参考 `examples/examples.jsonl` 与 `examples/examples_single_reference.jsonl`。

准备好 JSONL 后，可用 `data_preprocess.py` 手动预处理：

```bash
python finetune/data_preprocess.py --jsonl <path_to_jsonl> --model_path <path_to_model> --output_dir <output_directory> --data_name <data_name> --use_normalize
```

#### 参数说明

- `--jsonl`：JSONL 输入路径（必填）
- `--model_path`：预训练 MOSS-TTSD 模型目录（可选，不填默认使用 `fnlp/MOSS-TTSD-v0.5`）
- `--output_dir`：处理后数据的输出目录（必填）
- `--data_name`：输出文件名前缀（默认 `processed_data`）
- `--use_normalize`：是否启用文本归一化（默认 `False`）

#### 支持的 JSONL 格式

**格式1：单个音频 + 全量转写**

```json
{
  "file_path": "/path/to/audio.wav",
  "full_transcript": "[S1]内容[S2]内容..."
}
```

**格式2：分离的参考与主音频**

```json
{
  "reference_audio": "/path/to/reference.wav",
  "reference_text": "[S1]用于克隆的参考文本[S2]用于克隆的参考文本",
  "audio": "/path/to/main.wav",
  "text": "[S1]内容[S2]内容..."
}
```

#### 输出文件

1. `<data_name>.pkl`：包含处理好的训练样本（input_ids 与 labels）
2. `<data_name>_metas.npy`：偏移元数据，加速加载

### 训练

在生成处理好的训练数据后，你可以使用 `finetune.py` 脚本在自定义数据集上对 MOSS-TTSD 模型进行微调。该脚本同时支持完整模型微调和 LoRA（低秩适配）微调。

#### 使用方式

**全参微调：**

```bash
python finetune/finetune.py --model_path <path_to_model> --data_dir <path_to_processed_data> --output_dir <output_directory> --training_config <training_config_file>
```

**LoRA 微调：**

```bash
python finetune/finetune.py --model_path <path_to_model> --data_dir <path_to_processed_data> --output_dir <output_directory> --training_config <training_config_file> --lora_config <lora_config_file>  --lora
```

**多卡训练：**

```bash
torchrun --nproc_per_node=8 --master_port=29500 finetune/finetune.py \
    --model_path <path_to_model> \
    --data_dir <path_to_processed_data> \
    --output_dir <output_directory> \
    --training_config <training_config_file> \
    --lora_config <lora_config_file> \
    --lora
```

#### 参数说明

- `--model_path`：预训练 MOSS-TTSD 模型目录的路径（可选，默认 `fnlp/MOSS-TTSD-v0.5`）
- `--data_dir`：包含已处理训练数据的目录（必填，含 .pkl 与 _metas.npy）
- `--output_dir`：用于保存微调后模型的目录（必填）
- `--training_config`：训练配置 YAML 文件的路径（默认 `finetune/training_config.yaml`）
- `--lora_config`：LoRA 配置 YAML 文件的路径（默认 `finetune/lora_config.yaml`）
- `--lora`：启用 LoRA（低秩适配）微调以提升内存效率（可选）

#### LoRA配置

在使用 --lora 时，你可以通过编辑配置文件 `lora_config.yaml` 来自定义 LoRA 参数。

**LoRA参数**
- **r (rank)**：控制瓶颈层的大小。值越小占用的内存越少，但可能限制模型的适配能力。
- **lora_alpha**：LoRA 权重的缩放因子。值越大，LoRA 在模型中的影响力越强。
- **target_modules**：需要适配的线性层。默认会覆盖注意力层和前馈层。
- **lora_dropout**：用于防止过拟合的正则化方法。
- **use_rslora**：启用 Rank-Stabilized LoRA，以提升训练稳定性。

#### 训练配置

训练参数可通过 YAML 文件进行配置，默认配置文件位于 `finetune/training_config.yaml`。主要参数包括：

- `per_device_train_batch_size`: 每个设备上的批处理大小
- `gradient_accumulation_steps`: 梯度累积的步数
- `num_train_epochs`: 训练的总轮数
- `learning_rate`: 学习率
- `bf16`: 是否使用 bfloat16 精度
- `warmup_ratio`: 预热比例
- `lr_scheduler_type`: 学习率调度器类型

### 一键式微调工作流

为了简化微调流程，我们提供了一个完整的工作流脚本（`finetune_workflow.py`），它能在一条命令中自动完成数据预处理和模型微调，无需分别运行多个脚本，从而实现更流畅、高效的操作体验。

#### 快速开始

1.	**配置工作流**：在 `finetune/finetune_config.yaml` 中填写配置模板
2.	**运行工作流**：使用你的配置执行工作流脚本

#### 配置模板

我们的工作流通过 YAML 配置文件指定所有参数，你可以在 `finetune/finetune_config.yaml` 找到一个空的模板：

```yaml
path_to_jsonl :           # 训练数据的 JSONL 文件路径
data_output_directory :   # 处理后数据保存的目录
data_name :               # 数据集名称
use_normalize :           # 是否对数据进行归一化（true/false）
path_to_model :           # 预训练模型路径（留空则使用默认 HuggingFace 模型）
finetuned_model_output :  # 微调后模型保存的目录
training_config_file : finetune/training_config.yaml  # 训练配置文件路径
use_lora :                # 是否使用 LoRA 微调（true/false）
lora_config_file : finetune/lora_config.yaml  # LoRA 配置文件路径
```

#### 示例配置

```yaml
path_to_jsonl : /path/to/your/training_data.jsonl
data_output_directory : /path/to/processed_data
data_name : my_dataset
use_normalize : true
path_to_model : # 留空则使用 HuggingFace 上的 fnlp/MOSS-TTSD-v0.5
finetuned_model_output : /path/to/output/fine_tuned_model
training_config_file : /path/to/training_config.yaml
use_lora : true
lora_config_file : /path/to/lora_config.yaml
```

#### 使用方式

```bash
python finetune/finetune_workflow.py --config path/to/your/config.yaml
```

#### 参数说明

- `-c`, `--config`: 工作流配置 YAML 文件的路径（默认：`./finetune/finetune_config.yaml`）

## 演示

更多演示请查看我们的博客页面：https://www.open-moss.com/cn/moss-ttsd/

## 局限性

目前，我们的模型仍存在一些不稳定情况，例如说话人切换错误和音色克隆偏差。在后续版本中，我们将进一步优化模型以提升稳定性。

## 许可证

本项目基于 Apache 2.0 许可协议开源。

## 引用

```
@article{moss2025ttsd,
  title={Text to Spoken Dialogue Generation},
  author={OpenMOSS Team},
  year={2025}
}
```

## ⚠️ 使用声明

本项目提供的一个开源的语音对话合成模型，旨在用于学术研究、教育用途，以及合法应用场景，如 AI 播客制作、辅助技术和语言学研究。用户不得将该模型用于未经授权的语音克隆、冒充他人、诈骗、欺诈、深度伪造或任何非法活动，并应确保遵守当地法律法规，同时维护伦理规范。开发者对任何不当使用行为不承担责任，并倡导负责任的 AI 开发与使用，鼓励社区在 AI 研究和应用中遵守安全与伦理原则。如对伦理或不当使用有任何疑问，请与我们联系。
