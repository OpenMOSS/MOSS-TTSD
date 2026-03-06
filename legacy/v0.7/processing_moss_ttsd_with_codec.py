"""
Processor class for MOSS-TTSD fused with Codec.
"""

import re
from typing import Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."


class MossTTSDWithCodecProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "pad_token_id": 0,  # Fallback pad token ID, actual value comes from tokenizer.pad_token_id
        },
        "audio_kwargs": {
            "max_channels": 8,  # Maximum number of quantization channels
            "audio_pad_token_id": 1024,  # Padding token ID for non-text channels
            "silence_duration": 0.0,  # Duration of silence to append for encoder segmentation
            "input_sample_rate": 16000,  # Input audio sampling rate (fallback, inferred from audio_tokenizer.config)
            "encoder_downsample_rate": 320,  # Encoder downsampling rate (fallback, inferred from audio_tokenizer.config)
            "speech_token_range": [
                151665,
                152689,
            ],  # Token range for speech tokens (first codebook offset mapping)
            "audio_bos_token": "<|begin_of_speech|>",
            "audio_eos_token": "<|end_of_speech|>",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding": True,
            "use_normalize": False,
        },
    }


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


class MossTTSDWithCodecProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|image_pad|>",
    ):
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.pad_token = (
            tokenizer.pad_token
            if hasattr(tokenizer, "pad_token")
            else "<|end_of_text|>"
        )
        self.pad_token_id = (
            tokenizer.pad_token_id
            if hasattr(tokenizer, "pad_token_id")
            else tokenizer.convert_tokens_to_ids(self.pad_token)
        )
        self.max_channels = tokenizer.init_kwargs["audio_kwargs"].get("max_channels", 8)
        self.input_sample_rate = tokenizer.init_kwargs["audio_kwargs"].get(
            "input_sample_rate", 16000
        )
        self.encoder_downsample_rate = tokenizer.init_kwargs["audio_kwargs"].get(
            "encoder_downsample_rate", 320
        )
        self.hop_length = tokenizer.init_kwargs["audio_kwargs"].get("hop_length", 160)
        self.audio_pad_token_id = tokenizer.init_kwargs["audio_kwargs"].get(
            "audio_pad_token_id", 1024
        )
        self.silence_duration = tokenizer.init_kwargs["audio_kwargs"].get(
            "silence_duration", 0.0
        )
        self.use_normalize = tokenizer.init_kwargs["common_kwargs"].get(
            "use_normalize", False
        )
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        padding=True,
        return_tensors="pt",
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        **kwargs: Unpack[MossTTSDWithCodecProcessorKwargs],
    ) -> dict:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            normalized_text = (
                normalize_text(text)
                .replace("[S1]", "<speaker1>")
                .replace("[S2]", "<speaker2>")
            )
            input_text = f"<|begin_of_style|>{SYSTEM_PROMPT}<|end_of_style|>\n<|begin_of_text|>{normalized_text}<|end_of_text|>\n<|begin_of_speech|>"
        elif (
            isinstance(text, list)
            and all(isinstance(t, str) for t in text)
            and len(text) == 1
        ):
            merged_text = (
                normalize_text("".join(text))
                .replace("[S1]", "<speaker1>")
                .replace("[S2]", "<speaker2>")
            )
            input_text = f"<|begin_of_style|>{SYSTEM_PROMPT}<|end_of_style|>\n<|begin_of_text|>{merged_text}<|end_of_text|>\n<|begin_of_speech|>"
        else:
            raise ValueError("`text` input must be a string or list of strings (=1).")

        if audios is not None:
            if self.silence_duration > 0.0:
                silence_audio = np.zeros(
                    int(self.silence_duration * self.input_sample_rate),
                    dtype=np.float32,
                )
            if isinstance(audios, np.ndarray):
                ref_audio = self.pad_or_truncate_to_seconds(audios)
                audios = (
                    np.concatenate([audios, silence_audio], axis=0)
                    if self.silence_duration > 0.0
                    else audios
                )
                audios = [audios]
            elif (
                isinstance(audios, list)
                and all(isinstance(a, np.ndarray) for a in audios)
                and len(audios) <= 2
            ):
                audios = np.concatenate(audios, axis=0)
                ref_audio = self.pad_or_truncate_to_seconds(audios)
                audios = (
                    np.concatenate([audios, silence_audio], axis=0)
                    if self.silence_duration > 0.0
                    else audios
                )
                audios = [audios]
            else:
                raise ValueError(
                    "`audio` input must be a numpy array or list of numpy arrays (<=2)."
                )
            audio_inputs = self.feature_extractor(
                audios,
                sampling_rate=self.input_sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ref_audio_inputs = self.feature_extractor(
                ref_audio,
                sampling_rate=self.input_sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )

        text_input_ids = np.array(
            self.tokenizer.encode(input_text, add_special_tokens=False)
        )
        input_ids = np.full(
            (text_input_ids.shape[0], self.max_channels), self.audio_pad_token_id
        )
        input_ids[:, 0] = text_input_ids
        if audios is not None:
            audio_token_pads = np.full(
                (
                    audio_inputs["attention_mask"].sum()
                    // (self.encoder_downsample_rate // self.hop_length),
                    self.max_channels,
                ),
                self.audio_pad_token_id,
            )
            audio_token_pads[:, 0] = self.audio_token_id
            input_ids = np.concatenate([input_ids, audio_token_pads], axis=0)
            inputs = {
                "input_ids": input_ids,
                "audio_features": audio_inputs["input_features"],
                "audio_attention_mask": audio_inputs["attention_mask"],
                "ref_audio_features": ref_audio_inputs["input_features"],
                "ref_audio_attention_mask": ref_audio_inputs["attention_mask"],
            }
        else:
            inputs = {"input_ids": input_ids}

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    def pad_or_truncate_to_seconds(
        self, wav: np.ndarray, target_seconds: float = 20
    ) -> np.ndarray:
        """Pad or truncate a mono waveform to target length in seconds.

        Args:
            wav: (1, T) or (T,) numpy array
            target_seconds: target duration in seconds
        Returns:
            (1, T_target) numpy array
        """
        wav_array = np.asarray(wav)
        if wav_array.ndim == 2 and wav_array.shape[0] == 1:
            wav_1d = wav_array[0]
        else:
            wav_1d = wav_array.reshape(-1)

        target_len = int(round(target_seconds * self.input_sample_rate))
        cur_len = wav_1d.shape[-1]
        if cur_len == target_len:
            out = wav_1d
        elif cur_len > target_len:
            out = wav_1d[:target_len]
        else:
            pad_len = target_len - cur_len
            out = np.pad(wav_1d, (0, pad_len), mode="constant")
        return np.expand_dims(out, axis=0)


__all__ = ["MossTTSDWithCodecProcessor"]
