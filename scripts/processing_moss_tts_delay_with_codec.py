"""
Processor class for Asteroid fused with Codec.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class MossTTSDelayWithCodecProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    tokenizer: Any

    def __init__(
        self,
        tokenizer: Any = None,
        chat_template=None,
        AudioToken_PlaceHolder="<|Audio|>",
        audio_assistant_delay_slot_token="<|audio_assistant_delay_slot|>",
        audio_assistant_gen_slot_token="<|audio_assistant_gen_slot|>",
        audio_end_token="<|audio_end|>",
        audio_start_token="<|audio_start|>",
        audio_user_slot_token="<|audio_user_slot|>",
        audio_pad_code: int = 1024,
        text_pad_code: int = 151643,
        n_vq: int = 32,
        shift: bool = True,
        downsample_rate: int = 1920,
        continuation: bool = False,
    ):
        super().__init__(tokenizer, chat_template=chat_template)
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided.")

        self.tokenizer = tokenizer
        self.audio_user_slot_token = (
            tokenizer.audio_user_slot_token
            if hasattr(tokenizer, "audio_user_slot_token")
            else audio_user_slot_token
        )
        self.audio_start_token = (
            tokenizer.audio_start_token
            if hasattr(tokenizer, "audio_start_token")
            else audio_start_token
        )
        self.audio_end_token = (
            tokenizer.audio_end_token
            if hasattr(tokenizer, "audio_end_token")
            else audio_end_token
        )
        self.AudioToken_PlaceHolder = AudioToken_PlaceHolder
        self.audio_assistant_gen_slot_token = audio_assistant_gen_slot_token
        self.audio_assistant_delay_slot_token = audio_assistant_delay_slot_token

        self.audio_user_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_user_slot_token
        )
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_start_token
        )
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_end_token)
        self.audio_assistant_gen_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_assistant_gen_slot_token
        )
        self.audio_assistant_delay_slot_token_id = tokenizer.convert_tokens_to_ids(
            self.audio_assistant_delay_slot_token
        )

        self.audio_pad_code = audio_pad_code
        self.text_pad_code = text_pad_code
        self.n_vq = n_vq
        self.shift = shift
        self.downsample_rate = downsample_rate
        self.continuation = continuation

    def __call__(self, *args, **kwargs) -> BatchFeature:
        text = args[0] if len(args) > 0 else kwargs.pop("text", None)
        kwargs.pop("padding", True)
        return_tensors = kwargs.pop("return_tensors", "pt")
        audios = kwargs.pop("audios", None)
        continuation = kwargs.pop("continuation", self.continuation)

        text = self._normalize_text_input(text)
        audios, audio_pad_lengths, audio_lengths = self._normalize_audios(audios)

        continuation_feature = None
        continuation_audio_pad_length = None
        if continuation and audios is not None:
            continuation_feature = self._build_continuation_feature(audios)
            continuation_audio_pad_length = (
                self._get_audio_feature_length(continuation_feature)
                // self.downsample_rate
            )

        unified_tokens = self._get_unified_codes(
            text,
            audio_pad_lengths,
            continuation_audio_pad_length=continuation_audio_pad_length,
        )

        if audios is None:
            inputs = {
                "input_ids": unified_tokens,
            }
        else:
            audio_features = self._pack_audio_features(audios, audio_lengths)
            inputs = {
                "input_ids": unified_tokens,
                "audio_features": audio_features,
                "feature_attention_mask": torch.as_tensor(
                    audio_lengths, dtype=torch.long
                ),
            }
            if continuation_feature is not None:
                inputs["continuation_feature"] = torch.as_tensor(
                    continuation_feature, dtype=torch.float32
                )

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    def _normalize_text_input(
        self,
        text: Optional[
            Union[
                TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
            ]
        ],
    ) -> str:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        if isinstance(text, str):
            user_text = text
        elif (
            isinstance(text, list)
            and all(isinstance(t, str) for t in text)
            and len(text) == 1
        ):
            user_text = text[0]
        else:
            raise ValueError("`text` input must be a string or list of strings (=1).")

        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    def _normalize_audios(
        self, audios: Optional[Union[np.ndarray, list[np.ndarray]]]
    ) -> tuple[Optional[list[np.ndarray]], list[int], list[int]]:
        if audios is None:
            return None, [], []

        if isinstance(audios, np.ndarray):
            audios = [audios]
        elif isinstance(audios, list) and all(
            isinstance(a, np.ndarray) for a in audios
        ):
            if len(audios) == 0:
                raise ValueError("`audios` must not be an empty list.")
        else:
            raise ValueError(
                "`audio` input must be a numpy array or list of numpy arrays."
            )

        normalized_audios = [self.loudness_normalize(audio) for audio in audios]
        normalized_audios = [
            self._coerce_audio_feature_1d(audio) for audio in normalized_audios
        ]
        audio_lengths = [
            self._get_audio_feature_length(audio) for audio in normalized_audios
        ]
        audio_pad_lengths = [
            audio_length // self.downsample_rate for audio_length in audio_lengths
        ]
        return normalized_audios, audio_pad_lengths, audio_lengths

    def _coerce_audio_feature_1d(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 0:
            raise ValueError("Each audio must have at least one dimension.")
        if audio.ndim > 1:
            squeezed = np.squeeze(audio)
            if squeezed.ndim != 1:
                raise ValueError(
                    "Each audio must be 1D or squeezable to 1D for feature packing."
                )
            audio = squeezed
        return audio

    def _get_audio_feature_length(self, audio: np.ndarray) -> int:
        audio = self._coerce_audio_feature_1d(audio)
        return int(audio.shape[-1])

    def _prepare_audio_feature(
        self, audio: np.ndarray, target_length: int
    ) -> np.ndarray:
        audio = self._coerce_audio_feature_1d(audio)
        audio_length = int(audio.shape[-1])
        if audio_length > target_length:
            raise ValueError("Audio length exceeds target padding length.")

        padded_audio = np.zeros((target_length,), dtype=np.float32)
        padded_audio[:audio_length] = audio
        return padded_audio

    def _pack_audio_features(
        self, audios: Sequence[np.ndarray], audio_lengths: Sequence[int]
    ) -> torch.Tensor:
        if len(audios) != len(audio_lengths):
            raise ValueError("Audio features and lengths must have the same size.")
        if len(audios) == 0:
            raise ValueError("Audio features must not be empty.")

        max_audio_length = max(audio_lengths)
        padded_audios = [
            self._prepare_audio_feature(audio, max_audio_length) for audio in audios
        ]
        return torch.as_tensor(np.stack(padded_audios, axis=0), dtype=torch.float32)

    def _build_continuation_feature(self, audios: Sequence[np.ndarray]) -> np.ndarray:
        if len(audios) == 0:
            raise ValueError("Continuation audio features must not be empty.")
        return np.concatenate(
            [self._coerce_audio_feature_1d(audio) for audio in audios], axis=0
        )

    def _replace_audio_placeholders(self, content: str, lengths: Sequence[int]) -> str:
        parts = content.split(self.AudioToken_PlaceHolder)
        placeholder_count = len(parts) - 1

        if placeholder_count != len(lengths):
            raise ValueError(
                f"Number of {self.AudioToken_PlaceHolder} ({placeholder_count}) does not match lengths ({len(lengths)})"
            )

        if placeholder_count == 0:
            return content

        merged_parts = [parts[0]]
        for length, suffix in zip(lengths, parts[1:]):
            if length < 0:
                raise ValueError(f"length must be >= 0, got {length}")
            slot_count = length + self.n_vq - 1
            merged_parts.append(
                f"{self.audio_start_token}{self.audio_user_slot_token * slot_count}{self.audio_end_token}"
            )
            merged_parts.append(suffix)
        return "".join(merged_parts)

    def _build_padded_audio_codes(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.full(
            (length, self.n_vq),
            self.audio_pad_code,
            device=device,
            dtype=dtype,
        )

    def _build_assistant_audio_block(self, length: int) -> str:
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if length == 0:
            return f"{self.audio_start_token}{self.audio_end_token}"
        return (
            f"{self.audio_start_token}"
            f"{self.audio_user_slot_token * length}"
            f"{self.audio_assistant_delay_slot_token * (self.n_vq - 1)}"
            f"{self.audio_end_token}"
        )

    def _get_continuation_unified_codes(
        self, audio_pad_length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        text_codes = torch.tensor(
            self.tokenizer.encode(self._build_assistant_audio_block(audio_pad_length)),
            dtype=dtype,
            device=device,
        )
        delayed_audio_codes = self.apply_delay_pattern(
            self._build_padded_audio_codes(
                audio_pad_length,
                device=device,
                dtype=dtype,
            ),
            ch0_pad_id=self.audio_pad_code,
            pad_id=self.audio_pad_code,
        )
        delayed_audio_codes = torch.cat(
            [
                torch.full(
                    (1, self.n_vq),
                    self.audio_pad_code,
                    device=device,
                    dtype=dtype,
                ),
                delayed_audio_codes[:audio_pad_length],
            ],
            dim=0,
        )
        text_codes = text_codes[: delayed_audio_codes.shape[0]]
        return torch.cat([text_codes.unsqueeze(1), delayed_audio_codes], dim=1)

    def _get_unified_codes(
        self,
        content: str,
        audio_pad_lengths: Sequence[int],
        continuation_audio_pad_length: Optional[int] = None,
    ) -> torch.Tensor:
        if len(audio_pad_lengths) > 0:
            content = self._replace_audio_placeholders(content, audio_pad_lengths)

        text_codes = torch.tensor(self.tokenizer.encode(content), dtype=torch.long)

        if len(audio_pad_lengths) == 0:
            delay_audio_codes = torch.full(
                (len(text_codes), self.n_vq),
                self.audio_pad_code,
                dtype=text_codes.dtype,
                device=text_codes.device,
            )
        else:
            audio_start_indices = torch.where(text_codes == self.audio_start_token_id)[
                0
            ]
            audio_end_indices = torch.where(text_codes == self.audio_end_token_id)[0]

            if len(audio_start_indices) != len(audio_pad_lengths) or len(
                audio_end_indices
            ) != len(audio_pad_lengths):
                raise ValueError(
                    "Audio placeholders do not match the provided audio lengths."
                )

            delay_audio_codes_list = []
            prefix_idx = 0
            for audio_start_idx_t, audio_end_idx_t, audio_pad_length in zip(
                audio_start_indices, audio_end_indices, audio_pad_lengths
            ):
                audio_start_idx = int(audio_start_idx_t.item())
                audio_end_idx = int(audio_end_idx_t.item())
                delay_audio_codes = self.apply_delay_pattern(
                    self._build_padded_audio_codes(
                        audio_pad_length,
                        device=text_codes.device,
                        dtype=text_codes.dtype,
                    ),
                    ch0_pad_id=self.audio_pad_code,
                    pad_id=self.audio_pad_code,
                )
                pad_codes = torch.full(
                    (audio_start_idx - prefix_idx + 1, self.n_vq),
                    self.audio_pad_code,
                    device=text_codes.device,
                    dtype=text_codes.dtype,
                )
                delay_audio_codes_list.extend([pad_codes, delay_audio_codes])
                prefix_idx = audio_end_idx

            last_audio_end_idx = int(audio_end_indices[-1].item())
            delay_audio_codes_list.append(
                torch.full(
                    (len(text_codes) - last_audio_end_idx, self.n_vq),
                    self.audio_pad_code,
                    device=text_codes.device,
                    dtype=text_codes.dtype,
                )
            )
            delay_audio_codes = torch.cat(delay_audio_codes_list)

        if text_codes.shape[0] > delay_audio_codes.shape[0]:
            text_codes = text_codes[: delay_audio_codes.shape[0]]
        elif text_codes.shape[0] != delay_audio_codes.shape[0]:
            raise RuntimeError("Text/audio packing length mismatch.")

        unified_codes = torch.cat([text_codes.unsqueeze(1), delay_audio_codes], dim=1)
        if continuation_audio_pad_length is not None:
            unified_codes = torch.cat(
                [
                    unified_codes,
                    self._get_continuation_unified_codes(
                        continuation_audio_pad_length,
                        device=text_codes.device,
                        dtype=text_codes.dtype,
                    ),
                ],
                dim=0,
            )
        return unified_codes

    @staticmethod
    def apply_delay_pattern(
        tokens: torch.Tensor, ch0_pad_id: int, pad_id: int
    ) -> torch.Tensor:
        delayed_tokens = torch.full(
            (tokens.shape[0] + tokens.shape[1] - 1, tokens.shape[1]),
            pad_id,
            dtype=torch.long,
        )
        delayed_tokens[:, 0] = torch.cat(
            [
                tokens[:, 0],
                torch.full((tokens.shape[1] - 1,), ch0_pad_id, dtype=torch.long),
            ]
        )
        for i in range(1, tokens.shape[1]):
            delayed_tokens[i : i + tokens.shape[0], i] = tokens[:, i]
        return delayed_tokens

    @staticmethod
    def loudness_normalize(
        wav: np.ndarray,
        target_dbfs: float = -20,
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return wav
        rms = np.sqrt(np.mean(wav * wav))
        current_dbfs = 20.0 * np.log10(rms + 1e-9)
        gain = np.clip(target_dbfs - current_dbfs, gain_range[0], gain_range[1])
        factor = 10.0 ** (gain / 20.0)
        return wav * factor


__all__ = ["MossTTSDelayWithCodecProcessor"]
