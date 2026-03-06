import argparse
import json
import math
import os
import shutil
import struct
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from transformers import AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedAudioTokenizerBase


class XYTokenizerConfig(PretrainedConfig):
    model_type = "xy_tokenizer"

    def __init__(
        self,
        input_sampling_rate: int = 16000,
        sampling_rate: int = 16000,
        encoder_downsample_rate: int = 1280,
        decoder_upsample_rate: int = 1920,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        if "input_sample_rate" in kwargs and input_sampling_rate == 16000:
            input_sampling_rate = kwargs.pop("input_sample_rate")
        if "output_sample_rate" in kwargs and sampling_rate == 16000:
            sampling_rate = kwargs.pop("output_sample_rate")

        self.input_sampling_rate = input_sampling_rate
        self.sampling_rate = sampling_rate
        self.input_sample_rate = input_sampling_rate
        self.output_sample_rate = sampling_rate
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.params = kwargs

        super().__init__(**kwargs)


def sinusoids(
    length: int,
    channels: int,
    max_timescale: int = 10000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError("channels must be an even number for sinusoidal embeddings")
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment
        * torch.arange(channels // 2, device=device, dtype=torch.float32)
    )
    scaled_time = (
        torch.arange(length, device=device, dtype=torch.float32)[:, np.newaxis]
        * inv_timescales[np.newaxis, :]
    )
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def remap_positional_embedding_state_dict_key(
    state_dict: dict[str, Any], prefix: str
) -> None:
    old_key = prefix + "positional_embedding"
    new_key = old_key + ".weight"
    if new_key in state_dict and old_key not in state_dict:
        state_dict[old_key] = state_dict.pop(new_key)


class XYTokenizerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.scaling = self.head_dim**-0.5
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)


class XYTokenizerMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout, activation_function):
        super().__init__()
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)


class XYTokenizerTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        activation_function,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = XYTokenizerAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            is_causal=is_causal,
        )
        self.mlp = XYTokenizerMLP(
            hidden_size, intermediate_size, dropout, activation_function
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)


class XYTokenizerEncoder(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,
        sampling_rate=16000,
        hop_length=160,
        stride_size=2,
        kernel_size=3,
        d_model=1280,
        scale_embedding=True,
        max_audio_seconds=30,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.max_source_positions = (
            max_audio_seconds * sampling_rate // hop_length
        ) // stride_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins, self.d_model, self.stride_size = (
            num_mel_bins,
            d_model,
            stride_size,
        )
        self.conv1 = nn.Conv1d(
            num_mel_bins, d_model, kernel_size=kernel_size, padding=1
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=stride_size, padding=1
        )
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )
        self.layers = nn.ModuleList(
            [
                XYTokenizerTransformerLayer(
                    hidden_size=d_model,
                    num_attention_heads=encoder_attention_heads,
                    intermediate_size=encoder_ffn_dim,
                    activation_function=activation_function,
                    is_causal=False,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        remap_positional_embedding_state_dict_key(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class XYTokenizerDecoder(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,
        sampling_rate=16000,
        hop_length=160,
        stride_size=2,
        kernel_size=3,
        d_model=1280,
        scale_embedding=True,
        max_audio_seconds=30,
        decoder_layers=32,
        decoder_attention_heads=20,
        decoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.max_source_positions = (
            max_audio_seconds * sampling_rate // hop_length
        ) // stride_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins, self.d_model, self.stride_size = (
            num_mel_bins,
            d_model,
            stride_size,
        )
        self.deconv1 = nn.ConvTranspose1d(
            d_model, d_model, kernel_size, stride_size, padding=0, output_padding=0
        )
        self.deconv2 = nn.ConvTranspose1d(
            d_model, num_mel_bins, kernel_size, stride=1, padding=0
        )
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )
        self.layers = nn.ModuleList(
            [
                XYTokenizerTransformerLayer(
                    hidden_size=d_model,
                    num_attention_heads=decoder_attention_heads,
                    intermediate_size=decoder_ffn_dim,
                    activation_function=activation_function,
                    is_causal=False,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        remap_positional_embedding_state_dict_key(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ResidualDownConv(nn.Module):
    def __init__(self, d_model=1280, avg_pooler=4):
        super().__init__()
        self.d_model, self.avg_pooler = d_model, avg_pooler
        self.intermediate_dim = d_model * avg_pooler
        self.gate_proj = nn.Conv1d(
            d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )
        self.up_proj = nn.Conv1d(
            d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim, self.intermediate_dim, bias=False
        )
        self.act_fn = ACT2FN["silu"]
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)


class UpConv(nn.Module):
    def __init__(self, d_model=1280, stride=4):
        super().__init__()
        self.d_model, self.stride = d_model, stride
        self.up_conv = nn.ConvTranspose1d(
            self.stride * d_model, d_model, stride, stride, bias=False
        )


class XYTokenizerTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        d_model=1280,
        output_dim=1280,
        max_source_positions=1500,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        activation_function="gelu",
        attn_type="varlen",
    ):
        super().__init__()
        self.input_dim, self.d_model, self.output_dim, self.max_source_positions = (
            input_dim,
            d_model,
            output_dim,
            max_source_positions,
        )
        self.proj = (
            nn.Linear(input_dim, d_model, bias=True) if input_dim != d_model else None
        )
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )
        self.layers = nn.ModuleList(
            [
                XYTokenizerTransformerLayer(
                    hidden_size=d_model,
                    num_attention_heads=encoder_attention_heads,
                    intermediate_size=encoder_ffn_dim,
                    activation_function=activation_function,
                    is_causal=False,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.out_proj = (
            nn.Linear(d_model, output_dim, bias=True) if output_dim != d_model else None
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        remap_positional_embedding_state_dict_key(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ISTFT(nn.Module):
    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding, self.n_fft, self.hop_length, self.win_length = (
            padding,
            n_fft,
            hop_length,
            win_length,
        )
        self.register_buffer("window", torch.hann_window(win_length))


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, n_fft, padding)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, 7, 1, 3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim,
                    intermediate_dim,
                    layer_scale_init_value or 1 / num_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)


class Vocos(nn.Module):
    def __init__(
        self,
        input_channels=128,
        dim=512,
        intermediate_dim=4096,
        num_layers=30,
        n_fft=640,
        hop_size=160,
        padding="same",
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
        )
        self.head = ISTFTHead(dim, n_fft, hop_size, padding)
        self.hop_size = hop_size


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=1.0,
        decay=0.99,
        epsilon=1e-5,
        threshold_ema_dead=2,
        kmeans_init=True,
        kmeans_iters=10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        codebook = (
            torch.zeros(codebook_size, codebook_dim)
            if kmeans_init
            else torch.randn(codebook_size, codebook_dim)
        )
        self.register_buffer("codebook", codebook)
        self.register_buffer("inited", torch.tensor(not kmeans_init, dtype=torch.bool))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", codebook.clone())


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        rvq_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
        skip_rvq_ratio: float = 0.0,
        commitment: float = 1.0,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.skip_rvq_ratio = skip_rvq_ratio

        self.input_proj = (
            WNConv1d(input_dim, rvq_dim, 1) if input_dim != rvq_dim else nn.Identity()
        )
        self.output_proj = (
            WNConv1d(rvq_dim, self.output_dim, 1)
            if rvq_dim != self.output_dim
            else nn.Identity()
        )
        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    rvq_dim,
                    codebook_size,
                    codebook_dim,
                    commitment=commitment,
                    decay=decay,
                    epsilon=epsilon,
                    threshold_ema_dead=threshold_ema_dead,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )


class XYTokenizerPreTrainedModel(PreTrainedAudioTokenizerBase):
    config_class = XYTokenizerConfig
    base_model_prefix = "xy_tokenizer"
    main_input_name = "input_values"


class XYTokenizer(XYTokenizerPreTrainedModel):
    def __init__(self, config: XYTokenizerConfig):
        super().__init__(config)
        self.config = config

        params = config.params
        self.semantic_encoder = XYTokenizerEncoder(**params["semantic_encoder_kwargs"])
        self.semantic_encoder_adapter = XYTokenizerTransformer(
            **params["semantic_encoder_adapter_kwargs"]
        )
        self.acoustic_encoder = XYTokenizerEncoder(**params["acoustic_encoder_kwargs"])
        self.pre_rvq_adapter = XYTokenizerTransformer(
            **params["pre_rvq_adapter_kwargs"]
        )
        self.downsample = ResidualDownConv(**params["downsample_kwargs"])
        self.quantizer = ResidualVQ(**params["quantizer_kwargs"])
        self.post_rvq_adapter = XYTokenizerTransformer(
            **params["post_rvq_adapter_kwargs"]
        )
        self.upsample = UpConv(**params["upsample_kwargs"])
        self.acoustic_decoder = XYTokenizerDecoder(**params["acoustic_decoder_kwargs"])
        self.enhanced_vocos = Vocos(**params["vocos_kwargs"])

        self.post_init()


def _dtype_from_arg(dtype: str) -> Optional[torch.dtype]:
    if dtype == "auto":
        return None
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported --dtype: {dtype}")


_CODEC_CANONICAL_NAME_OR_PATH = "OpenMOSS-Team/MOSS_TTSD_tokenizer_hf"
_PROCESSOR_FQN = "processing_moss_ttsd_with_codec.MossTTSDWithCodecProcessor"
_LEGACY_SAFETENSORS_INDEX_TOTAL_SIZE_BIAS_BYTES = 4


def _write_json_file(
    path: str,
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    newline: bool = False,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
        if newline:
            f.write("\n")


def _cleanup_chat_template_files(output_dir: str) -> None:
    chat_template_file = os.path.join(output_dir, "chat_template.jinja")
    if os.path.isfile(chat_template_file):
        os.remove(chat_template_file)

    chat_templates_dir = os.path.join(output_dir, "chat_templates")
    if os.path.isdir(chat_templates_dir):
        shutil.rmtree(chat_templates_dir)


def _cleanup_transformers_saved_code_files(output_dir: str) -> None:
    for fn in ("configuration_moss_ttsd.py", "modeling_moss_ttsd.py"):
        p = os.path.join(output_dir, fn)
        if os.path.isfile(p):
            os.remove(p)

    pycache_dir = os.path.join(output_dir, "__pycache__")
    if os.path.isdir(pycache_dir):
        shutil.rmtree(pycache_dir)


def _patch_exported_config_json(output_dir: str, *, dtype: str) -> None:
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["architectures"] = ["MossTTSDWithCodec"]
    cfg["auto_map"] = {"AutoProcessor": _PROCESSOR_FQN}
    cfg["codec_model_name_or_path"] = _CODEC_CANONICAL_NAME_OR_PATH
    cfg["model_type"] = "moss_ttsd_with_codec"
    cfg["tie_word_embeddings"] = True

    if dtype == "auto":
        cfg["dtype"] = "bfloat16"
    elif dtype in {"fp32", "float32"}:
        cfg["dtype"] = "float32"
    elif dtype in {"fp16", "float16"}:
        cfg["dtype"] = "float16"
    elif dtype in {"bf16", "bfloat16"}:
        cfg["dtype"] = "bfloat16"

    codec_cfg = cfg.get("codec_config")
    if isinstance(codec_cfg, dict):
        codec_cfg["_name_or_path"] = _CODEC_CANONICAL_NAME_OR_PATH

    _write_json_file(config_path, cfg, indent=2, sort_keys=True, newline=False)


def _patch_exported_generation_config(output_dir: str) -> None:
    gen_path = os.path.join(output_dir, "generation_config.json")
    if not os.path.isfile(gen_path):
        return

    existing: Dict[str, Any] = {}
    try:
        with open(gen_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            existing = obj
    except (OSError, json.JSONDecodeError):
        existing = {}

    bos_token_id = None
    eos_token_id = None
    config_path = os.path.join(output_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            bos_token_id = cfg.get("bos_token_id")
            eos_token_id = cfg.get("eos_token_id")
    except (OSError, json.JSONDecodeError):
        pass

    if bos_token_id is None:
        bos_token_id = existing.get("bos_token_id")
    if eos_token_id is None:
        eos_token_id = existing.get("eos_token_id")

    gen_cfg: Dict[str, Any] = {
        "_from_model_config": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }
    if "transformers_version" in existing:
        gen_cfg["transformers_version"] = existing["transformers_version"]

    _write_json_file(gen_path, gen_cfg, indent=2, sort_keys=True, newline=True)


def _safetensors_total_tensor_bytes(path: str) -> int:
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"Invalid safetensors header: {path}")
        hdr = json.loads(header)

    total = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        start, end = v["data_offsets"]
        total += end - start
    return total


def _patch_exported_safetensors_index_json(output_dir: str) -> None:
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    if not isinstance(idx, dict):
        return
    metadata = idx.get("metadata")
    if not isinstance(metadata, dict):
        return

    shard_files = sorted(
        fn
        for fn in os.listdir(output_dir)
        if fn.startswith("model-") and fn.endswith(".safetensors")
    )
    if not shard_files:
        return

    total_tensor_bytes = 0
    for fn in shard_files:
        total_tensor_bytes += _safetensors_total_tensor_bytes(
            os.path.join(output_dir, fn)
        )

    if metadata.get("total_size") == total_tensor_bytes:
        metadata["total_size"] = (
            total_tensor_bytes + _LEGACY_SAFETENSORS_INDEX_TOTAL_SIZE_BIAS_BYTES
        )
        idx["metadata"] = metadata
        _write_json_file(index_path, idx, indent=2, sort_keys=True, newline=True)


def _export_reference_processor_assets(output_dir: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_py = os.path.join(script_dir, "processing_moss_ttsd_with_codec.py")
    dst_py = os.path.join(output_dir, "processing_moss_ttsd_with_codec.py")
    shutil.copyfile(src_py, dst_py)

    preprocessor_cfg = {
        "processor_class": _PROCESSOR_FQN,
        "auto_map": {"AutoProcessor": _PROCESSOR_FQN},
    }
    _write_json_file(
        os.path.join(output_dir, "preprocessor_config.json"),
        preprocessor_cfg,
        indent=2,
        sort_keys=False,
        newline=False,
    )


def _export_reference_tokenizer_assets(model_path: str, output_dir: str) -> None:
    files_to_copy = [
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
    ]
    for fn in files_to_copy:
        src = os.path.join(model_path, fn)
        dst = os.path.join(output_dir, fn)
        shutil.copyfile(src, dst)

    tok_cfg_path = os.path.join(model_path, "tokenizer_config.json")
    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    tok_cfg["processor_class"] = _PROCESSOR_FQN
    for k in ["text_kwargs", "audio_kwargs", "common_kwargs"]:
        if k in tok_cfg:
            del tok_cfg[k]

    tok_cfg["text_kwargs"] = {"pad_token_id": 151643}
    tok_cfg["audio_kwargs"] = {
        "max_channels": 8,
        "audio_pad_token_id": 1024,
        "silence_duration": 0.0,
        "input_sample_rate": 16000,
        "encoder_downsample_rate": 1280,
        "speech_token_range": [151665, 152689],
        "audio_bos_token": "<|begin_of_speech|>",
        "audio_eos_token": "<|end_of_speech|>",
        "hop_length": 160,
    }
    tok_cfg["common_kwargs"] = {
        "return_tensors": "pt",
        "padding": True,
        "use_normalize": True,
    }

    with open(
        os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    with tempfile.TemporaryDirectory(prefix="moss_ttsd_tok_") as tmpdir:
        tokenizer.save_pretrained(tmpdir, save_jinja_files=False)
        shutil.copyfile(
            os.path.join(tmpdir, "special_tokens_map.json"),
            os.path.join(output_dir, "special_tokens_map.json"),
        )


@torch.no_grad()
def merge_and_export(
    model_path: str,
    codec_path: str,
    output_dir: str,
    *,
    dtype: str = "auto",
    device: str = "cpu",
    save_codec_subdir: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    torch_dtype = _dtype_from_arg(dtype)

    moss_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    codec_config = XYTokenizerConfig.from_pretrained(codec_path)
    codec_model = XYTokenizer.from_pretrained(
        codec_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )

    moss_model.config.codec_config = codec_config.to_dict()
    moss_model.config.codec_model_type = getattr(
        codec_config, "model_type", "xy_tokenizer"
    )
    moss_model.config.codec_model_name_or_path = codec_path
    moss_model.config.architectures = ["MOSSTTSDWithCodec"]

    moss_model.codec_model = codec_model
    moss_model.to(device)
    moss_model.eval()

    if hasattr(moss_model, "tie_weights"):
        moss_model.tie_weights()

    moss_model.save_pretrained(
        output_dir, max_shard_size="5GB", safe_serialization=True
    )

    _cleanup_transformers_saved_code_files(output_dir)

    _patch_exported_config_json(output_dir, dtype=dtype)
    _patch_exported_generation_config(output_dir)
    _cleanup_chat_template_files(output_dir)

    _export_reference_processor_assets(output_dir)
    _export_reference_tokenizer_assets(model_path, output_dir)

    _patch_exported_safetensors_index_json(output_dir)

    if save_codec_subdir:
        codec_out = os.path.join(output_dir, "codec")
        codec_model.save_pretrained(codec_out)

    meta: Dict[str, Any] = {
        "moss_path": os.path.abspath(model_path),
        "codec_path": os.path.abspath(codec_path),
        "dtype": dtype,
        "device": device,
        "note": "Fused MossTTSDForCausalLM + XYTokenizer into MOSSTTSDWithCodec.",
    }
    _write_json_file(
        os.path.join(output_dir, "merge_meta.json"),
        meta,
        indent=2,
        sort_keys=False,
        newline=False,
    )

    print(f"Saved fused model to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse MOSS-TTSD and XYTokenizer into one exported HF model."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./MOSS-TTSD-v0.7",
        help="Path to MOSS-TTSD model directory.",
    )
    parser.add_argument(
        "--codec-path",
        type=str,
        default="./MOSS_TTSD_Tokenizer_hf",
        help="Path to XYTokenizer model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./MOSS-TTSD-v0.7-with-codec",
        help="Where to save the fused model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Load/export dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Where to place the fused model before saving (cpu/cuda/... ).",
    )

    args = parser.parse_args()
    merge_and_export(
        args.model_path,
        args.codec_path,
        args.output_dir,
        dtype=args.dtype,
        device=args.device,
    )


if __name__ == "__main__":
    main()
