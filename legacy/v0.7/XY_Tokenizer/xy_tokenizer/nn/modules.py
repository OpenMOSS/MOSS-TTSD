import copy
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
import scipy
import torch
import torch.distributed
from torch import nn, view_as_complex, view_as_real
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz
from transformers import WhisperModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput


# Define function to generate positional embeddings using sine and cosine functions to represent sequence position information
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoidal waves for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# Generate sequence mask to distinguish valid sequence and padding parts
def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(
        bsz, tgt_len, 1
    )
    return sequence_mask


# Define RMSNorm layer for normalizing hidden states and stabilizing training process
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


# Modified variable-length attention mechanism, supporting FP32 with unified interface
class VarLenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, dropout=0.0):
        """
        Initialize variable-length attention module.

        Parameters:
            embed_dim (int): Embedding dimension (model's hidden dimension)
            num_heads (int): Number of attention heads
            causal (bool): Whether to enable causal attention (only attend to current and previous positions)
            dropout (float): Attention dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim**-0.5  # Scaling factor

        # Linear projection layers for Q, K, V and output
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _create_attention_mask(self, seq_len, max_len, device, dtype):
        """
        Create attention mask supporting variable-length sequences and causality.

        Parameters:
            seq_len (torch.Tensor): Sequence length for each sample, shape [bsz]
            max_len (int): Maximum sequence length in the batch
            device: Device for tensor creation
            dtype: Data type for mask values

        Returns:
            mask (torch.Tensor): Attention mask, shape [bsz, 1, max_len, max_len], invalid positions set to minimum value
        """
        bsz = seq_len.size(0)
        # Initialize mask as 1 (valid positions)
        mask = torch.ones(bsz, 1, max_len, max_len, device=device, dtype=dtype)

        # Generate sequence indices
        seq_indices = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
        seq_len_expanded = seq_len.unsqueeze(1)  # [bsz, 1]

        # Mark valid positions (less than seq_len)
        valid_mask = seq_indices < seq_len_expanded.unsqueeze(-1)  # [bsz, 1, max_len]
        mask = mask * (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(3)).to(
            dtype
        )  # [bsz, 1, max_len, max_len]

        # If causal attention, add upper triangular mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(max_len, max_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            mask = mask * (~causal_mask.unsqueeze(0).unsqueeze(1)).to(
                dtype
            )  # Keep only lower triangular part

        # Set invalid positions (0) to dtype's minimum value
        mask = (
            mask + (1.0 - mask) * torch.finfo(dtype).min
        )  # Valid positions unchanged, invalid positions to minimum value
        return mask

    def forward(
        self, hidden_states: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation, input and output are [bsz, max_len, embed_dim].

        Parameters:
            hidden_states (torch.Tensor): Input hidden states, shape [bsz, max_len, embed_dim]
            seq_len (torch.Tensor): Sequence length for each sample, shape [bsz]

        Returns:
            attn_output (torch.Tensor): Attention output, shape [bsz, max_len, embed_dim]
        """
        bsz, max_len, _ = hidden_states.size()

        # Project to Q, K, V
        query = self.q_proj(hidden_states) * self.scaling  # [bsz, max_len, embed_dim]
        key = self.k_proj(hidden_states)  # [bsz, max_len, embed_dim]
        value = self.v_proj(hidden_states)  # [bsz, max_len, embed_dim]

        # Reshape to multi-head form
        query = query.view(bsz, max_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [bsz, num_heads, max_len, head_dim]
        key = key.view(bsz, max_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [bsz, num_heads, max_len, head_dim]
        value = value.view(bsz, max_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [bsz, num_heads, max_len, head_dim]

        # Calculate attention scores
        attn_scores = torch.matmul(
            query, key.transpose(-1, -2)
        )  # [bsz, num_heads, max_len, max_len]

        # Generate attention mask
        attn_mask = self._create_attention_mask(
            seq_len, max_len, hidden_states.device, attn_scores.dtype
        )  # [bsz, 1, max_len, max_len]
        # Apply mask (additive form, consistent with HubertEncoder)
        attn_scores = (
            attn_scores + attn_mask
        )  # Invalid positions set to very small value

        # Softmax calculate attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # [bsz, num_heads, max_len, max_len]
        attn_weights = self.dropout(attn_weights)

        # Calculate attention output
        attn_output = torch.matmul(
            attn_weights, value
        )  # [bsz, num_heads, max_len, head_dim]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, max_len, self.embed_dim)
        )  # [bsz, max_len, embed_dim]

        # Output projection
        attn_output = self.out_proj(attn_output)  # [bsz, max_len, embed_dim]

        return attn_output


# Define Transformer layer containing attention mechanism and feedforward network for feature extraction and transformation
class OmniWhisperTransformerLayer(nn.Module):
    def __init__(
        self,
        activation_function="gelu",
        d_model=1280,
        attention_heads=20,
        ffn_dim=5120,
        causal=False,
        ln_type="LayerNorm",
        attn_type="varlen",
    ):
        super().__init__()
        self.embed_dim = d_model
        # Only keep varlen attention mechanism
        if attn_type != "varlen":
            raise ValueError(
                f"Unknown attn_type: {attn_type}. Only 'varlen' is supported."
            )
        self.self_attn = VarLenAttention(self.embed_dim, attention_heads, causal)
        if ln_type == "LayerNorm":
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.self_attn_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        if ln_type == "LayerNorm":
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.final_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

    def forward(
        self, hidden_states: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states  # [bsz, max_len, embed_dim]
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # from torch.cuda.amp import autocast
        # print(f"{residual.dtype = }")
        # print(f"Autocast enabled: {torch.is_autocast_enabled():}")
        # print(f"after layernorm {hidden_states.dtype = }")
        hidden_states = self.self_attn(
            hidden_states, seq_len
        )  # [bsz, max_len, embed_dim]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if (
            hidden_states.dtype == torch.float16
            or hidden_states.dtype == torch.bfloat16
        ) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


# Define audio encoder to convert input audio features to hidden state representation
class OmniAudioEncoder(nn.Module):
    def __init__(
        self,
        num_mel_bins=128,  # Input feature Mel band number, usually the dimension of Mel spectrogram
        sampling_rate=16000,  # Audio sampling rate, unit Hz
        hop_length=160,  # Frame shift length (sample number) when calculating Mel spectrogram
        stride_size=2,  # Convolution layer step, used for downsampling
        kernel_size=3,  # Convolution kernel size, controlling receptive field
        d_model=1280,  # Model's hidden state dimension (embedding dimension)
        scale_embedding=True,  # Whether to scale embedding (usually used for stabilizing training)
        max_audio_seconds=30,  # Maximum audio duration supported (seconds)
        encoder_layers=32,  # Transformer encoder layer number
        encoder_attention_heads=20,  # Attention head number for each Transformer layer
        encoder_ffn_dim=5120,  # Intermediate dimension for feedforward network
        activation_function="gelu",  # Activation function type, default GELU
        attn_type="varlen",  # New parameter, select attention mechanism type
    ):
        super().__init__()
        # Calculate maximum sequence length: Convert sampling rate to frame number after considering downsampling step
        self.max_source_positions = (
            max_audio_seconds * sampling_rate // hop_length
        ) // stride_size
        # Embedding scaling factor, if enabled sqrt(d_model), otherwise 1.0
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins = num_mel_bins  # Save Mel band number
        self.d_model = d_model  # Save hidden state dimension
        self.stride_size = stride_size

        # First convolution layer: Convert Mel spectrogram features (num_mel_bins) to hidden dimension (d_model)
        self.conv1 = nn.Conv1d(
            num_mel_bins, d_model, kernel_size=kernel_size, padding=1
        )
        # Second convolution layer: Apply downsampling with stride_size
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=stride_size, padding=1
        )

        # Register positional embedding buffer, using sine function to generate, shape (max_source_positions, d_model)
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )

        # Create Transformer encoder layer list, each layer contains attention mechanism and feedforward network
        self.layers = nn.ModuleList(
            [
                OmniWhisperTransformerLayer(
                    activation_function=activation_function,
                    d_model=d_model,
                    attention_heads=encoder_attention_heads,
                    ffn_dim=encoder_ffn_dim,
                    causal=False,  # Encoder does not need causal attention
                    attn_type=attn_type,  # Pass attention type
                )
                for _ in range(encoder_layers)
            ]
        )

        # Last layer normalization for stable output
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_features, input_length, output_hidden_states=False):
        """
        Forward propagation function to convert input audio features to hidden state representation

        Parameters:
            input_features (torch.Tensor): Input Mel spectrogram features, shape [bsz, num_mel_bins, seq_len]
            input_length (torch.Tensor): Input sequence length for each sample, shape [bsz]
            output_hidden_states (bool, optional): Whether to return hidden states for each layer, default False

        Returns:
            if output_hidden_states is False:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, d_model, tgt_len]
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
            else:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, d_model, tgt_len]
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
                hidden_states_all_layers (tuple): Tuple containing hidden states for each layer, including initial input
        """
        # Ensure input feature data type consistent with convolution layer weights
        input_features = input_features.to(self.conv1.weight.dtype)  # (B, D, T)

        # First layer convolution + GELU activation, Convert Mel spectrogram to hidden states
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (B, D, T)

        # Second layer convolution + GELU activation, Apply downsampling with stride_size
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (B, D, T)

        # Calculate output length: Result after downsampling with stride_size
        output_length = (input_length // self.stride_size).long()  # (B,)

        # Adjust dimension order to [bsz, seq_len, d_model] for Transformer input
        hidden_states = inputs_embeds.permute(0, 2, 1)  # (B, T, D)

        # Get batch size and target sequence length
        bsz, tgt_len, _ = hidden_states.size()

        # According to current sequence length, take or use complete positional embedding
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]
        else:
            current_positional_embedding = self.positional_embedding

        # Add input embedding to positional embedding, convert to float to avoid precision issues
        hidden_states = (
            hidden_states.to(torch.float32) + current_positional_embedding
        ).to(hidden_states.dtype)

        # Generate sequence mask for processing variable-length sequence
        attention_mask = get_sequence_mask(
            hidden_states, output_length
        )  # [bsz, tgt_len, 1]

        # Initialize hidden states list for storing output for each layer (if needed)
        hidden_states_all_layers = () if output_hidden_states else None

        # Process hidden states through Transformer encoder layer by layer
        for encoder_layer in self.layers:
            if output_hidden_states:
                hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states, output_length
            )  # [bsz, tgt_len, d_model]

        # Normalize hidden states
        hidden_states = self.layer_norm(hidden_states)  # [bsz, tgt_len, d_model]
        if output_hidden_states:
            hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)

        # Use mask to zero out padding parts and ensure output only retains valid data
        hidden_states = torch.where(
            attention_mask, hidden_states, 0
        )  # [bsz, tgt_len, d_model]
        hidden_states = hidden_states.transpose(1, 2)  # [bsz, d_model, tgt_len]

        if not output_hidden_states:
            return hidden_states, output_length
        else:
            return hidden_states, output_length, hidden_states_all_layers


# Define audio decoder to convert hidden states to Mel spectrogram
class OmniAudioDecoder(nn.Module):
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
        attn_type="varlen",  # New parameter, select attention mechanism type
    ):
        super().__init__()
        self.max_source_positions = (
            max_audio_seconds * sampling_rate // hop_length
        ) // stride_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.stride_size = stride_size

        # Correct transpose convolution layer to ensure output length close to stride_size times
        self.deconv1 = nn.ConvTranspose1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=stride_size,
            padding=0,  # Do not fill input side
            output_padding=0,  # Can be adjusted to precisely control length
        )
        self.deconv2 = nn.ConvTranspose1d(
            d_model,
            num_mel_bins,
            kernel_size=kernel_size,
            stride=1,  # Only convert channels, do not change length
            padding=0,
        )

        # Positional embedding remains consistent
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )  # (T, D)

        # Transformer decoder layer
        self.layers = nn.ModuleList(
            [
                OmniWhisperTransformerLayer(
                    activation_function=activation_function,
                    d_model=d_model,
                    attention_heads=decoder_attention_heads,
                    ffn_dim=decoder_ffn_dim,
                    causal=False,  # Decoder uses causal attention
                    attn_type=attn_type,  # Pass attention type
                )
                for _ in range(decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, input_length):  # (B, D, T)
        # Input is hidden state output from encoder
        hidden_states = hidden_states.transpose(1, 2)  # (B, T, D)
        bsz, tgt_len, _ = hidden_states.size()

        # Add positional embedding
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[:tgt_len]  # (T, D)
        else:
            current_positional_embedding = self.positional_embedding
        hidden_states = (
            hidden_states.to(torch.float32) + current_positional_embedding
        ).to(
            hidden_states.dtype
        )  # (B, T, D)

        # Generate sequence mask
        attention_mask = get_sequence_mask(
            hidden_states, input_length
        )  # [bsz, tgt_len, 1]

        # Process through decoder layer
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, input_length
            )  # [bsz, tgt_len, d_model]

        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)  # [bsz, tgt_len, d_model]

        # Use mask to zero out padding parts
        hidden_states = torch.where(
            attention_mask, hidden_states, 0
        )  # [bsz, tgt_len, d_model]

        # Process through transpose convolution layer to reconstruct audio features
        hidden_states = hidden_states.permute(0, 2, 1)  # (B, D, T)
        output_features = nn.functional.gelu(self.deconv1(hidden_states))  # (B, D, T)
        output_features = nn.functional.gelu(self.deconv2(output_features))  # (B, D, T)

        # If strictly stride_size times length is needed, can trim extra parts
        expected_length = tgt_len * self.stride_size
        if output_features.size(2) > expected_length:
            output_features = output_features[:, :, :expected_length]

        output_length = input_length * self.stride_size
        # Output shape: [bsz, num_mel_bins, seq_len]
        return output_features, output_length


# The following part remains unchanged
class ResidualDownConv(nn.Module):
    def __init__(self, d_model=1280, avg_pooler=4):
        """
        Downsampling module containing residual connection and convolution operation

        Parameters:
            d_model (int): Input and output hidden dimension
            avg_pooler (int): Downsampling factor (convolution step)
        """
        super().__init__()
        self.d_model = d_model
        self.avg_pooler = avg_pooler
        self.intermediate_dim = d_model * avg_pooler

        # Convolution layer for downsampling
        self.gate_proj = nn.Conv1d(
            d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )
        self.up_proj = nn.Conv1d(
            d_model, self.intermediate_dim, avg_pooler, avg_pooler, bias=False
        )

        # Downsampled linear projection
        self.down_proj = nn.Linear(
            self.intermediate_dim, self.intermediate_dim, bias=False
        )

        # Activation function and layer normalization
        self.act_fn = ACT2FN["silu"]
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)

    def forward(self, x, input_length):
        """
        Forward propagation, execute downsampling and residual processing

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, D, T]

        Returns:
            res (torch.Tensor): Downsampled feature, shape [B, intermediate_dim, seq_len // avg_pooler]
            valid_mask (torch.Tensor): Valid sequence mask
        """
        output_length = input_length // self.avg_pooler
        x = x.transpose(1, 2)  # (B, T, D)
        batch_size, seq_len, _ = x.shape  # (B, T, D)
        if seq_len % self.avg_pooler != 0:
            pad_size = self.avg_pooler - seq_len % self.avg_pooler
            x = F.pad(x, (0, pad_size), "constant", 0)

        xt = x.permute(0, 2, 1)  # (B, D, T)
        g = self.gate_proj(xt).permute(0, 2, 1)  # (B, T, D)
        u = self.up_proj(xt).permute(0, 2, 1)  # (B, T, D)
        x = x.reshape(batch_size, -1, self.intermediate_dim)  # (B, T, D)

        c = self.down_proj(self.act_fn(g) * u)  # (B, T, D)
        res = self.layer_norm(c + x)  # (B, T, D)
        res = res.transpose(1, 2)  # (B, D, T)
        return res, output_length  # (B, D, T)


class UpConv(nn.Module):
    def __init__(self, d_model=1280, stride=4):
        """
        Simple upsampling module using transpose convolution

        Parameters:
            d_model (int): Input and output hidden dimension
            stride (int): Upsampling factor (transpose convolution step)
        """
        super().__init__()
        self.d_model = d_model
        self.stride = stride

        # Simple transpose convolution layer to keep channel number consistent
        self.up_conv = nn.ConvTranspose1d(
            self.stride * d_model,
            d_model,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

    def forward(self, x, input_length):
        """
        Forward propagation, execute upsampling

        Parameters:
            x (torch.Tensor): Input tensor, shape [B, D * stride, T]

        Returns:
            res (torch.Tensor): Upsampled feature, shape [B, D, T * stride]
        """
        # Directly apply transpose convolution
        res = self.up_conv(x)
        output_length = input_length * self.stride
        return res, output_length


# Define Transformer encoder containing multiple Transformer layers for feature extraction and transformation
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=1280,  # Input feature dimension
        d_model=1280,  # Model's hidden state dimension (embedding dimension)
        output_dim=1280,  # Output feature dimension
        max_source_positions=1500,  # Maximum sequence length for positional embedding
        encoder_layers=32,  # Transformer encoder layer number
        encoder_attention_heads=20,  # Attention head number for each Transformer layer
        encoder_ffn_dim=5120,  # Intermediate dimension for feedforward network
        activation_function="gelu",  # Activation function type, default GELU
        attn_type="varlen",  # Attention mechanism type
    ):
        super().__init__()
        self.input_dim = input_dim  # Save input dimension
        self.d_model = d_model  # Save hidden state dimension
        self.output_dim = output_dim  # Save output dimension
        self.max_source_positions = max_source_positions  # Save maximum sequence length

        # If input dimension and model dimension are not consistent, add input projection layer
        if input_dim != d_model:
            self.proj = nn.Linear(input_dim, d_model, bias=True)
        else:
            self.proj = None  # No need for input projection layer

        # Register positional embedding buffer, using sine function to generate, shape (max_source_positions, d_model)
        self.register_buffer(
            "positional_embedding", sinusoids(self.max_source_positions, d_model)
        )

        # Create Transformer encoder layer list, each layer contains attention mechanism and feedforward network
        self.layers = nn.ModuleList(
            [
                OmniWhisperTransformerLayer(
                    activation_function=activation_function,
                    d_model=d_model,
                    attention_heads=encoder_attention_heads,
                    ffn_dim=encoder_ffn_dim,
                    causal=False,  # Encoder does not need causal attention
                    attn_type=attn_type,  # Pass attention type
                )
                for _ in range(encoder_layers)
            ]
        )

        # Last layer normalization for stable output
        self.layer_norm = nn.LayerNorm(d_model)

        # If output dimension and model dimension are not consistent, add output projection layer
        if output_dim != d_model:
            self.out_proj = nn.Linear(d_model, output_dim, bias=True)
        else:
            self.out_proj = None  # No need for output projection layer

    def forward(
        self,
        input_features: torch.Tensor,
        input_length: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        Forward propagation function to convert input features through Transformer layer to hidden state representation

        Parameters:
            input_features (torch.Tensor): Input features, shape [bsz, input_dim, seq_len] (B, input_dim, T)
            input_length (torch.Tensor): Input sequence length for each sample, shape [bsz]
            output_hidden_states (bool, optional): Whether to return hidden states for each layer, default False

        Returns:
            if output_hidden_states is False:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, output_dim, seq_len] (B, output_dim, T)
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
            else:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, output_dim, seq_len] (B, output_dim, T)
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
                hidden_states_all_layers (tuple): Tuple containing hidden states for each layer, each shape [bsz, seq_len, d_model]
        """
        # Output length is the same as input length, Transformer does not change sequence length
        output_length = input_length.long()  # [bsz]

        # If there is input projection layer, map input features from input_dim to d_model
        if self.proj is not None:
            hidden_states = self.proj(input_features.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # [bsz, d_model, seq_len] (B, D, T)
        else:
            hidden_states = input_features  # [bsz, d_model, seq_len] (B, D, T)

        # Adjust input dimension order to [bsz, seq_len, d_model] for Transformer input
        hidden_states = hidden_states.permute(
            0, 2, 1
        )  # [bsz, seq_len, d_model] (B, T, D)

        # Get batch size and target sequence length
        bsz, tgt_len, _ = hidden_states.size()

        # According to current sequence length, take or use complete positional embedding
        if tgt_len < self.positional_embedding.shape[0]:
            current_positional_embedding = self.positional_embedding[
                :tgt_len
            ]  # [tgt_len, d_model]
        else:
            current_positional_embedding = (
                self.positional_embedding
            )  # [max_source_positions, d_model]

        # Add input features to positional embedding, convert to float to avoid precision issues
        hidden_states = (
            hidden_states.to(torch.float32) + current_positional_embedding
        ).to(
            hidden_states.dtype
        )  # [bsz, seq_len, d_model]

        # Generate sequence mask for processing variable-length sequence
        attention_mask = get_sequence_mask(
            hidden_states, output_length
        )  # [bsz, tgt_len, 1]

        # Initialize hidden states list for storing output for each layer (if needed)
        hidden_states_all_layers = () if output_hidden_states else None

        # Process hidden states through Transformer encoder layer by layer
        for encoder_layer in self.layers:
            if output_hidden_states:
                hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)
            hidden_states = encoder_layer(
                hidden_states, output_length
            )  # [bsz, seq_len, d_model]

        # Normalize hidden states
        hidden_states = self.layer_norm(hidden_states)  # [bsz, seq_len, d_model]
        if output_hidden_states:
            hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)

        # Use mask to zero out padding parts and ensure output only retains valid data
        hidden_states = torch.where(
            attention_mask, hidden_states, 0
        )  # [bsz, seq_len, d_model]

        # Adjust dimension order to [bsz, d_model, seq_len]
        hidden_states = hidden_states.transpose(
            1, 2
        )  # [bsz, d_model, seq_len] (B, D, T)

        # If there is output projection layer, map hidden states from d_model to output_dim
        if self.out_proj is not None:
            hidden_states = self.out_proj(hidden_states.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # [bsz, output_dim, seq_len] (B, output_dim, T)

        if not output_hidden_states:
            return hidden_states, output_length
        else:
            return hidden_states, output_length, hidden_states_all_layers


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center=True,
    ):
        super().__init__()
        self.center = center
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T * hop_length)

        if not self.center:
            pad = self.win_length - self.hop_length
            x = torch.nn.functional.pad(x, (pad // 2, pad // 2), mode="reflect")

        stft_spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=False,
        )  # (B, n_fft // 2 + 1, T, 2)

        rea = stft_spec[:, :, :, 0]  # (B, n_fft // 2 + 1, T, 2)
        imag = stft_spec[:, :, :, 1]  # (B, n_fft // 2 + 1, T, 2)

        log_mag = torch.log(
            torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2))) + 1e-5
        )  # (B, n_fft // 2 + 1, T)
        phase = torch.atan2(imag, rea)  # (B, n_fft // 2 + 1, T)

        return log_mag, phase


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class MDCT(nn.Module):
    """
    Modified Discrete Cosine Transform (MDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(-1j * torch.pi * torch.arange(frame_len) / frame_len)
        post_twiddle = torch.exp(-1j * torch.pi * n0 * (torch.arange(N) + 0.5) / N)
        # view_as_real: NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply the Modified Discrete Cosine Transform (MDCT) to the input audio.

        Args:
            audio (Tensor): Input audio waveform of shape (B, T), where B is the batch size
                and T is the length of the audio.

        Returns:
            Tensor: MDCT coefficients of shape (B, L, N), where L is the number of output frames
                and N is the number of frequency bins.
        """
        if self.padding == "center":
            audio = torch.nn.functional.pad(
                audio, (self.frame_len // 2, self.frame_len // 2)
            )
        elif self.padding == "same":
            # hop_length is 1/2 frame_len
            audio = torch.nn.functional.pad(
                audio, (self.frame_len // 4, self.frame_len // 4)
            )
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        x = audio.unfold(-1, self.frame_len, self.frame_len // 2)
        N = self.frame_len // 2
        x = x * self.window.expand(x.shape)
        X = torch.fft.fft(
            x * view_as_complex(self.pre_twiddle).expand(x.shape), dim=-1
        )[..., :N]
        res = X * view_as_complex(self.post_twiddle).expand(X.shape) * np.sqrt(1 / N)
        return torch.real(res) * np.sqrt(2)


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(
            Y * view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1
        )
        y = (
            torch.real(y * view_as_complex(self.post_twiddle).expand(y.shape))
            * np.sqrt(N)
            * np.sqrt(2)
        )
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        original_dtype = x.dtype
        S = mag.float() * (x.float() + 1j * y.float())
        audio = self.istft(S)
        audio = audio.to(original_dtype)
        return audio


class IMDCTSymExpHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with symmetric exponential function

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        sample_rate: Optional[int] = None,
        clip_audio: bool = False,
    ):
        super().__init__()
        out_dim = mdct_frame_len // 2
        self.out = nn.Linear(dim, out_dim)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)
        self.clip_audio = clip_audio

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.out.weight.mul_(scale.view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        x = symexp(x)
        x = torch.clip(
            x, min=-1e2, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio


class IMDCTCosHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with parametrizing MDCT = exp(m) · cos(p)

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        clip_audio: bool = False,
    ):
        super().__init__()
        self.clip_audio = clip_audio
        self.out = nn.Linear(dim, mdct_frame_len)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTCosHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x)
        m, p = x.chunk(2, dim=2)
        m = torch.exp(m).clip(
            max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(m * torch.cos(p))
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)
        return audio


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.shift = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get("bandwidth_id", None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(
            nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[
                ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x


class Vocos(nn.Module):
    def __init__(
        self,
        input_channels: int = 128,
        dim: int = 512,
        intermediate_dim: int = 4096,
        num_layers: int = 30,
        n_fft: int = 640,
        hop_size: int = 160,
        padding: str = "same",
        adanorm_num_embeddings=None,
    ):
        super().__init__()

        self.backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            adanorm_num_embeddings=adanorm_num_embeddings,
        )
        self.head = ISTFTHead(dim, n_fft, hop_size, padding)
        self.hop_size = hop_size

    def forward(self, x, input_length):
        x = self.backbone(x)
        x = self.head(x)
        output_length = input_length * self.hop_size
        return x[:, None, :], output_length
