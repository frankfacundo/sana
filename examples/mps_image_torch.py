#!/usr/bin/env python
"""Generate a Sana image with either parity or pure-torch execution.

The default parity backend mirrors ``examples/mps_image_diffusers.py`` so both
scripts produce the same output for the same arguments.  The optional pure
backend intentionally does not call ``diffusers.SanaPipeline``.  It mirrors the
pure-torch style used by the local Qwen implementation: config is read from the
checkpoint folders, modules are ordinary ``torch.nn.Module`` classes, weights
are copied from safetensors, and the denoising loop is written out directly.

The tokenizer is loaded from the model's ``tokenizer.json`` via Hugging Face's
``tokenizers`` package.  Everything else in the inference path is torch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


LOCAL_SANA_600M_512_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/Sana_600M_512px_diffusers"
LOCAL_SANA15_16B_1024_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
DEFAULT_MODEL_PATH = LOCAL_SANA15_16B_1024_MODEL

DEFAULT_COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ns_from_dict(data: dict) -> SimpleNamespace:
    return SimpleNamespace(**data)


def iter_safetensor_shards(model_dir: Path, *, index_name: str = "model.safetensors.index.json") -> Iterable[Path]:
    index_path = model_dir / index_name
    if index_path.exists():
        data = read_json(index_path)
        for shard in sorted(set(data["weight_map"].values())):
            yield model_dir / shard
        return

    single = model_dir / "model.safetensors"
    if single.exists():
        yield single
        return

    diffusion_single = model_dir / "diffusion_pytorch_model.safetensors"
    if diffusion_single.exists():
        yield diffusion_single
        return

    for shard in sorted(model_dir.glob("*.safetensors")):
        yield shard


def load_safetensors_into_module(
    module: nn.Module,
    model_dir: Path,
    *,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> dict[str, list[str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to load local checkpoints") from exc

    wanted = set(module.state_dict().keys())
    loaded: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []

    for shard in iter_safetensor_shards(model_dir):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key not in wanted:
                    unexpected.append(key)
                    continue
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                loaded[key] = tensor

    missing = sorted(wanted - set(loaded))
    if strict and missing:
        raise RuntimeError(f"Missing {len(missing)} keys while loading {model_dir}: {missing[:8]}")

    model_state = module.state_dict()
    for key, value in loaded.items():
        if model_state[key].shape != value.shape:
            raise ValueError(f"Shape mismatch for {key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}")

    has_meta_tensors = any(tensor.is_meta for tensor in module.state_dict().values())
    module.load_state_dict(loaded, strict=False, assign=has_meta_tensors)
    if strict:
        unexpected = [key for key in unexpected if key in wanted]
    return {"missing": missing, "unexpected": unexpected, "loaded": sorted(loaded)}


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class GemmaTokenizer:
    def __init__(self, tokenizer_dir: Path, pad_token_id: int = 0):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("tokenizers is required to read tokenizer/tokenizer.json") from exc

        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self.pad_token_id = pad_token_id

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(
        self,
        texts: str | list[str],
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=True)
            if truncation:
                ids = ids[:max_length]
            if padding == "max_length":
                mask = [1] * len(ids)
                if len(ids) < max_length:
                    pad_len = max_length - len(ids)
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
            else:
                mask = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class Gemma2TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    attention_dropout: float
    hidden_activation: str
    query_pre_attn_scalar: int
    sliding_window: int
    attn_logit_softcapping: float | None
    pad_token_id: int
    layer_types: list[str]

    @classmethod
    def from_json(cls, path: Path) -> "Gemma2TextConfig":
        data = read_json(path)
        num_layers = int(data["num_hidden_layers"])
        layer_types = data.get("layer_types")
        if layer_types is None:
            layer_types = ["sliding_attention" if (i + 1) % 2 else "full_attention" for i in range(num_layers)]
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=num_layers,
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            attention_bias=bool(data["attention_bias"]),
            attention_dropout=float(data["attention_dropout"]),
            hidden_activation=str(data.get("hidden_activation", data.get("hidden_act", "gelu_pytorch_tanh"))),
            query_pre_attn_scalar=int(data.get("query_pre_attn_scalar", data.get("head_dim", 256))),
            sliding_window=int(data.get("sliding_window", 4096)),
            attn_logit_softcapping=data.get("attn_logit_softcapping"),
            pad_token_id=int(data.get("pad_token_id", 0)),
            layer_types=layer_types,
        )


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(dtype=x.dtype)


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class Gemma2MLP(nn.Module):
    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported Gemma activation: {config.hidden_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x))


class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        inv_freq = self._build_inv_freq(device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )

    def reset_inv_freq(self, device: torch.device) -> None:
        self.inv_freq = self._build_inv_freq(device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * repeats, seq_len, head_dim)


def build_gemma_causal_mask(
    attention_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    batch, seq_len = attention_mask.shape
    device = attention_mask.device
    i = torch.arange(seq_len, device=device)[:, None]
    j = torch.arange(seq_len, device=device)[None, :]
    allowed = j <= i
    if sliding_window is not None and seq_len > sliding_window:
        allowed = allowed & (j > i - sliding_window)

    mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    mask = mask.masked_fill(~allowed, -10000.0)
    mask = mask[None, None, :, :].expand(batch, 1, seq_len, seq_len).clone()
    key_padding = attention_mask[:, None, None, :].to(torch.bool)
    mask = mask.masked_fill(~key_padding, -10000.0)
    return mask.to(dtype=dtype)


class Gemma2Attention(nn.Module):
    def __init__(self, config: Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask[:, :, :, : key.shape[-2]],
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.o_proj(out)


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma2Attention(config, layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma2Model(nn.Module):
    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Gemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Gemma2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        full_mask = build_gemma_causal_mask(attention_mask, dtype=hidden_states.dtype, sliding_window=None)
        sliding_mask = build_gemma_causal_mask(
            attention_mask,
            dtype=hidden_states.dtype,
            sliding_window=self.config.sliding_window,
        )

        for layer in self.layers:
            mask = sliding_mask if layer.attention_type == "sliding_attention" else full_mask
            hidden_states = layer(hidden_states, position_embeddings, mask)
        return self.norm(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, dim: int | tuple[int, ...], eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.dim = (dim,) if isinstance(dim, int) else tuple(dim)
        self.weight = nn.Parameter(torch.ones(self.dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(self.dim)) if elementwise_affine and bias else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = scale * timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, *, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def forward(self, timestep: torch.Tensor, *, batch_size: int, hidden_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        del batch_size
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PatchEmbed(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        interpolation_scale: int | None = None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self.proj(latent)
        return latent.flatten(2).transpose(1, 2)


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: str | None = None,
        residual_connection: bool = True,
    ):
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True) if norm_type == "rms_norm" else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)
        if self.norm_type == "rms_norm":
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


class SanaModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, scale_shift_table: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None] + temb[:, None].to(scale_shift_table.device)).chunk(2, dim=1)
        return hidden_states * (1 + scale) + shift


class SanaAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        cross_attention_dim: int | None,
        bias: bool,
        out_bias: bool = True,
        qk_norm: str | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.inner_kv_dim = self.inner_dim
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.rescale_output_factor = 1.0
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=out_bias), nn.Dropout(0.0)])
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(self.inner_dim, eps=1e-5)
            self.norm_k = RMSNorm(self.inner_kv_dim, eps=1e-5)
        else:
            raise ValueError(f"Unsupported Sana qk_norm: {qk_norm}")

    def linear_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.transpose(1, 2).unflatten(1, (self.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (self.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (self.heads, -1))
        query = F.relu(query).float()
        key = F.relu(key).float()
        value = value.float()
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2).to(original_dtype)
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states

    def cross_attention(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        sequence_length = encoder_hidden_states.shape[1]
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        attn_mask = None
        if attention_mask is not None:
            current_length = attention_mask.shape[-1]
            if current_length != sequence_length:
                if attention_mask.device.type == "mps":
                    padding_shape = (attention_mask.shape[0], attention_mask.shape[1], sequence_length)
                    padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, padding], dim=2)
                else:
                    attention_mask = F.pad(attention_mask, (0, sequence_length), value=0.0)
            if attention_mask.shape[0] < batch_size * self.heads:
                attention_mask = attention_mask.repeat_interleave(
                    self.heads,
                    dim=0,
                    output_size=attention_mask.shape[0] * self.heads,
                )
            attn_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim).to(query.dtype)
        return self.to_out[1](self.to_out[0](hidden_states))


class SanaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_cross_attention_heads: int,
        cross_attention_head_dim: int,
        cross_attention_dim: int,
        attention_bias: bool,
        norm_eps: float,
        mlp_ratio: float,
        qk_norm: str | None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
            bias=attention_bias,
            qk_norm=qk_norm,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn2 = SanaAttention(
            query_dim=dim,
            heads=num_cross_attention_heads,
            dim_head=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            bias=True,
            out_bias=True,
            qk_norm=qk_norm,
        )
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (norm_hidden_states * (1 + scale_msa) + shift_msa).to(hidden_states.dtype)
        hidden_states = hidden_states + gate_msa * self.attn1.linear_attention(norm_hidden_states)

        hidden_states = hidden_states + self.attn2.cross_attention(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states).flatten(2, 3).permute(0, 2, 1)
        return hidden_states + gate_mlp * ff_output


class SanaTransformer2DModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        inner_dim = int(config.num_attention_heads) * int(config.attention_head_dim)
        self.patch_embed = PatchEmbed(
            height=int(config.sample_size),
            width=int(config.sample_size),
            patch_size=int(config.patch_size),
            in_channels=int(config.in_channels),
            embed_dim=inner_dim,
            interpolation_scale=getattr(config, "interpolation_scale", None),
        )
        if getattr(config, "guidance_embeds", False):
            raise ValueError("This torch example currently supports Sana checkpoints without guidance_embeds.")
        self.time_embed = AdaLayerNormSingle(inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(int(config.caption_channels), inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)
        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=int(config.num_attention_heads),
                    attention_head_dim=int(config.attention_head_dim),
                    num_cross_attention_heads=int(config.num_cross_attention_heads),
                    cross_attention_head_dim=int(config.cross_attention_head_dim),
                    cross_attention_dim=int(config.cross_attention_dim),
                    attention_bias=bool(config.attention_bias),
                    norm_eps=float(config.norm_eps),
                    mlp_ratio=float(config.mlp_ratio),
                    qk_norm=getattr(config, "qk_norm", None),
                )
                for _ in range(int(config.num_layers))
            ]
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = SanaModulatedNorm(inner_dim)
        self.proj_out = nn.Linear(inner_dim, int(config.patch_size) * int(config.patch_size) * int(config.out_channels))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = hidden_states.shape
        patch_size = int(self.config.patch_size)
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size

        if encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        hidden_states = self.patch_embed(hidden_states)
        timestep, embedded_timestep = self.time_embed(
            timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=post_patch_height,
                width=post_patch_width,
            )

        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            patch_size,
            patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        return hidden_states.reshape(batch_size, -1, post_patch_height * patch_size, post_patch_width * patch_size)


class SanaMultiscaleAttentionProjection(nn.Module):
    def __init__(self, in_channels: int, num_attention_heads: int, kernel_size: int):
        super().__init__()
        channels = 3 * in_channels
        self.proj_in = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.proj_in(hidden_states))


class SanaMultiscaleLinearAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_head_dim: int = 32,
        kernel_sizes: tuple[int, ...] = (5,),
        norm_type: str = "rms_norm",
        residual_connection: bool = True,
        eps: float = 1e-15,
    ):
        super().__init__()
        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        num_attention_heads = int(in_channels // attention_head_dim)
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_qkv_multiscale = nn.ModuleList(
            [SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, k) for k in kernel_sizes]
        )
        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

    def apply_linear_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query).float()
        return hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

    def apply_quadratic_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(key.transpose(-1, -2), query).float()
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        return torch.matmul(value, scores.to(value.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        use_linear_attention = height * width > self.attention_head_dim
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3).movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in self.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))
        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            hidden_states = hidden_states.float()
        hidden_states = hidden_states.reshape(batch_size, -1, 3 * self.attention_head_dim, height * width)
        query, key, value = hidden_states.chunk(3, dim=2)
        query = self.nonlinearity(query)
        key = self.nonlinearity(key)
        if use_linear_attention:
            hidden_states = self.apply_linear_attention(query, key, value).to(original_dtype)
        else:
            hidden_states = self.apply_quadratic_attention(query, key, value)
        hidden_states = hidden_states.reshape(batch_size, -1, height, width)
        hidden_states = self.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


def get_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "relu6":
        return nn.ReLU6()
    raise ValueError(f"Unsupported activation: {name}")


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "rms_norm", act_fn: str = "silu"):
        super().__init__()
        if norm_type != "rms_norm":
            raise ValueError(f"Unsupported VAE ResBlock norm: {norm_type}")
        self.norm_type = norm_type
        self.nonlinearity = get_activation(act_fn)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return hidden_states + residual


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attention_head_dim: int = 32,
        qkv_multiscales: tuple[int, ...] = (5,),
        norm_type: str = "rms_norm",
    ):
        super().__init__()
        self.attn = SanaMultiscaleLinearAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            attention_head_dim=attention_head_dim,
            kernel_sizes=qkv_multiscales,
            norm_type=norm_type,
            residual_connection=True,
        )
        self.conv_out = GLUMBConv(in_channels, in_channels, norm_type="rms_norm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.attn(x))


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels
        conv_out_channels = out_channels if interpolate else out_channels * self.factor**2
        self.conv = nn.Conv2d(in_channels, conv_out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = F.pixel_shuffle(self.conv(hidden_states), self.factor)
        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats)
            y = F.pixel_shuffle(y, self.factor)
            x = x + y
        return x


def make_vae_block(
    block_type: str,
    channels: int,
    *,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_multiscales: tuple[int, ...],
) -> nn.Module:
    if block_type == "ResBlock":
        return ResBlock(channels, channels, norm_type=norm_type, act_fn=act_fn)
    if block_type == "EfficientViTBlock":
        return EfficientViTBlock(
            channels,
            attention_head_dim=attention_head_dim,
            qkv_multiscales=qkv_multiscales,
            norm_type=norm_type,
        )
    raise ValueError(f"Unsupported VAE block type: {block_type}")


class DCDecoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        block_out_channels = tuple(config.decoder_block_out_channels)
        layers_per_block = tuple(config.decoder_layers_per_block)
        block_types = tuple(config.decoder_block_types)
        qkv_multiscales = tuple(tuple(x) for x in config.decoder_qkv_multiscales)
        norm_types = tuple(config.decoder_norm_types for _ in block_out_channels) if isinstance(config.decoder_norm_types, str) else tuple(config.decoder_norm_types)
        act_fns = tuple(config.decoder_act_fns for _ in block_out_channels) if isinstance(config.decoder_act_fns, str) else tuple(config.decoder_act_fns)

        self.conv_in = nn.Conv2d(int(config.latent_channels), block_out_channels[-1], 3, 1, 1)
        self.in_shortcut = True
        self.in_shortcut_repeats = block_out_channels[-1] // int(config.latent_channels)
        up_blocks: list[nn.Sequential] = []
        num_blocks = len(block_out_channels)
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            layers: list[nn.Module] = []
            if i < num_blocks - 1 and num_layers > 0:
                layers.append(
                    DCUpBlock2d(
                        block_out_channels[i + 1],
                        out_channel,
                        interpolate=config.upsample_block_type == "interpolate",
                        shortcut=True,
                    )
                )
            for _ in range(num_layers):
                layers.append(
                    make_vae_block(
                        block_types[i],
                        out_channel,
                        attention_head_dim=int(config.attention_head_dim),
                        norm_type=norm_types[i],
                        act_fn=act_fns[i],
                        qkv_multiscales=qkv_multiscales[i],
                    )
                )
            up_blocks.insert(0, nn.Sequential(*layers))
        self.up_blocks = nn.ModuleList(up_blocks)
        channels = block_out_channels[0]
        self.norm_out = RMSNorm(channels, eps=1e-5, elementwise_affine=True, bias=True)
        self.conv_act = get_activation("relu")
        self.conv_out = nn.Conv2d(channels, int(config.in_channels), 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.repeat_interleave(
            self.in_shortcut_repeats,
            dim=1,
            output_size=hidden_states.shape[1] * self.in_shortcut_repeats,
        )
        hidden_states = self.conv_in(hidden_states) + x
        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)
        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        return self.conv_out(hidden_states)


class AutoencoderDCDecoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.decoder = DCDecoder(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DPMSolverMultistepScheduler:
    order = 1

    def __init__(self, config: SimpleNamespace):
        self.config = config
        if config.beta_schedule != "linear":
            raise ValueError(f"Unsupported beta_schedule: {config.beta_schedule}")
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.timesteps = torch.from_numpy(np.linspace(0, config.num_train_timesteps - 1, config.num_train_timesteps, dtype=np.float32)[::-1].copy())
        self.num_inference_steps = None
        self.model_outputs: list[torch.Tensor | None] = [None] * int(config.solver_order)
        self.lower_order_nums = 0
        self._step_index: int | None = None
        self.sigmas = self.sigmas.to("cpu")

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        if not bool(self.config.use_flow_sigmas):
            raise ValueError("This example implements the Sana flow-sigma scheduler path.")
        alphas = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas))[:-1].copy()
        timesteps = (sigmas * self.config.num_train_timesteps).copy()
        sigma_last = 0.0 if self.config.final_sigmas_type == "zero" else sigmas[-1]
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [sigma_last]]).astype(np.float32)).to("cpu")
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * int(self.config.solver_order)
        self.lower_order_nums = 0
        self._step_index = None
        return self.timesteps

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def index_for_timestep(self, timestep: int | torch.Tensor) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        candidates = (self.timesteps == timestep).nonzero()
        if len(candidates) == 0:
            return len(self.timesteps) - 1
        if len(candidates) > 1:
            return candidates[1].item()
        return candidates[0].item()

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.config.use_flow_sigmas):
            return 1 - sigma, sigma
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def convert_model_output(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self.config.algorithm_type != "dpmsolver++":
            raise ValueError(f"Unsupported algorithm_type: {self.config.algorithm_type}")
        if self.config.prediction_type != "flow_prediction":
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")
        sigma_t = self.sigmas[self.step_index].to(device=sample.device, dtype=sample.dtype)
        return sample - sigma_t * model_output

    def dpm_solver_first_order_update(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        sigma_t = sigma_t.to(sample.device, sample.dtype)
        sigma_s = sigma_s.to(sample.device, sample.dtype)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - (torch.log(alpha_s) - torch.log(sigma_s))
        return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

    def multistep_dpm_solver_second_order_update(
        self,
        model_outputs: list[torch.Tensor | None],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        sigma_t = sigma_t.to(sample.device, sample.dtype)
        sigma_s0 = sigma_s0.to(sample.device, sample.dtype)
        sigma_s1 = sigma_s1.to(sample.device, sample.dtype)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        m0, m1 = model_outputs[-1], model_outputs[-2]
        if m0 is None or m1 is None:
            raise RuntimeError("Second-order update requested before two model outputs were available.")
        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        d0 = m0
        d1 = (1.0 / r0) * (m0 - m1)
        if self.config.solver_type != "midpoint":
            raise ValueError(f"Unsupported solver_type: {self.config.solver_type}")
        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * d0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
        )

    def step(self, model_output: torch.Tensor, timestep: int | torch.Tensor, sample: torch.Tensor) -> tuple[torch.Tensor]:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step.")
        if self.step_index is None:
            self._step_index = self.index_for_timestep(timestep)

        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            bool(self.config.euler_at_final)
            or (bool(self.config.lower_order_final) and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            self.step_index == len(self.timesteps) - 2
            and bool(self.config.lower_order_final)
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(int(self.config.solver_order) - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        sample = sample.float()
        if int(self.config.solver_order) == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample)
        elif int(self.config.solver_order) == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample)
        else:
            raise ValueError("Only solver_order <= 2 is implemented for this Sana example.")

        if self.lower_order_nums < int(self.config.solver_order):
            self.lower_order_nums += 1
        self._step_index += 1
        return (prev_sample.to(model_output.dtype),)


def classify_height_width_bin(height: int, width: int, ratios: dict[str, list[float]]) -> tuple[int, int]:
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:
    orig_height, orig_width = samples.shape[2], samples.shape[3]
    if orig_height == new_height and orig_width == new_width:
        return samples
    ratio = max(new_height / orig_height, new_width / orig_width)
    resized_width = int(orig_width * ratio)
    resized_height = int(orig_height * ratio)
    samples = F.interpolate(samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    start_x = (resized_width - new_width) // 2
    start_y = (resized_height - new_height) // 2
    return samples[:, :, start_y : start_y + new_height, start_x : start_x + new_width]


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


def text_preprocessing(text: str | list[str]) -> list[str]:
    if isinstance(text, str):
        text = [text]
    return [str(t).lower().strip() for t in text]


@torch.inference_mode()
def get_gemma_prompt_embeds(
    *,
    tokenizer: GemmaTokenizer,
    text_encoder: Gemma2Model,
    prompt: str | list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    complex_human_instruction: list[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_list = text_preprocessing(prompt)
    if complex_human_instruction:
        chi_prompt = "\n".join(complex_human_instruction)
        prompt_list = [chi_prompt + p for p in prompt_list]
        max_length_all = len(tokenizer.encode(chi_prompt, add_special_tokens=True)) + max_sequence_length - 2
    else:
        max_length_all = max_sequence_length

    text_inputs = tokenizer(prompt_list, padding="max_length", max_length=max_length_all, truncation=True)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    prompt_embeds = text_encoder(input_ids, attention_mask).to(dtype=dtype, device=device)
    return prompt_embeds, attention_mask


@torch.inference_mode()
def encode_prompt(
    *,
    tokenizer: GemmaTokenizer,
    text_encoder: Gemma2Model,
    prompt: str | list[str],
    negative_prompt: str | list[str],
    do_classifier_free_guidance: bool,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    complex_human_instruction: list[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, prompt_attention_mask = get_gemma_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=complex_human_instruction,
    )
    select_index = [0] + list(range(-max_sequence_length + 1, 0))
    prompt_embeds = prompt_embeds[:, select_index]
    prompt_attention_mask = prompt_attention_mask[:, select_index]

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    if not do_classifier_free_guidance:
        return prompt_embeds, prompt_attention_mask

    negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
    negative_prompt_embeds, negative_prompt_attention_mask = get_gemma_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=negative_prompt,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=None,
    )
    negative_prompt_embeds = negative_prompt_embeds[:, select_index]
    negative_prompt_attention_mask = negative_prompt_attention_mask[:, select_index]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
    negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

    return (
        torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
        torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0),
    )


def prepare_latents(
    *,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    vae_scale_factor: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    shape = (batch_size, num_channels, height // vae_scale_factor, width // vae_scale_factor)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


def aspect_ratio_bins_for_sample_size(sample_size: int) -> dict[str, list[float]]:
    if sample_size == 16:
        return ASPECT_RATIO_512_BIN
    if sample_size == 32:
        return ASPECT_RATIO_1024_BIN
    raise ValueError(f"No built-in aspect-ratio bins for transformer sample_size={sample_size}")


def load_text_encoder(model_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[GemmaTokenizer, Gemma2Model]:
    text_dir = model_path / "text_encoder"
    text_config = Gemma2TextConfig.from_json(text_dir / "config.json")
    tokenizer = GemmaTokenizer(model_path / "tokenizer", pad_token_id=text_config.pad_token_id)
    with torch.device("meta"):
        text_encoder = Gemma2Model(text_config)
    load_safetensors_into_module(text_encoder, text_dir, dtype=dtype, strict=True)
    text_encoder.rotary_emb.reset_inv_freq(torch.device("cpu"))
    text_encoder.eval().to(device=device, dtype=dtype)
    return tokenizer, text_encoder


def load_transformer(model_path: Path, device: torch.device, dtype: torch.dtype) -> SanaTransformer2DModel:
    transformer_dir = model_path / "transformer"
    config = ns_from_dict(read_json(transformer_dir / "config.json"))
    with torch.device("meta"):
        transformer = SanaTransformer2DModel(config)
    load_safetensors_into_module(transformer, transformer_dir, dtype=dtype, strict=True)
    transformer.eval().to(device=device, dtype=dtype)
    return transformer


def load_vae_decoder(model_path: Path, device: torch.device, dtype: torch.dtype) -> AutoencoderDCDecoder:
    vae_dir = model_path / "vae"
    config = ns_from_dict(read_json(vae_dir / "config.json"))
    with torch.device("meta"):
        vae = AutoencoderDCDecoder(config)
    load_safetensors_into_module(vae, vae_dir, dtype=dtype, strict=True)
    vae.eval().to(device=device, dtype=dtype)
    return vae


def load_scheduler(model_path: Path) -> DPMSolverMultistepScheduler:
    config = ns_from_dict(read_json(model_path / "scheduler" / "scheduler_config.json"))
    return DPMSolverMultistepScheduler(config)


@torch.inference_mode()
def generate_images_pure(args: argparse.Namespace) -> list[Image.Image]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    device = torch.device(args.device) if args.device else get_device()
    dtype = resolve_dtype(args.dtype, device)
    vae_dtype = resolve_dtype(args.vae_dtype, device) if args.vae_dtype else dtype

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(f"`height` and `width` must be divisible by 32, got {args.height}x{args.width}")

    print(f"device={device} dtype={dtype} model={model_path} backend=pure")

    tokenizer, text_encoder = load_text_encoder(model_path, device, dtype)
    prompt_embeds, prompt_attention_mask = encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=args.guidance_scale > 1.0,
        num_images_per_prompt=args.num_images,
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        complex_human_instruction=None if args.no_complex_human_instruction else DEFAULT_COMPLEX_HUMAN_INSTRUCTION,
    )
    del text_encoder
    empty_device_cache(device)

    transformer = load_transformer(model_path, device, dtype)
    scheduler = load_scheduler(model_path)

    orig_height, orig_width = args.height, args.width
    height, width = args.height, args.width
    if args.use_resolution_binning:
        bins = aspect_ratio_bins_for_sample_size(int(transformer.config.sample_size))
        height, width = classify_height_width_bin(height, width, bins)

    timesteps = scheduler.set_timesteps(args.steps, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = prepare_latents(
        batch_size=args.num_images,
        num_channels=int(transformer.config.in_channels),
        height=height,
        width=width,
        vae_scale_factor=32,
        device=device,
        generator=generator,
    )

    for i, timestep in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
        timestep_batch = timestep.expand(latent_model_input.shape[0])
        timestep_batch = timestep_batch * float(getattr(transformer.config, "timestep_scale", 1.0))
        noise_pred = transformer(
            latent_model_input.to(dtype=dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=dtype),
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep_batch,
        ).float()

        if args.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if int(transformer.config.out_channels) // 2 == int(transformer.config.in_channels):
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        latents = scheduler.step(noise_pred, timestep, latents)[0]
        if args.progress:
            print(f"step {i + 1}/{len(timesteps)}")

    del transformer, prompt_embeds, prompt_attention_mask
    empty_device_cache(device)

    vae = load_vae_decoder(model_path, device, vae_dtype)
    latents = latents.to(device=device, dtype=vae_dtype)
    images = vae.decode(latents / float(vae.config.scaling_factor))
    if args.use_resolution_binning:
        images = resize_and_crop_tensor(images, orig_width, orig_height)
    return tensor_to_pil(images)


@torch.inference_mode()
def get_hf_gemma_prompt_embeds(
    *,
    tokenizer,
    text_encoder,
    prompt: str | list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    complex_human_instruction: list[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_list = text_preprocessing(prompt)
    tokenizer.padding_side = "right"

    if complex_human_instruction:
        chi_prompt = "\n".join(complex_human_instruction)
        prompt_list = [chi_prompt + p for p in prompt_list]
        max_length_all = len(tokenizer.encode(chi_prompt)) + max_sequence_length - 2
    else:
        max_length_all = max_sequence_length

    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=max_length_all,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    attention_mask = text_inputs.attention_mask.to(device)
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)
    return prompt_embeds, attention_mask


@torch.inference_mode()
def encode_prompt_hf(
    *,
    tokenizer,
    text_encoder,
    prompt: str | list[str],
    negative_prompt: str | list[str],
    do_classifier_free_guidance: bool,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    complex_human_instruction: list[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, prompt_attention_mask = get_hf_gemma_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=complex_human_instruction,
    )

    select_index = [0] + list(range(-max_sequence_length + 1, 0))
    prompt_embeds = prompt_embeds[:, select_index]
    prompt_attention_mask = prompt_attention_mask[:, select_index]

    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    if not do_classifier_free_guidance:
        return prompt_embeds, prompt_attention_mask

    negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
    negative_prompt_embeds, negative_prompt_attention_mask = get_hf_gemma_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=negative_prompt,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        complex_human_instruction=None,
    )
    negative_prompt_embeds = negative_prompt_embeds[:, select_index]
    negative_prompt_attention_mask = negative_prompt_attention_mask[:, select_index]
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)

    return (
        torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
        torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0),
    )


def generate_images_parity(args: argparse.Namespace) -> list[Image.Image]:
    from diffusers import SanaPipeline

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    device = torch.device(args.device) if args.device else get_device()
    dtype = resolve_dtype(args.dtype, device)
    vae_dtype = resolve_dtype(args.vae_dtype, device) if args.vae_dtype else dtype

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(f"`height` and `width` must be divisible by 32, got {args.height}x{args.width}")

    print(f"device={device} dtype={dtype} model={model_path} backend=parity")

    pipe = SanaPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)

    if hasattr(pipe, "vae"):
        pipe.vae.to(vae_dtype)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipeline_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.steps,
        "generator": generator,
    }

    # Keep the default path byte-for-byte aligned with mps_image_diffusers.py.
    # Only pass the broader torch-wrapper options when they change behavior.
    if args.negative_prompt:
        pipeline_kwargs["negative_prompt"] = args.negative_prompt
    if args.num_images != 1:
        pipeline_kwargs["num_images_per_prompt"] = args.num_images
    if not args.use_resolution_binning:
        pipeline_kwargs["use_resolution_binning"] = False
    if args.max_sequence_length != 300:
        pipeline_kwargs["max_sequence_length"] = args.max_sequence_length
    if args.no_complex_human_instruction:
        pipeline_kwargs["complex_human_instruction"] = None

    with torch.inference_mode():
        return pipe(**pipeline_kwargs).images


def generate_images(args: argparse.Namespace) -> list[Image.Image]:
    if args.backend == "pure":
        return generate_images_pure(args)
    return generate_images_parity(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sana image with Diffusers parity or a local torch implementation on Mac MPS."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SANA_IMAGE_MODEL_PATH", DEFAULT_MODEL_PATH),
        help=(
            "A local Sana image diffusers-format model path. Defaults to the local SANA1.5 1.6B 1024px model. "
            f"For a lighter local model, use: {LOCAL_SANA_600M_512_MODEL}"
        ),
    )
    parser.add_argument("--prompt", default='a cyberpunk cat with a neon sign that says "Sana"')
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="sana_mps_image.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--vae-dtype", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    parser.add_argument(
        "--backend",
        choices=["parity", "pure"],
        default="pure",
        help="parity matches mps_image_diffusers.py byte-for-byte; pure uses the local torch reimplementation.",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | mps; auto-detect if omitted.")
    parser.add_argument("--no-resolution-binning", dest="use_resolution_binning", action="store_false")
    parser.add_argument("--no-complex-human-instruction", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.set_defaults(use_resolution_binning=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = generate_images(args)
    output = Path(args.output)
    if len(images) == 1:
        images[0].save(output)
        print(f"saved {output}")
        return

    stem = output.with_suffix("")
    suffix = output.suffix or ".png"
    for i, image in enumerate(images):
        path = output if i == 0 else Path(f"{stem}_{i:02d}{suffix}")
        image.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
