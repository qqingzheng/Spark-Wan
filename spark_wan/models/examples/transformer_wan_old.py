# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph


try:
    from .attention import flash_attention
except Exception:
    from attention import flash_attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    # pt, ph, pw = kernel_size
    # cp = (pt * ph * pw) // 2
    # xp = einops.rearrange(x, 'b c (t pt) (h ph) (w pw) -> (pt ph pw) b c t h w', pt=pt, ph=ph, pw=pw)
    # xc = xp[cp]
    # return xc
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


def apply_rotary_emb_transposed(x, freqs_cis):
    x = x.transpose(1, 2)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = x.float() * cos + x_rotated.float() * sin
    out = out.to(x)
    out = out.transpose(1, 2)
    return out


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        flash_attn=False,
        cross_attn=False,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            # way 1
            query = apply_rotary_emb_transposed(query, rotary_emb)
            key = apply_rotary_emb_transposed(key, rotary_emb)

            # way 2
            # def apply_rotary_emb(
            #     hidden_states: torch.Tensor,
            #     freqs_cos: torch.Tensor,
            #     freqs_sin: torch.Tensor,
            # ):
            #     x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
            #     x1, x2 = x[..., 0], x[..., 1]
            #     cos = freqs_cos[..., 0::2]
            #     sin = freqs_sin[..., 1::2]
            #     out = torch.empty_like(hidden_states)
            #     out[..., 0::2] = x1 * cos - x2 * sin
            #     out[..., 1::2] = x1 * sin + x2 * cos
            #     return out.type_as(hidden_states)

            # query = apply_rotary_emb(query, *rotary_emb)
            # key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if flash_attn:
                hidden_states_img = flash_attention(
                    query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2), k_lens=None
                )
                hidden_states_img = hidden_states_img.flatten(2, 3)
            else:
                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if flash_attn:
            if cross_attn:
                k_lens = None
                window_size = None
            else:
                k_lens = torch.tensor([query.shape[2]])
                window_size = (-1, -1)
            hidden_states = flash_attention(
                q=query.transpose(1, 2),
                k=key.transpose(1, 2),
                v=value.transpose(1, 2),
                k_lens=k_lens,
                window_size=window_size,
            )
            hidden_states = hidden_states.flatten(2, 3)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, in_channels, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        weight = weight[:, :16, :, :, :]

        sd = {
            "proj.weight": weight.clone(),
            "proj.bias": bias.clone(),
            "proj_2x.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2) / 8.0,
            "proj_2x.bias": bias.clone(),
            "proj_4x.weight": einops.repeat(weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4) / 64.0,
            "proj_4x.bias": bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)
        return


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta

    @torch.no_grad()
    def get_frequency(self, dim, pos):
        T, H, W = pos.shape
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim)
        )
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (T, H, W)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward_inner(self, frame_indices, height, width, device):
        GT, GY, GX = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32),
            torch.arange(0, width, device=device, dtype=torch.float32),
            indexing="ij",
        )

        FCT, FST = self.get_frequency(self.DT, GT)
        FCY, FSY = self.get_frequency(self.DY, GY)
        FCX, FSX = self.get_frequency(self.DX, GX)

        result = torch.cat([FCT, FCY, FCX, FST, FSY, FSX], dim=0)

        return result.to(device)

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        frame_indices = frame_indices.unbind(0)
        results = [self.forward_inner(f, height, width, device) for f in frame_indices]
        results = torch.stack(results, dim=0)
        return results


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,  # 128
        patch_size: Tuple[int, int, int],  # [1, 2, 2]
        max_seq_len: int,  # max_seq_len
        theta: float = 10000.0,  # 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:  # (44, 42, 42)
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        flash_attn=False,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb, flash_attn=flash_attn, cross_attn=False
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            flash_attn=flash_attn,
            cross_attn=True,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        has_clean_patch_embedding=False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        # self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.rope = HunyuanVideoRotaryPosEmbed(rope_dim=(44, 42, 42), theta=10000.0)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # 5. Initial FramePack
        self.inner_dim = inner_dim
        if has_clean_patch_embedding:
            self.install_clean_patch_embedding()

        self.gradient_checkpointing = False

    def install_clean_patch_embedding(self):
        self.clean_patch_embedding = WanPatchEmbedForCleanLatents(16, self.inner_dim)
        self.config["has_clean_patch_embedding"] = True
        self.config.has_clean_patch_embedding = True

    def gradient_checkpointing_method(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            result = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
        else:
            result = block(*args)
        return result

    def merge_input_hidden_states(
        self,
        latents,
    ):
        pass

    def process_input_hidden_states(
        self,
        latents,
        indices_latents=None,
        latents_clean=None,
        indices_latents_clean=None,
        latents_history_2x=None,
        indices_latents_history_2x=None,
        latents_history_4x=None,
        indices_latents_history_4x=None,
        is_dance_forcing=False,
    ):
        if is_dance_forcing:
            hidden_states = None
            rope_freqs = None
            dance_forcing_height_list = []
            dance_forcing_width_list = []
            dance_forcing_temporal_list = []
            dance_forcing_seq_list = []
            for idx, cur_hidden_states in enumerate(latents):
                cur_hidden_states = self.gradient_checkpointing_method(
                    self.patch_embedding, cur_hidden_states.to(latents_clean.device, latents_clean.dtype)
                )
                B, C, T, H, W = cur_hidden_states.shape

                cur_hidden_states = cur_hidden_states.flatten(2).transpose(1, 2)

                cur_indices_latents = indices_latents[:, idx * T : (idx + 1) * T]
                cur_rope_freqs = self.rope(
                    frame_indices=cur_indices_latents, height=H, width=W, device=cur_hidden_states.device
                )
                cur_rope_freqs = cur_rope_freqs.flatten(2).transpose(1, 2)

                dance_forcing_height_list.append(H)
                dance_forcing_width_list.append(W)
                dance_forcing_temporal_list.append(T)
                dance_forcing_seq_list.append(cur_hidden_states.shape[1])

                if hidden_states is None:
                    hidden_states = cur_hidden_states
                    rope_freqs = cur_rope_freqs
                else:
                    hidden_states = torch.cat([cur_hidden_states, hidden_states], dim=1)
                    rope_freqs = torch.cat([cur_rope_freqs, rope_freqs], dim=1)
        else:
            hidden_states = self.gradient_checkpointing_method(self.patch_embedding, latents)
            B, C, T, H, W = hidden_states.shape

            if indices_latents is None:
                indices_latents = torch.arange(0, T).unsqueeze(0).expand(B, -1)

            hidden_states = hidden_states.flatten(2).transpose(
                1, 2
            )  # torch.Size([1, 3072, 9, 44, 34]) -> torch.Size([1, 13464, 3072])

            rope_freqs = self.rope(
                frame_indices=indices_latents,
                height=H,
                width=W,
                device=hidden_states.device,
            )  # torch.Size([1, 9]) -> torch.Size([1, 256, 9, 44, 34])
            rope_freqs = rope_freqs.flatten(2).transpose(1, 2)  # torch.Size([1, 13464, 256])

        if latents_clean is not None and indices_latents_clean is not None:
            latents_clean = latents_clean.to(hidden_states)  # torch.Size([1, 16, 2, 88, 68])
            latents_clean = self.gradient_checkpointing_method(
                self.clean_patch_embedding.proj, latents_clean
            )  # torch.Size([1, 3072, 2, 44, 34])
            _, _, _, H1, W1 = latents_clean.shape
            latents_clean = latents_clean.flatten(2).transpose(1, 2)  # torch.Size([1, 2992, 3072])

            clean_latent_rope_freqs = self.rope(
                frame_indices=indices_latents_clean,
                height=H1,
                width=W1,
                device=latents_clean.device,
            )
            clean_latent_rope_freqs = clean_latent_rope_freqs.flatten(2).transpose(1, 2)  # torch.Size([1, 2992, 256])

            hidden_states = torch.cat([latents_clean, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

        if latents_history_2x is not None and indices_latents_history_2x is not None:
            latents_history_2x = latents_history_2x.to(hidden_states)
            latents_history_2x = pad_for_3d_conv(latents_history_2x, (2, 4, 4))
            latents_history_2x = self.gradient_checkpointing_method(
                self.clean_patch_embedding.proj_2x, latents_history_2x
            )
            latents_history_2x = latents_history_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(
                frame_indices=indices_latents_history_2x,
                height=H1,
                width=W1,
                device=latents_history_2x.device,
            )
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if latents_history_4x is not None and indices_latents_history_4x is not None:
            latents_history_4x = latents_history_4x.to(hidden_states)
            latents_history_4x = pad_for_3d_conv(latents_history_4x, (4, 8, 8))
            latents_history_4x = self.gradient_checkpointing_method(
                self.clean_patch_embedding.proj_4x, latents_history_4x
            )
            latents_history_4x = latents_history_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(
                frame_indices=indices_latents_history_4x,
                height=H1,
                width=W1,
                device=latents_history_4x.device,
            )
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        # ori_hidden_states   torch.Size([2, 3510, 3072])
        # latents_clean       torch.Size([2, 780, 3072])
        # latents_history_2x  torch.Size([2, 104, 3072])
        # latents_history_4x  torch.Size([2, 112, 3072])
        # hidden_states       torch.Size([2, 4506, 3072])

        return hidden_states, rope_freqs

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        flash_attn=False,
        is_dance_forcing=False,
        indices_latents=None,
        latents_clean=None,
        indices_latents_clean=None,
        latents_history_2x=None,
        indices_latents_history_2x=None,
        latents_history_4x=None,
        indices_latents_history_4x=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if indices_latents is not None and indices_latents.ndim == 1:
            indices_latents = indices_latents.unsqueeze(0)
        if indices_latents_clean is not None and indices_latents_clean.ndim == 1:
            indices_latents_clean = indices_latents_clean.unsqueeze(0)
        if indices_latents_history_2x is not None and indices_latents_history_2x.ndim == 1:
            indices_latents_history_2x = indices_latents_history_2x.unsqueeze(0)
        if indices_latents_history_4x is not None and indices_latents_history_4x.ndim == 1:
            indices_latents_history_4x = indices_latents_history_4x.unsqueeze(0)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if is_dance_forcing:
            hidden_states, rotary_emb = self.process_input_hidden_states(
                hidden_states,
                indices_latents,
                latents_clean,
                indices_latents_clean,
                latents_history_2x,
                indices_latents_history_2x,
                latents_history_4x,
                indices_latents_history_4x,
                is_dance_forcing=is_dance_forcing,
            )
        else:
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w
            original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

            if self.config.has_clean_patch_embedding:
                hidden_states, rotary_emb = self.process_input_hidden_states(
                    hidden_states,
                    indices_latents,
                    latents_clean,
                    indices_latents_clean,
                    latents_history_2x,
                    indices_latents_history_2x,
                    latents_history_4x,
                    indices_latents_history_4x,
                )  # torch.Size([1, 4506, 5120]), torch.Size([1, 4506, 256])
            else:
                # way 1
                indices_latents = torch.arange(0, num_frames).unsqueeze(0).expand(batch_size, -1)
                rotary_emb = self.rope(
                    frame_indices=indices_latents,
                    height=post_patch_height,
                    width=post_patch_width,
                    device=hidden_states.device,
                )  # torch.Size([1, 9]) -> torch.Size([1, 256, 9, 15, 26])
                rotary_emb = rotary_emb.flatten(2).transpose(1, 2)  # torch.Size([1, 3510, 256])
                # way 2
                # rotary_emb = self.rope(hidden_states)  # torch.Size([1, 1, 3510, 128]), torch.Size([1, 1, 3510, 128])

                hidden_states = self.patch_embedding(hidden_states)
                hidden_states = hidden_states.flatten(2).transpose(1, 2)  # torch.Size([1, 3510, 5120])

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, flash_attn
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, flash_attn)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if self.config.has_clean_patch_embedding:
            hidden_states = hidden_states[:, -original_context_length:, :]

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    device = "cuda"
    weight_dtype = torch.bfloat16
    transformer = WanTransformer3DModel.from_pretrained(
        "/mnt/workspace/checkpoints/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/transformer"
    )
    transformer.requires_grad_(False)
    transformer.eval()

    transformer.install_clean_patch_embedding()
    transformer.clean_patch_embedding.initialize_weight_from_another_conv3d(transformer.patch_embedding)
    transformer = transformer.to(device, dtype=weight_dtype)

    dance_forcing = True
    batch_size = 2

    if dance_forcing:
        # timesteps = torch.randint(0, 1000, (batch_size, 9)).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        noisy_model_input = [
            torch.randn(batch_size, 36, 3, 48, 80),
            torch.randn(batch_size, 36, 3, 24, 40),
            torch.randn(batch_size, 36, 3, 12, 20),
        ]
    else:
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        noisy_model_input = torch.randn(batch_size, 36, 9, 48, 80).to(device, dtype=weight_dtype)

    prompt_embeds = torch.randn(batch_size, 512, 4096).to(device, dtype=weight_dtype)
    image_embeds = torch.randn(batch_size, 257, 1280).to(device, dtype=weight_dtype)
    indices_latents = torch.randint(0, 10, (batch_size, 9)).to(device)
    latents_clean = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    indices_clean_latents = torch.randint(0, 3, (batch_size, 2)).to(device)
    latents_history_2x = torch.randn(batch_size, 16, 2, 48, 80).to(device, dtype=weight_dtype)
    indices_latents_history_2x = torch.randint(0, 3, (batch_size, 2)).to(device)
    latents_history_4x = torch.randn(batch_size, 16, 16, 48, 80).to(device, dtype=weight_dtype)
    indices_latents_history_4x = torch.randint(0, 17, (batch_size, 16)).to(device)

    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_image=image_embeds,
        return_dict=False,
        flash_attn=False,
        is_dance_forcing=dance_forcing,
        indices_latents=indices_latents,  # torch.Size([2, 9])
        latents_clean=latents_clean.to(weight_dtype),  # torch.Size([2, 16, 2, 60, 104])
        indices_latents_clean=indices_clean_latents,  # torch.Size([2, 2])
        latents_history_2x=latents_history_2x.to(weight_dtype),  # torch.Size([2, 16, 2, 60, 104])
        indices_latents_history_2x=indices_latents_history_2x,  # torch.Size([2, 2])
        latents_history_4x=latents_history_4x.to(weight_dtype),  # torch.Size([2, 16, 16, 60, 104])
        indices_latents_history_4x=indices_latents_history_4x,  # torch.Size([2, 16])
    )[0]

    import pdb

    pdb.set_trace()
