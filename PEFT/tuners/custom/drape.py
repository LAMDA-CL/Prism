# coding=utf-8
# Copyright 2024-present the HuggingFace Inc. team.
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
import warnings
import enum
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
import torch
import torch.nn as nn
import os
import json
import time
from ...utils import PeftConfig
from ...peft_model import PeftModel
from functools import wraps
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


def _generator_fp32(fn):
    """Keep generator attention in FP32 on CPU, CUDA and Ascend alike."""
    @wraps(fn)
    def wrapped(self, vision_features, text_features, *args, **kwargs):
        with torch.autocast(device_type=text_features.device.type, enabled=False):
            return fn(self, vision_features, text_features, *args, **kwargs)
    return wrapped

from config.backbone.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


def split_system_user(
    input_ids,
    user_role_ids=[3148, 1001, 29901],
    assistant_role_ids=[319, 1799, 9047, 13566, 29901],
    pad_value=IGNORE_INDEX,
    inference=False,
):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, L = input_ids.shape
    user_role_ids = list(user_role_ids)
    assistant_role_ids = list(assistant_role_ids)

    def find_first(seq, pat):
        n, m = len(seq), len(pat)
        for i in range(n - m + 1):
            if seq[i : i + m] == pat:
                return i
        return -1

    system_mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)
    user_mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        ids = input_ids[b].tolist()
        u_pos = find_first(ids, user_role_ids)
        if u_pos == -1:
            continue
        system_mask[b, :u_pos] = True
        start = u_pos + len(user_role_ids)
        if inference:
            end = L
        else:
            # 训练：用 assistant role 作为 user 的结束
            a_pos = find_first(ids, assistant_role_ids)
            if a_pos == -1 or a_pos <= start:
                end = L
            else:
                end = a_pos
        if start < end:
            user_mask[b, start:end] = True
    system_ids = input_ids.masked_fill(~system_mask, pad_value)
    user_ids = input_ids.masked_fill(~user_mask, pad_value)

    no_assistant_mask = torch.ones((B, L), dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        ids = input_ids[b].tolist()
        u_pos = find_first(ids, user_role_ids)
        if u_pos == -1:
            continue

        a_pos = find_first(ids, assistant_role_ids)
        if a_pos == -1:
            continue  # 无助手回答，无需排除

        no_assistant_mask[b, a_pos:] = False

    # 生成排除回答后的ID：助手回答部分替换为pad_value
    # no_assistant_ids = input_ids.masked_fill(~no_assistant_mask, pad_value)

    return system_ids, user_ids, system_mask, user_mask, no_assistant_mask


import torch
import torch.nn.functional as F


def masked_segment_avgpool(x: torch.Tensor, mask: torch.Tensor, P: int):
    """
    x:    [B, L, H]
    mask: [B, L]  (True=valid, False=pad)
    returns: [B, P, H]
    """
    B, L, H = x.shape
    device = x.device
    dtype = x.dtype

    # segment boundaries (roughly equal splits)
    # idx: [P+1]
    idx = torch.linspace(0, L, steps=P + 1, device=device)
    idx = idx.round().long().clamp(0, L)

    pooled = []
    for i in range(P):
        s = idx[i].item()
        e = idx[i + 1].item()
        if e <= s:
            # degenerate segment -> take zeros
            pooled.append(torch.zeros(B, 1, H, device=device, dtype=dtype))
            continue
        seg = x[:, s:e, :]  # [B, segL, H]
        seg_m = mask[:, s:e].unsqueeze(-1)  # [B, segL, 1]
        seg = seg * seg_m.to(dtype=dtype)

        denom = seg_m.sum(dim=1).clamp_min(1.0).to(dtype=dtype)  # [B,1]
        mean = seg.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)  # [B,1,H]
        pooled.append(mean)

    return torch.cat(pooled, dim=1)  # [B,P,H]


class DrapeGeneratorType(str, enum.Enum):
    """
    The type of cross-modal generator architecture to use.
    """

    MLP_MIXER = "MLP_MIXER"
    CROSS_ATTENTION = "CROSS_ATTENTION"
    GATED_FUSION = "GATED_FUSION"


@dataclass
class DrapeConfig(PeftConfig):
    generator_type: Union[str, DrapeGeneratorType] = field(
        default=DrapeGeneratorType.CROSS_ATTENTION,
        metadata={
            "help": "The architecture type of the cross-modal parameter generator"
        },
    )
    generator_hidden_size: int = field(
        default=512,
        metadata={"help": "The hidden dimension of the generator network"},
    )
    generator_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers/depth of the generator"},
    )
    generator_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability of the generator"},
    )
    vision_input_dim: int = field(
        default=4096,
        metadata={"help": "The dimension of the visual features from the vision tower"},
    )
    text_input_dim: int = field(
        default=4096,
        metadata={
            "help": "The dimension of the text instruction embeddings (LLM hidden size)"
        },
    )
    vision_feature_select_strategy: str = field(
        default="full_patch",
        metadata={
            "help": "Strategy to handle visual tokens: 'cls_token', 'mean', or 'full_patch'"
        },
    )
    use_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the generator linear layers"},
    )
    num_virtual_tokens: int = field(default=10)

    token_dim: int = field(default=4096)

    separate: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate separate prompts for image and text modalities"
        },
    )
    skip_vision_attn: bool = field(
        default=False,
        metadata={"help": "Ablation flag: skip vision cross-attention in generator"},
    )
    pool_mode: str = field(
        default="segment",
        metadata={"help": "Text pooling mode for generator: 'segment' or 'mean'"},
    )
    query_mode: str = field(
        default="segpool",
        metadata={
            "help": "Prompt query mode for non-separate generator: 'segpool' or 'learnable_mm'"
        },
    )
    static_prompt: bool = field(
        default=False,
        metadata={"help": "Ablation flag: use per-task static prompt table instead of generation"},
    )

    # === Continual Learning Configuration ===
    enable_continual_learning: bool = field(
        default=False,
        metadata={"help": "Enable continual learning with null-space projection and routing"},
    )
    clip_model_name: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "CLIP model for text-based task routing"},
    )
    router_num_centers: int = field(
        default=1,
        metadata={"help": "Number of routing centers per task (1 = single-center router)"},
    )
    router_kmeans_iters: int = field(
        default=20,
        metadata={"help": "K-means iterations for multi-center routing (router_num_centers > 1)"},
    )
    router_k1_center_mode: str = field(
        default="learned",
        metadata={"help": "Single-center mode when router_num_centers=1: 'mean' or 'learned'"},
    )
    null_space_threshold: float = field(
        default=0.99,
        metadata={"help": "SVD explained variance threshold for null-space projection"},
    )
    apply_null_space_projection: bool = field(
        default=True,
        metadata={"help": "Whether to enable null-space pipeline (feature collection, covariance/projection update, and gradient projection)"},
    )
    current_task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the current task being trained"},
    )
    previous_task_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory of the previous task's saved model (for loading router state & null-space)"},
    )

    def __post_init__(self):
        self.peft_type = "DRAPETUNING"
        self.task_type = "CAUSAL_LM_DRAPE"

        # Validate generator type
        if isinstance(self.generator_type, str):
            try:
                self.generator_type = DrapeGeneratorType(self.generator_type)
            except ValueError:
                raise ValueError(
                    f"generator_type {self.generator_type} not recognized. "
                    f"Please use one of {[t.value for t in DrapeGeneratorType]}"
                )

        # Inherited from PromptLearningConfig check
        if self.num_virtual_tokens is None:
            # Default fallback if not specified
            self.num_virtual_tokens = 10

        if self.pool_mode not in ("segment", "mean"):
            raise ValueError("pool_mode must be either 'segment' or 'mean'")
        if self.query_mode not in ("segpool", "learnable_mm"):
            raise ValueError("query_mode must be either 'segpool' or 'learnable_mm'")


class DrapeGenerator(torch.nn.Module):
    def __init__(self, config: DrapeConfig):
        super().__init__()
        self.generator_type = config.generator_type
        self.num_virtual_tokens = config.num_virtual_tokens
        self.token_dim = config.token_dim  # The hidden dim of the LLM
        self.generator_hidden_size = config.generator_hidden_size
        self.separate = config.separate
        self.skip_vision_attn = getattr(config, "skip_vision_attn", False)
        self.pool_mode = getattr(config, "pool_mode", "segment")
        self.query_mode = getattr(config, "query_mode", "segpool")
        self.static_prompt = getattr(config, "static_prompt", False)
        self.vision_seq_len = getattr(config, "vision_seq_len", 576)

        H = self.generator_hidden_size
        P = self.num_virtual_tokens
        D = self.token_dim

        self.vision_proj = nn.Linear(config.vision_input_dim, H)
        self.text_proj = nn.Linear(config.text_input_dim, H)

        if self.separate:
            # === Cross-Modal Aware Separate Generation ===
            # Shared cross-modal interaction: vision attends to text
            self.cross_modal_attn = nn.MultiheadAttention(
                embed_dim=H,
                num_heads=8,
                batch_first=True,
                dropout=config.generator_dropout,
            )
            self.cross_modal_norm = nn.LayerNorm(H)

            # Image branch: learnable queries pool from fused vision (attention pooling, no seq_reducer)
            self.img_queries = nn.Parameter(torch.randn(1, self.num_virtual_tokens, H))
            nn.init.normal_(self.img_queries, mean=0.0, std=0.02)
            self.img_cross_attn = nn.MultiheadAttention(
                embed_dim=H,
                num_heads=8,
                batch_first=True,
                dropout=config.generator_dropout,
            )
            self.img_norm = nn.LayerNorm(H)
            self.image_head = nn.Sequential(
                nn.Linear(H, H), nn.GELU(), nn.Linear(H, self.token_dim)
            )

            # Text branch: learnable queries pool from vision-enriched text
            self.txt_queries = nn.Parameter(torch.randn(1, self.num_virtual_tokens, H))
            nn.init.normal_(self.txt_queries, mean=0.0, std=0.02)
            self.txt_cross_attn = nn.MultiheadAttention(
                embed_dim=H,
                num_heads=8,
                batch_first=True,
                dropout=config.generator_dropout,
            )
            self.txt_norm = nn.LayerNorm(H)
            self.text_head = nn.Sequential(
                nn.Linear(H, H), nn.GELU(), nn.Linear(H, self.token_dim)
            )
            self.apply(self._init_weights)
        else:
            # Interaction Module (The "Manifold" Learner)
                if self.generator_type == DrapeGeneratorType.CROSS_ATTENTION:
                    if self.query_mode == "learnable_mm":
                        # Learnable prompt queries cross-attend to multimodal tokens
                        self.mm_queries = nn.Parameter(torch.randn(1, self.num_virtual_tokens, H))
                        nn.init.normal_(self.mm_queries, mean=0.0, std=0.02)
                        self.mm_query_attn = nn.MultiheadAttention(
                            embed_dim=H,
                            num_heads=8,
                            batch_first=True,
                            dropout=config.generator_dropout,
                        )
                        self.mm_query_norm = nn.LayerNorm(H)
                    else:
                        # self.text_queries = nn.Parameter(torch.randn(1, self.num_virtual_tokens, H))
                        # --- (1) Text attention pooling -> t_summary [B,P,H]
                        self.text_pool_attn = nn.MultiheadAttention(
                            embed_dim=H,
                            num_heads=8,
                            batch_first=True,
                            dropout=config.generator_dropout,
                        )
                        self.text_pool_norm = nn.LayerNorm(H)

                        # --- (2) P queries attend to vision tokens -> [B,P,H]
                        self.vision_attn = nn.MultiheadAttention(
                            embed_dim=H,
                            num_heads=8,
                            batch_first=True,
                            dropout=config.generator_dropout,
                        )
                        self.prompt_norm = nn.LayerNorm(H)

                    # --- (3) Project to LLM hidden size D -> [B,P,D]
                    self.prompt_head = nn.Sequential(
                        nn.Linear(H, 2 * H),
                        nn.GELU(),
                        nn.Linear(2 * H, D),
                    )

                    self.prompt_dropout = nn.Dropout(getattr(config, "prompt_dropout", 0.0))

        # Static prompt ablation: one per-task learnable prompt table.
        if self.static_prompt and not self.separate:
            self.static_prompts = nn.Parameter(torch.randn(1, P, D) * 0.02)


    # CHANGED(fp32-generator): disable autocast for generator forward so it stays in fp32.
    @_generator_fp32
    def forward(
            self,
            vision_features: torch.Tensor,
            text_features: torch.Tensor,
            mask: torch.Tensor = None,
            return_attn: bool = False,
        ):
            debug_nan = os.environ.get("DEBUG_DRAPE_NAN", "0") == "1"

            def _debug_tensor(name, tensor):
                if not debug_nan or tensor is None:
                    return
                try:
                    t = tensor.detach().float()
                    has_nan = torch.isnan(t).any().item()
                    has_inf = torch.isinf(t).any().item()
                    finite_ratio = torch.isfinite(t).float().mean().item()
                    print(f"[DRAPE_DEBUG] {name}: shape={tuple(t.shape)} nan={has_nan} inf={has_inf} finite_ratio={finite_ratio:.6f}")
                except Exception as e:
                    print(f"[DRAPE_DEBUG] {name}: stat_error={e}")

            prev_dtype = text_features.dtype
            device = text_features.device
            # CHANGED(fp32-generator): run the current generator forward in fp32 even
            # when the base model is bf16/fp16, then cast prompts back to prev_dtype.
            compute_dtype = self.vision_proj.weight.dtype
            text_features = text_features.to(dtype=compute_dtype)
            if vision_features is not None:
                vision_features = vision_features.to(dtype=compute_dtype)
            B = text_features.size(0)
            prompt_attn_weights = None

            if self.static_prompt and not self.separate:
                prompts = self.static_prompts.expand(B, -1, -1).to(prev_dtype)
                if return_attn:
                    return prompts, None
                return prompts

            if self.separate:
                if text_features.dim() == 3:
                    if mask is not None:
                        mask_expanded = mask.unsqueeze(-1).to(dtype=text_features.dtype)
                        text_features = text_features * mask_expanded

            if self.separate:
                if vision_features is None:
                    # Text-only fallback for separate mode:
                    # generate a single prompt stream from text branch only.
                    t = self.text_proj(text_features)  # [B, S, H]
                    key_padding_mask = None
                    if mask is not None:
                        key_padding_mask = mask == 0
                    txt_q = self.txt_queries.expand(B, -1, -1)  # [B, P, H]
                    txt_ctx, _ = self.txt_cross_attn(
                        query=txt_q,
                        key=t,
                        value=t,
                        key_padding_mask=key_padding_mask,
                    )
                    txt_ctx = self.txt_norm(txt_q + txt_ctx)
                    txt_prompts = self.text_head(txt_ctx)  # [B, P, token_dim]
                    txt_prompts = txt_prompts.to(prev_dtype)
                    if return_attn:
                        return txt_prompts, None
                    return txt_prompts

                # === Cross-Modal Aware Separate Generation ===
                # Step 1: Project both modalities to shared latent space
                v = self.vision_proj(vision_features)  # [B, N, H]
                t = self.text_proj(text_features)  # [B, S, H]

                # Step 2: Cross-modal fusion - vision attends to text
                key_padding_mask = None
                if mask is not None:
                    key_padding_mask = mask == 0
                fused_v, _ = self.cross_modal_attn(
                    query=v, key=t, value=t, key_padding_mask=key_padding_mask
                )
                fused_v = self.cross_modal_norm(v + fused_v)  # [B, N, H]

                # Step 3: Image branch - learnable queries attend to fused vision
                B = vision_features.size(0)
                img_q = self.img_queries.expand(B, -1, -1)  # [B, P, H]
                img_ctx, _ = self.img_cross_attn(query=img_q, key=fused_v, value=fused_v)
                img_ctx = self.img_norm(img_q + img_ctx)
                img_prompts = self.image_head(img_ctx)  # [B, P, token_dim]

                # Step 4: Text branch - learnable queries attend to vision-enriched text
                v_global = fused_v.mean(dim=1, keepdim=True)  # [B, 1, H]
                t_enriched = t + v_global.expand_as(t)  # text enriched with visual context
                txt_q = self.txt_queries.expand(B, -1, -1)  # [B, P, H]
                txt_ctx, _ = self.txt_cross_attn(
                    query=txt_q,
                    key=t_enriched,
                    value=t_enriched,
                    key_padding_mask=key_padding_mask,
                )
                txt_ctx = self.txt_norm(txt_q + txt_ctx)
                txt_prompts = self.text_head(txt_ctx)  # [B, P, token_dim]

                outputs = (img_prompts.to(prev_dtype), txt_prompts.to(prev_dtype))
                if return_attn:
                    return outputs, None
                return outputs
            else:
                t_latents = self.text_proj(text_features)  # [B, L, H]
                _debug_tensor("t_latents", t_latents)

                # key_padding_mask: True means "ignore"
                key_padding_mask = None
                if mask is not None:
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    key_padding_mask = ~mask  # [B, L]
                else:
                    mask = torch.ones(
                        (t_latents.size(0), t_latents.size(1)),
                        dtype=torch.bool,
                        device=t_latents.device,
                    )
                if key_padding_mask is None:
                    key_padding_mask = ~mask

                B = t_latents.size(0)
                H = self.generator_hidden_size
                P = self.num_virtual_tokens

                if self.query_mode == "learnable_mm":
                    mm_tokens = t_latents
                    mm_key_padding_mask = key_padding_mask
                    q = self.mm_queries.expand(B, -1, -1).to(device=t_latents.device, dtype=t_latents.dtype)
                    mm_ctx, _ = self.mm_query_attn(
                        query=q,
                        key=mm_tokens,
                        value=mm_tokens,
                        key_padding_mask=mm_key_padding_mask,
                    )
                    prompt_latents = self.mm_query_norm(q + mm_ctx)
                    prompt_latents = self.prompt_dropout(prompt_latents)
                else:
                    # (1) attention pooling on text
                    if self.pool_mode == "mean":
                        if mask is None:
                            mean_q = t_latents.mean(dim=1, keepdim=True)
                        else:
                            mask_f = mask.to(dtype=t_latents.dtype).unsqueeze(-1)
                            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                            mean_q = (t_latents * mask_f).sum(dim=1, keepdim=True) / denom
                        pool_q = mean_q.expand(-1, P, -1)
                        _debug_tensor("pool_q_mean", pool_q)
                    else:
                        pool_q = masked_segment_avgpool(t_latents, mask, P)  # [B,P,H]
                        _debug_tensor("pool_q_segment", pool_q)
                    # queries = self.text_queries.expand(B, -1, -1).to(device=t_latents.device, dtype=t_latents.dtype)
                    # pool_q=queries
                    t_sum, _ = self.text_pool_attn(
                        query=pool_q,
                        key=t_latents,
                        value=t_latents,
                        key_padding_mask=key_padding_mask,
                    )  # [B,P,H]w
                    _debug_tensor("t_sum_attn", t_sum)
                    t_sum = self.text_pool_norm(t_sum)  # residual
                    _debug_tensor("t_sum_norm", t_sum)
                    # (2) generate P queries from text summary
                    q = t_sum
                    if vision_features is not None and not self.skip_vision_attn:
                        v_latents = self.vision_proj(vision_features)  # [B, N, H]
                        _debug_tensor("v_latents", v_latents)
                        # (3) cross-attend to vision tokens
                        if return_attn:
                            v_ctx, prompt_attn_weights = self.vision_attn(
                                query=q,
                                key=v_latents,
                                value=v_latents,
                                need_weights=True,
                                average_attn_weights=False,
                            )  # [B, num_heads, P, N]
                        else:
                            v_ctx, _ = self.vision_attn(
                                query=q, key=v_latents, value=v_latents
                            )  # [B,P,H]
                        _debug_tensor("v_ctx", v_ctx)
                        prompt_latents = self.prompt_norm(v_ctx+q)
                        _debug_tensor("prompt_latents_norm", prompt_latents)
                        prompt_latents = self.prompt_dropout(prompt_latents)
                    else:
                        prompt_latents = q
                        _debug_tensor("prompt_latents_skip_vision", prompt_latents)
                        prompt_latents = self.prompt_dropout(prompt_latents)

                # (4) project to LLM prompt embeddings
                _debug_tensor("prompt_latents_before_head", prompt_latents)
                prompts = self.prompt_head(prompt_latents)  # [B,P,D]
                _debug_tensor("prompts_after_head", prompts)
                prompts = prompts.to(prev_dtype)
                if return_attn:
                    return prompts, prompt_attn_weights
            return prompts


# Task routing and shared-projector gradient protection.
IMAGE_TAGS = ("<image>", "<im_start>", "<im_end>", "<im_patch>")


class NullSpaceProjection:
    """
    Null-space projection for continual learning anti-forgetting.

    After each task finishes training, we collect the feature representations
    (activations) that pass through the shared projection layer (mm_projector).
    We compute the covariance of these features, find the principal subspace via SVD,
    and build a projection matrix onto its null space.

    When training on the next task, we project the gradients of mm_projector
    onto this null space so that updates don't interfere with previously
    learned representations.
    """

    def __init__(self, threshold=0.99):
        if not 0 < threshold <= 1:
            raise ValueError("null-space threshold must be in (0, 1]")
        self.cov_matrix = None
        self.num_samples = 0
        self.threshold = threshold
        self.projection_matrix = None

    def update(self, features):
        """
        Accumulate the covariance matrix with new features.

        Args:
            features: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        """
        if features.dim() == 3:
            features = features.reshape(-1, features.size(-1))

        features = features.detach().float()

        if self.cov_matrix is None:
            self.cov_matrix = torch.zeros(
                (features.size(-1), features.size(-1)),
                device=features.device,
                dtype=torch.float32,
            )

        cov = torch.matmul(features.t(), features)
        self.cov_matrix += cov
        self.num_samples += features.size(0)

    def compute_projection_matrix(self):
        """
        Compute the projection matrix onto the null space of the covariance matrix.

        Returns:
            projection_matrix: (D, D) matrix that projects gradients onto the null space
        """
        if self.cov_matrix is None:
            return None

        if self.num_samples == 0:
            return None
        cov_norm = self.cov_matrix / self.num_samples
        # CPU SVD also works when the accelerator lacks this linalg kernel.
        if cov_norm.device.type == "npu":
            cov_norm = cov_norm.cpu()
        if not torch.isfinite(cov_norm).all():
            raise ValueError("Non-finite projector covariance")
        if not torch.count_nonzero(cov_norm):
            self.projection_matrix = torch.eye(cov_norm.shape[0], device=cov_norm.device)
            return self.projection_matrix
        U, S, V = torch.linalg.svd(cov_norm)

        # Determine rank by explained variance ratio
        explained_variance = torch.cumsum(S, dim=0) / torch.sum(S)
        rank = torch.searchsorted(explained_variance, self.threshold).item() + 1
        rank = min(rank, S.size(0) - 1)  # Ensure at least 1-dim null space

        # Null space basis: columns of V^T beyond the rank
        null_space = V[rank:].t()  # (D, D - rank)
        self.projection_matrix = torch.matmul(null_space, null_space.t())
        return self.projection_matrix

    def project_gradients(self, gradients):
        """
        Project gradients onto the null space.

        Args:
            gradients: (out_features, in_features) gradient tensor

        Returns:
            Projected gradients of the same shape
        """
        if self.projection_matrix is None:
            return gradients

        proj = self.projection_matrix.to(device=gradients.device, dtype=gradients.dtype)

        # For weight gradients: project along the input dimension
        # grad shape: (out_features, in_features), proj shape: (in_features, in_features)
        if gradients.dim() == 2:
            return torch.matmul(gradients, proj)
        elif gradients.dim() == 1:
            if gradients.size(0) == proj.size(0):
                return torch.matmul(gradients.unsqueeze(0), proj).squeeze(0)
            else:
                return gradients
        else:
            return gradients

    def merge(self, other):
        """
        Merge another NullSpaceProjection's accumulated covariance.
        Used when combining projections from multiple previous tasks.
        """
        if other.cov_matrix is None:
            return
        if self.cov_matrix is None:
            self.cov_matrix = other.cov_matrix.clone()
            self.num_samples = other.num_samples
        else:
            self.cov_matrix += other.cov_matrix.to(self.cov_matrix.device)
            self.num_samples += other.num_samples

    def save(self, path):
        """Save the null-space projection state."""
        state = {
            "cov_matrix": self.cov_matrix,
            "num_samples": self.num_samples,
            "threshold": self.threshold,
            "projection_matrix": self.projection_matrix,
        }
        torch.save(state, path)

    def load(self, path, device="cpu"):
        """Load the null-space projection state."""
        state = torch.load(path, map_location=device)
        self.cov_matrix = state["cov_matrix"]
        self.num_samples = state["num_samples"]
        self.threshold = state.get("threshold", self.threshold)
        self.projection_matrix = state.get("projection_matrix", None)


class NullSpaceGradientHook:
    """
    A gradient hook that projects mm_projector gradients onto the null space
    of previously learned feature representations.

    Usage:
        hook = NullSpaceGradientHook(projection_matrices)
        hook.register(model)
        # ... training ...
        hook.remove()
    """

    def __init__(self, projection_matrices):
        """
        Args:
            projection_matrices: dict mapping parameter names (or layer indices)
                to projection matrices. Each matrix is (D, D).
                For mm_projector, typically we have separate projections for
                each linear layer in the projector.
        """
        self.projection_matrices = projection_matrices
        self.hooks = []

    def _make_hook(self, proj_matrix):
        """Create a gradient hook function for a specific projection matrix."""
        def hook_fn(grad):
            if proj_matrix is None:
                return grad
            proj = proj_matrix.to(device=grad.device, dtype=grad.dtype)
            if grad.dim() == 2:
                return torch.matmul(grad, proj)
            elif grad.dim() == 1:
                # Bias gradient: project in the output dimension
                return grad
            return grad
        return hook_fn

    def register(self, model):
        """Register gradient hooks on mm_projector parameters."""
        for name, param in model.named_parameters():
            if "mm_projector" in name and param.requires_grad:
                # Find the matching projection matrix
                for key, proj_matrix in self.projection_matrices.items():
                    if key in name:
                        handle = param.register_hook(self._make_hook(proj_matrix))
                        self.hooks.append(handle)
                        break

    def remove(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


class CLIPRouter(nn.Module):
    """
    CLIP-based task router for continual learning.

    Supports:
    1) Order-independent single-center routing (K=1) via accumulated feature sums.
    2) Multi-center routing (K>1) where each task owns multiple prototypes.
    """

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        num_centers_per_task=1,
        kmeans_iters=20,
        shared_vision_tower=None,
        k1_center_mode="mean",
        k1_learn_epochs=4,
        k1_learn_batch_size=1024,
        k1_learn_lr=5e-3,
        k1_temperature=0.07,
        k1_sep_margin=0.02,
        k1_lambda_sep=0.5,
        k1_lambda_orth=0.1,
    ):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.num_centers_per_task = max(1, int(num_centers_per_task))
        self.kmeans_iters = max(1, int(kmeans_iters))
        self.k1_center_mode = str(k1_center_mode).lower()
        self.k1_learn_epochs = max(1, int(k1_learn_epochs))
        self.k1_learn_batch_size = max(1, int(k1_learn_batch_size))
        self.k1_learn_lr = float(k1_learn_lr)
        self.k1_temperature = float(k1_temperature)
        self.k1_sep_margin = float(k1_sep_margin)
        self.k1_lambda_sep = float(k1_lambda_sep)
        self.k1_lambda_orth = float(k1_lambda_orth)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
        # Reuse LLaVA vision tower when available to avoid loading a second CLIP vision backbone.
        self._shared_refs = {"vision_tower": shared_vision_tower}
        self.vision_encoder = None
        self.visual_projection = None
        if shared_vision_tower is None:
            self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
            self.visual_projection = self.vision_encoder.visual_projection
        else:
            if hasattr(shared_vision_tower, "image_processor"):
                self.image_processor = shared_vision_tower.image_processor
            # Only keep the lightweight visual projection head; backbone is reused from LLaVA.
            _tmp = CLIPVisionModelWithProjection.from_pretrained(clip_model_name)
            in_dim = int(_tmp.visual_projection.in_features)
            out_dim = int(_tmp.visual_projection.out_features)
            self.visual_projection = nn.Linear(in_dim, out_dim, bias=False)
            self.visual_projection.load_state_dict(_tmp.visual_projection.state_dict(), strict=True)
            del _tmp
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if self.visual_projection is not None:
            for param in self.visual_projection.parameters():
                param.requires_grad = False
        self.text_encoder.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()

        # Task centers: {task_name: (K, embed_dim)} normalized embeddings
        self.task_centers = nn.ParameterDict()
        # Track number of samples accumulated per task.
        self._task_sample_counts = {}
        # For strict order-independent K=1 updates.
        self._task_feature_sums = {}
        # For K>1 center recomputation.
        self._task_feature_buffers = {}

    def _collect_old_task_keys(self, task_name, device, dtype):
        old_keys = []
        for name, centers in self.task_centers.items():
            if name == task_name:
                continue
            c = centers.detach()
            if c.dim() == 2 and c.size(0) > 1:
                c = F.normalize(c.mean(dim=0, keepdim=True), dim=-1)
            elif c.dim() == 1:
                c = c.unsqueeze(0)
            old_keys.append(c.to(device=device, dtype=dtype))
        if not old_keys:
            return torch.empty((0, 0), device=device, dtype=dtype)
        return torch.cat(old_keys, dim=0)

    @torch.enable_grad()
    def _learn_single_center(self, features, old_keys):
        """Learn one task center with discriminative objective (key_router-style)."""
        features = F.normalize(features.float(), dim=-1)
        key = nn.Parameter(F.normalize(features.mean(dim=0, keepdim=True), dim=-1).clone())
        opt = torch.optim.Adam([key], lr=self.k1_learn_lr)
        loader = DataLoader(
            TensorDataset(features),
            batch_size=self.k1_learn_batch_size,
            shuffle=True,
            drop_last=False,
        )

        pbar = tqdm(range(self.k1_learn_epochs), desc="Learn key", ncols=100)
        for _ in pbar:
            for (z,) in loader:
                z = F.normalize(z, dim=-1)
                k = F.normalize(key, dim=-1)
                sim_new = torch.matmul(z, k.t()).squeeze(-1)  # [B]
                if old_keys.numel() > 0:
                    sim_old = torch.matmul(z, old_keys.t())  # [B, T_old]
                    logits = torch.cat([sim_new.unsqueeze(1), sim_old], dim=1) / self.k1_temperature
                    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                    loss_cls = F.cross_entropy(logits, targets)
                    loss_sep = torch.relu(sim_old.max(dim=1).values - sim_new + self.k1_sep_margin).mean()
                    loss_orth = torch.matmul(k, old_keys.t()).pow(2).mean()
                    loss = loss_cls + self.k1_lambda_sep * loss_sep + self.k1_lambda_orth * loss_orth
                else:
                    loss = (1.0 - sim_new).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        return F.normalize(key.detach(), dim=-1)

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        for tag in IMAGE_TAGS:
            text = text.replace(tag, " ")
        text = text.strip()

        # Drop assistant side if a full chat prompt is provided.
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[0]

        # Keep only the last user turn if chat template markers exist.
        user_markers = ["### Human:", "USER:", "Human:", "User:"]
        last_pos = -1
        last_marker = None
        for m in user_markers:
            p = text.rfind(m)
            if p > last_pos:
                last_pos = p
                last_marker = m
        if last_marker is not None and last_pos >= 0:
            text = text[last_pos + len(last_marker) :]

        # Remove common system preamble lines.
        filtered_lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            lower = s.lower()
            if lower.startswith("a chat between"):
                continue
            if lower.startswith("system:") or lower.startswith("### system:"):
                continue
            filtered_lines.append(s)

        text = " ".join(filtered_lines) if filtered_lines else text
        return " ".join(text.split()).strip()

    def encode_text(self, texts, device):
        """
        Encode a list of text strings using CLIP's text encoder.

        Args:
            texts: list of strings
            device: torch device

        Returns:
            embeddings: (N, embed_dim) L2-normalized embeddings
        """
        texts = [self._normalize_text(t) for t in texts]
        inputs = self.tokenizer(
            texts, padding=True, return_tensors="pt", truncation=True, max_length=77
        ).to(device)
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.text_embeds  # (N, embed_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_image(self, images, device, dtype=None):
        """Encode image batch into CLIP image embedding space."""
        if images is None:
            return None

        shared_vision_tower = self._shared_refs.get("vision_tower", None)
        use_shared_backbone = shared_vision_tower is not None

        if torch.is_tensor(images):
            pixel_values = images.to(device=device)
            if pixel_values.dim() == 5:
                # [B, N, C, H, W] -> keep first frame/view
                pixel_values = pixel_values[:, 0]
            if use_shared_backbone and hasattr(shared_vision_tower, "config"):
                image_size = int(getattr(shared_vision_tower.config, "image_size", pixel_values.shape[-1]))
            else:
                image_size = int(getattr(self.vision_encoder.config, "image_size", pixel_values.shape[-1]))
            if pixel_values.shape[-1] != image_size or pixel_values.shape[-2] != image_size:
                pixel_values = F.interpolate(
                    pixel_values.float(), size=(image_size, image_size), mode="bilinear", align_corners=False
                )
            if dtype is None:
                if use_shared_backbone:
                    base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                    dtype = next(base.parameters()).dtype
                else:
                    dtype = next(self.vision_encoder.parameters()).dtype
            pixel_values = pixel_values.to(dtype=dtype)

            if use_shared_backbone:
                base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                outputs = base(pixel_values=pixel_values, output_hidden_states=False)
                pooled = outputs.last_hidden_state[:, 0, :]
                proj = self.visual_projection
                pooled = pooled.to(device=proj.weight.device, dtype=proj.weight.dtype)
                emb = proj(pooled)
            else:
                outputs = self.vision_encoder(pixel_values=pixel_values)
                emb = outputs.image_embeds
            return F.normalize(emb, dim=-1)

        img_inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(device=device)
        if dtype is None:
            if use_shared_backbone:
                base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                dtype = next(base.parameters()).dtype
            else:
                dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=dtype)
        if use_shared_backbone:
            base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
            outputs = base(pixel_values=pixel_values, output_hidden_states=False)
            pooled = outputs.last_hidden_state[:, 0, :]
            proj = self.visual_projection
            pooled = pooled.to(device=proj.weight.device, dtype=proj.weight.dtype)
            emb = proj(pooled)
        else:
            outputs = self.vision_encoder(pixel_values=pixel_values)
            emb = outputs.image_embeds
        return F.normalize(emb, dim=-1)

    def reset_task_center(self, task_name):
        """Remove one task center and its running statistics."""
        if task_name in self.task_centers:
            del self.task_centers[task_name]
        self._task_sample_counts.pop(task_name, None)
        self._task_feature_sums.pop(task_name, None)
        self._task_feature_buffers.pop(task_name, None)

    def update_task_center(self, task_name, texts, device, encode_batch_size=512):
        """
        Update or create the task center embedding.

        The task center is the L2-normalized mean of all text embeddings
        belonging to that task. For incremental updates, we use a running
        weighted average.

        Args:
            task_name: string identifier for the task
            texts: list of text descriptions for this task
            device: torch device
            encode_batch_size: encode chunk size used for CLIP text forward
        """
        if texts is None or len(texts) == 0:
            return

        with torch.no_grad():
            encode_batch_size = max(1, int(encode_batch_size))
            if len(texts) <= encode_batch_size:
                embeddings = self.encode_text(texts, device)
            else:
                vecs = []
                for start in range(0, len(texts), encode_batch_size):
                    chunk = texts[start : start + encode_batch_size]
                    emb = self.encode_text(chunk, device)
                    vecs.append(emb)
                embeddings = torch.cat(vecs, dim=0)
            if embeddings.numel() == 0:
                return
            if self.num_centers_per_task <= 1:
                if self.k1_center_mode == "learned":
                    old_keys = self._collect_old_task_keys(task_name, device=device, dtype=embeddings.dtype)
                    if old_keys.numel() > 0:
                        old_keys = old_keys.to(device=device, dtype=embeddings.dtype)
                    center = self._learn_single_center(embeddings, old_keys)
                else:
                    center = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)
                self.task_centers[task_name] = nn.Parameter(center.detach(), requires_grad=False)
                return

            # K > 1: keep all seen features for this task and recompute centers.
            if task_name not in self._task_feature_buffers:
                self._task_feature_buffers[task_name] = embeddings
            else:
                self._task_feature_buffers[task_name] = torch.cat(
                    [self._task_feature_buffers[task_name], embeddings], dim=0
                )

            self._task_sample_counts[task_name] = int(self._task_feature_buffers[task_name].size(0))
            features = self._task_feature_buffers[task_name]
            centers = self._compute_kmeans_centers(
                features=features,
                num_centers=min(self.num_centers_per_task, features.size(0)),
                num_iters=self.kmeans_iters,
            )
            self.task_centers[task_name] = nn.Parameter(centers.detach(), requires_grad=False)

    def update_task_center_from_embeddings(self, task_name, embeddings, device):
        """Update task center directly from precomputed embeddings [N, D]."""
        if embeddings is None:
            return
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
        if embeddings.numel() == 0:
            return
        embeddings = F.normalize(embeddings.float().to(device), dim=-1)
        if self.num_centers_per_task <= 1:
            if self.k1_center_mode == "learned":
                old_keys = self._collect_old_task_keys(task_name, device=device, dtype=embeddings.dtype)
                if old_keys.numel() > 0:
                    old_keys = old_keys.to(device=device, dtype=embeddings.dtype)
                center = self._learn_single_center(embeddings, old_keys)
            else:
                center = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)
            self.task_centers[task_name] = nn.Parameter(center.detach(), requires_grad=False)
            self._task_sample_counts[task_name] = int(embeddings.size(0))
            return
        centers = self._compute_kmeans_centers(
            features=embeddings,
            num_centers=min(self.num_centers_per_task, embeddings.size(0)),
            num_iters=self.kmeans_iters,
        )
        self.task_centers[task_name] = nn.Parameter(centers.detach(), requires_grad=False)
        self._task_sample_counts[task_name] = int(embeddings.size(0))

    def _compute_kmeans_centers(self, features, num_centers, num_iters):
        """
        Deterministic farthest-point init + k-means refinement.
        """
        features = F.normalize(features.float(), dim=-1)
        if features.size(0) <= num_centers:
            return features[:num_centers]

        mean_vec = features.mean(dim=0, keepdim=True)  # (1, D)
        dists = torch.sum((features - mean_vec) ** 2, dim=1)  # (N,)
        first_idx = torch.argmax(dists).item()
        centers = [features[first_idx:first_idx + 1]]

        for _ in range(1, num_centers):
            all_centers = torch.cat(centers, dim=0)  # (k, D)
            # Pick point farthest to the nearest existing center.
            sim = torch.matmul(features, all_centers.t())  # (N, k)
            min_dist = 1.0 - sim.max(dim=1).values
            next_idx = torch.argmax(min_dist).item()
            centers.append(features[next_idx:next_idx + 1])

        centers = torch.cat(centers, dim=0)  # (K, D)

        for _ in range(num_iters):
            sim = torch.matmul(features, centers.t())  # (N, K)
            assign = torch.argmax(sim, dim=1)  # (N,)
            new_centers = []
            for center_idx in range(centers.size(0)):
                mask = assign == center_idx
                if mask.any():
                    c = features[mask].mean(dim=0, keepdim=True)
                    c = F.normalize(c, dim=-1)
                else:
                    c = centers[center_idx:center_idx + 1]
                new_centers.append(c)
            new_centers = torch.cat(new_centers, dim=0)
            if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        return centers

    def forward(self, texts, device, top_k=1, images=None, image_embeddings=None):
        """
        Route each text sample to the most similar task.

        Args:
            texts: list of text strings
            device: torch device
            top_k: number of top tasks to return (default 1)

        Returns:
            best_tasks: list of task name strings (one per input text)
            similarities: (N, num_tasks) cosine similarity matrix
        """
        if len(self.task_centers) == 0:
            return None, None
        with torch.no_grad():
            text_embeddings = self.encode_text(texts, device)  # (N, Dt)
            sample_center = next(iter(self.task_centers.values()))
            center_dim = int(sample_center.shape[-1])
            text_dim = int(text_embeddings.shape[-1])

            if center_dim == text_dim:
                embeddings = text_embeddings
            elif center_dim == text_dim * 2:
                if image_embeddings is not None:
                    img_emb = image_embeddings.to(device=device, dtype=text_embeddings.dtype)
                    img_emb = F.normalize(img_emb, dim=-1)
                else:
                    image_dtype = None
                    if self.vision_encoder is not None:
                        image_dtype = next(self.vision_encoder.parameters()).dtype
                    else:
                        shared_vision_tower = self._shared_refs.get("vision_tower", None)
                        if shared_vision_tower is not None:
                            base = getattr(shared_vision_tower, "vision_tower", shared_vision_tower)
                            image_dtype = next(base.parameters()).dtype
                    img_emb = self.encode_image(
                        images=images,
                        device=device,
                        dtype=image_dtype,
                    )
                if img_emb is None:
                    img_emb = torch.zeros_like(text_embeddings)
                embeddings = torch.cat([text_embeddings, img_emb.to(text_embeddings.dtype)], dim=-1)
                embeddings = F.normalize(embeddings, dim=-1)
            else:
                # Fallback: align dimensions by truncating or zero-padding text embeddings.
                if center_dim < text_dim:
                    embeddings = text_embeddings[:, :center_dim]
                else:
                    pad = torch.zeros(
                        (text_embeddings.size(0), center_dim - text_dim),
                        device=text_embeddings.device,
                        dtype=text_embeddings.dtype,
                    )
                    embeddings = torch.cat([text_embeddings, pad], dim=-1)
                embeddings = F.normalize(embeddings, dim=-1)
        task_names = list(self.task_centers.keys())
        task_similarities = []
        for task_name in task_names:
            centers = self.task_centers[task_name].to(device=device, dtype=embeddings.dtype)  # (K, D)
            sim_to_centers = torch.matmul(embeddings.float(), centers.float().t())  # (N, K)
            task_score = sim_to_centers.max(dim=1).values  # (N,)
            task_similarities.append(task_score.unsqueeze(1))
        similarities = torch.cat(task_similarities, dim=1)  # (N, num_tasks)

        if top_k == 1:
            best_indices = torch.argmax(similarities, dim=-1)  # (N,)
            best_tasks = [task_names[i.item()] for i in best_indices]
        else:
            topk_indices = torch.topk(similarities, k=min(top_k, len(task_names)), dim=-1).indices
            best_tasks = [
                [task_names[j.item()] for j in topk_indices[i]]
                for i in range(len(texts))
            ]

        return best_tasks, similarities

    def get_task_index(self, task_name):
        """Get the index of a task in the task center list."""
        task_names = list(self.task_centers.keys())
        if task_name in task_names:
            return task_names.index(task_name)
        return -1

    def get_num_tasks(self):
        """Return the number of registered tasks."""
        return len(self.task_centers)

    def save_state(self, save_dir):
        """Save router state (task centers and sample counts)."""
        os.makedirs(save_dir, exist_ok=True)

        # Save task centers
        centers_dict = {
            name: param.data.cpu() for name, param in self.task_centers.items()
        }
        torch.save(centers_dict, os.path.join(save_dir, "task_centers.pt"))

        # Save sample counts
        with open(os.path.join(save_dir, "task_sample_counts.json"), "w") as f:
            json.dump(self._task_sample_counts, f)

        # Save minimal router config used for future loads.
        with open(os.path.join(save_dir, "router_config.json"), "w") as f:
            json.dump(
                {
                    "clip_model_name": self.clip_model_name,
                    "num_centers_per_task": self.num_centers_per_task,
                    "kmeans_iters": self.kmeans_iters,
                    "k1_center_mode": self.k1_center_mode,
                },
                f,
            )

    def load_state(self, save_dir, device="cpu"):
        """Load router state."""
        config_path = os.path.join(save_dir, "router_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.num_centers_per_task = int(cfg.get("num_centers_per_task", self.num_centers_per_task))
            self.kmeans_iters = int(cfg.get("kmeans_iters", self.kmeans_iters))
            self.k1_center_mode = str(cfg.get("k1_center_mode", self.k1_center_mode)).lower()

        centers_path = os.path.join(save_dir, "task_centers.pt")
        if os.path.exists(centers_path):
            centers_dict = torch.load(centers_path, map_location=device)
            for name, center in centers_dict.items():
                self.task_centers[name] = nn.Parameter(
                    center.to(device), requires_grad=False
                )

        counts_path = os.path.join(save_dir, "task_sample_counts.json")
        if os.path.exists(counts_path):
            with open(counts_path, "r") as f:
                self._task_sample_counts = json.load(f)


class DrapeModel(torch.nn.Module):
    def __init__(self, model, config: DrapeConfig, adapter_name: str = "default"):
        super().__init__()
        # model: LlavaLlamaForCausalLM
        self.model = model
        self.peft_config = config
        self.active_adapter = adapter_name
        if isinstance(config, DrapeConfig):
            config_dict = {adapter_name: config}
        else:
            config_dict = config
        active_config = config_dict[adapter_name]

        # Initialize the Generator Network for the current task
        self.drape_generator = DrapeGenerator(active_config)

        base_model_dtype = next(model.parameters()).dtype
        base_model_device = next(model.parameters()).device
        self.dtype = base_model_dtype
        # CHANGED(fp32-generator): keep the current trainable generator in fp32.
        # This isolates generator-side attention from the base model dtype and
        # avoids bf16/fp16 NaNs in the current task prompt path.
        self.drape_generator = self.drape_generator.to(
            device=base_model_device, dtype=torch.float32
        )

        self.word_embeddings = self.model.get_input_embeddings()
        self.base_model_prepare_inputs_for_generation = (
            self.model.prepare_inputs_for_generation
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # Ensure Generator Parameters are Trainable
        for param in self.drape_generator.parameters():
            param.requires_grad = True

        # === Continual Learning Components ===
        self.enable_cl = active_config.enable_continual_learning
        self.current_task_name = active_config.current_task_name
        self.apply_null_space_projection = getattr(
            active_config, "apply_null_space_projection", True
        )

        # Store previous task generators (frozen, for inference routing)
        # Dict: task_name -> DrapeGenerator (frozen)
        self.previous_generators = nn.ModuleDict()

        # CLIP-based router for task routing
        self.clip_router = None

        # Null-space projection for mm_projector anti-forgetting
        self.null_space_projections = {}  # layer_name -> NullSpaceProjection
        self.null_space_stats = {}  # layer_name -> {"cov_matrix", "num_samples", "threshold"}
        self.null_space_hook = None

        # Feature collector for computing null-space after training
        self._feature_collectors = {}  # layer_name -> NullSpaceProjection
        self._feature_hooks = []

        # Optional override for oracle routing ablation.
        self.forced_task_name = None
        self._pending_routing_texts = None
        self._pending_route_in_prepare = False
        # CPU cache for fused [text; image] features collected during training.
        self._task_fused_cache = {}
        self._task_fused_cache_counts = {}
        self._task_fused_cache_max_samples = 500000

        self._efficiency_profile_enabled = False
        self.reset_efficiency_profile()

        if self.enable_cl:
            self._init_continual_learning(active_config, base_model_device)

    def train(self, mode=True):
        super().train(mode)
        self.previous_generators.eval()
        if self.clip_router is not None:
            self.clip_router.eval()
        self.model.get_vision_tower().eval()
        return self

    def encode_images(self, images):
        tower = self.model.get_vision_tower()
        with torch.no_grad():
            outputs = tower.vision_tower(
                images.to(device=tower.device, dtype=tower.dtype),
                output_hidden_states=True,
            )
            patches = tower.feature_select(outputs).to(images.dtype)
            clip_features = outputs.image_embeds
        projector = self.model.get_model().mm_projector
        weight = next(projector.parameters())
        projected = projector(patches.to(device=weight.device, dtype=weight.dtype))
        return clip_features, projected

    def set_efficiency_profile_enabled(self, enabled: bool = True):
        self._efficiency_profile_enabled = enabled

    def reset_efficiency_profile(self):
        self._efficiency_profile = {
            "vision_prepare_time": 0.0,
            "routing_time": 0.0,
            "prompt_construction_time": 0.0,
            "calls": 0,
        }

    def get_efficiency_profile(self):
        return dict(getattr(self, "_efficiency_profile", {}))

    def _efficiency_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _efficiency_start(self):
        if not getattr(self, "_efficiency_profile_enabled", False):
            return None
        self._efficiency_sync()
        return time.perf_counter()

    def _efficiency_add(self, key, start):
        if start is None:
            return
        self._efficiency_sync()
        self._efficiency_profile[key] = self._efficiency_profile.get(key, 0.0) + (time.perf_counter() - start)

    def _init_continual_learning(self, config, device):
        """Initialize continual learning components."""
        if CLIPRouter is None:
            raise ImportError("CLIPRouter requires transformers with CLIP support")

        shared_vision_tower = None
        if hasattr(self.model, "get_vision_tower"):
            try:
                candidate_vision_tower = self.model.get_vision_tower()
                if candidate_vision_tower is not None and candidate_vision_tower.__class__.__name__ == "CLIPVisionTower":
                    shared_vision_tower = candidate_vision_tower
            except Exception:
                shared_vision_tower = None

        # Initialize CLIP router
        self.clip_router = CLIPRouter(
            clip_model_name=config.clip_model_name,
            num_centers_per_task=config.router_num_centers,
            kmeans_iters=config.router_kmeans_iters,
            shared_vision_tower=shared_vision_tower,
            k1_center_mode=config.router_k1_center_mode,
        )
        self.clip_router = self.clip_router.to(device)
        self._bind_router_shared_vision_tower()

        # Load previous task state if available
        prev_dir = config.previous_task_model_dir
        if prev_dir is not None and prev_dir not in ("", "None", "null", "NO_PREVIOUS_MODEL") and os.path.isdir(prev_dir):
            self._load_previous_task_state(prev_dir, device)

        # Keep current-run setting authoritative even if loaded router config differs.
        self.clip_router.k1_center_mode = str(config.router_k1_center_mode).lower()

        # Initialize null-space feature collector
        if self.apply_null_space_projection and NullSpaceProjection is not None:
            self._setup_projector_feature_collectors(config)
        elif not self.apply_null_space_projection:
            print("[CL] Null-space pipeline disabled: skip feature collector setup.")

        # Setup null-space gradient projection hook for mm_projector
        self._setup_null_space_hooks()

    def _bind_router_shared_vision_tower(self):
        """Bind LLaVA vision tower to router so routing reuses the same visual backbone."""
        if self.clip_router is None:
            return
        if not hasattr(self.model, "get_vision_tower"):
            return
        try:
            vt = self.model.get_vision_tower()
        except Exception:
            vt = None
        if vt is None or vt.__class__.__name__ != "CLIPVisionTower":
            if hasattr(self.clip_router, "_shared_refs"):
                self.clip_router._shared_refs["vision_tower"] = None
            return
        if hasattr(self.clip_router, "_shared_refs"):
            self.clip_router._shared_refs["vision_tower"] = vt
            if not getattr(self.clip_router, "_reuse_log_printed", False):
                print("[CL] Router reuses LLaVA vision tower (shared CLIP vision model).")
                self.clip_router._reuse_log_printed = True

    def _setup_projector_feature_collectors(self, config):
        """
        Register forward hooks on each Linear layer in mm_projector to collect
        its *input* features for null-space computation.
        """
        # Clear existing hooks (if any)
        for handle in self._feature_hooks:
            handle.remove()
        self._feature_hooks = []
        self._feature_collectors = {}

        mm_projector = getattr(self.model.get_model(), "mm_projector", None)
        if mm_projector is None:
            return

        for name, module in mm_projector.named_modules():
            if isinstance(module, nn.Linear):
                key = f"mm_projector.{name}"
                self._feature_collectors[key] = NullSpaceProjection(
                    threshold=config.null_space_threshold
                )

                def make_hook(k):
                    def hook_fn(mod, inp, out):
                        if not self.training or inp is None or len(inp) == 0:
                            return
                        x = inp[0]
                        if x is not None:
                            self._feature_collectors[k].update(x)
                    return hook_fn

                handle = module.register_forward_hook(make_hook(key))
                self._feature_hooks.append(handle)

    def _pack_null_space_state(self):
        """Serialize null-space projections and cumulative statistics."""
        packed_proj = {}
        for key, proj in self.null_space_projections.items():
            if torch.is_tensor(proj):
                packed_proj[key] = proj.detach().cpu()
            else:
                packed_proj[key] = proj

        packed_stats = {}
        for key, stat in self.null_space_stats.items():
            if not isinstance(stat, dict):
                continue
            cov = stat.get("cov_matrix", None)
            packed_stats[key] = {
                "cov_matrix": cov.detach().cpu() if torch.is_tensor(cov) else cov,
                "num_samples": int(stat.get("num_samples", 0)),
                "threshold": float(stat.get("threshold", self.active_peft_config.null_space_threshold)),
            }

        return {
            "format_version": 2,
            "projection_matrices": packed_proj,
            "statistics": packed_stats,
        }

    def _load_null_space_state(self, ns_state):
        """
        Load null-space state with backward compatibility.
        Old format: dict[layer_name] = projection_matrix
        New format: {"projection_matrices": ..., "statistics": ...}
        """
        self.null_space_projections = {}
        self.null_space_stats = {}

        if not isinstance(ns_state, dict):
            return

        # New format
        if "projection_matrices" in ns_state:
            raw_proj = ns_state.get("projection_matrices", {})
            if isinstance(raw_proj, dict):
                self.null_space_projections = raw_proj

            raw_stats = ns_state.get("statistics", {})
            if isinstance(raw_stats, dict):
                for key, stat in raw_stats.items():
                    if not isinstance(stat, dict):
                        continue
                    self.null_space_stats[key] = {
                        "cov_matrix": stat.get("cov_matrix", None),
                        "num_samples": int(stat.get("num_samples", 0)),
                        "threshold": float(
                            stat.get("threshold", self.active_peft_config.null_space_threshold)
                        ),
                    }
            return

        # Old format: projection matrices only
        self.null_space_projections = ns_state

    def _load_previous_task_state(self, prev_dir, device):
        """Load previous task generators, router state, and null-space projections."""
        # 1. Load router state (task centers)
        router_dir = os.path.join(prev_dir, "router")
        if os.path.exists(router_dir):
            self.clip_router.load_state(router_dir, device=device)
            print(f"[CL] Loaded router state from {router_dir}, "
                  f"tasks: {list(self.clip_router.task_centers.keys())}")

        # 2. Load previous task generators
        generators_dir = os.path.join(prev_dir, "task_generators")
        if os.path.exists(generators_dir):
            gen_config_path = os.path.join(generators_dir, "generators_config.json")
            if os.path.exists(gen_config_path):
                with open(gen_config_path, "r") as f:
                    gen_config = json.load(f)

                for task_name in gen_config.get("task_names", []):
                    task_weight_path = os.path.join(generators_dir, f"{task_name}.bin")
                    if os.path.exists(task_weight_path):
                        # Create a generator with same config and load weights
                        active_config = self.peft_config
                        if isinstance(active_config, dict):
                            active_config = active_config[self.active_adapter]
                        prev_gen = DrapeGenerator(active_config)
                        prev_gen.load_state_dict(
                            torch.load(task_weight_path, map_location=device),strict=False
                        )
                        # CHANGED(fp32-generator): keep loaded historical generators in fp32.
                        prev_gen = prev_gen.to(device=device, dtype=torch.float32)
                        # Freeze previous generator
                        for param in prev_gen.parameters():
                            param.requires_grad = False
                        self.previous_generators[task_name] = prev_gen
                        print(f"[CL] Loaded previous generator: {task_name}")

        # Also load the immediately previous task's generator from adapter_model.bin
        prev_adapter_path = os.path.join(prev_dir, "adapter_model.bin")
        prev_config_path = os.path.join(prev_dir, "adapter_config.json")
        if os.path.exists(prev_adapter_path) and os.path.exists(prev_config_path):
            with open(prev_config_path, "r") as f:
                prev_cfg = json.load(f)
            prev_task_name = prev_cfg.get("current_task_name", None)
            if prev_task_name and prev_task_name not in self.previous_generators:
                active_config = self.peft_config
                if isinstance(active_config, dict):
                    active_config = active_config[self.active_adapter]
                prev_gen = DrapeGenerator(active_config)
                prev_gen.load_state_dict(
                    torch.load(prev_adapter_path, map_location=device),strict=False
                )
                # CHANGED(fp32-generator): keep the immediate previous task generator in fp32.
                prev_gen = prev_gen.to(device=device, dtype=torch.float32)
                for param in prev_gen.parameters():
                    param.requires_grad = False
                self.previous_generators[prev_task_name] = prev_gen
                print(f"[CL] Loaded previous task generator: {prev_task_name}")

        # 3. Load null-space projection matrices
        if self.apply_null_space_projection:
            ns_path = os.path.join(prev_dir, "null_space_projection.pt")
            if os.path.exists(ns_path):
                ns_state = torch.load(ns_path, map_location=device)
                self._load_null_space_state(ns_state)
                print(f"[CL] Loaded null-space projections from {ns_path}")
        else:
            print("[CL] Null-space pipeline disabled: skip loading null-space state.")

    def _setup_null_space_hooks(self):
        """Register gradient hooks on mm_projector to project gradients onto null space."""
        if not self.apply_null_space_projection:
            if self.null_space_hook is not None:
                self.null_space_hook.remove()
                self.null_space_hook = None
            print("[CL] Null-space gradient projection is disabled by config.")
            return

        if not self.null_space_projections:
            return

        if self.null_space_hook is not None:
            self.null_space_hook.remove()

        if NullSpaceGradientHook is not None:
            self.null_space_hook = NullSpaceGradientHook(self.null_space_projections)
            self.null_space_hook.register(self.model)
            print("[CL] Registered null-space gradient projection hooks on mm_projector")

    def compute_and_save_null_space(self, save_dir):
        """
        After training on the current task, compute the null-space projection
        and save it. This should be called before moving to the next task.
        """
        if not self.apply_null_space_projection:
            return

        if not self._feature_collectors:
            return

        # Compute projection for each Linear layer in mm_projector
        for key, collector in self._feature_collectors.items():
            # Merge cumulative statistics from previous tasks (if available)
            prev_stat = self.null_space_stats.get(key, None)
            if isinstance(prev_stat, dict):
                prev_cov = prev_stat.get("cov_matrix", None)
                prev_num_samples = int(prev_stat.get("num_samples", 0))
                if torch.is_tensor(prev_cov) and prev_num_samples > 0:
                    if collector.cov_matrix is None:
                        collector.cov_matrix = prev_cov.to(dtype=torch.float32).clone()
                        collector.num_samples = prev_num_samples
                    else:
                        collector.cov_matrix = (
                            collector.cov_matrix
                            + prev_cov.to(
                                device=collector.cov_matrix.device,
                                dtype=collector.cov_matrix.dtype,
                            )
                        )
                        collector.num_samples += prev_num_samples

            proj_matrix = collector.compute_projection_matrix()
            if proj_matrix is None:
                continue
            # Projection computed from cumulative covariance (previous + current task)
            self.null_space_projections[key] = proj_matrix.to(dtype=self.dtype).cpu()
            self.null_space_stats[key] = {
                "cov_matrix": collector.cov_matrix.detach().cpu()
                if collector.cov_matrix is not None
                else None,
                "num_samples": int(collector.num_samples),
                "threshold": float(collector.threshold),
            }

        # Save
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                self._pack_null_space_state(),
                os.path.join(save_dir, "null_space_projection.pt"),
            )

    def update_task_center(self, task_name, texts, device, encode_batch_size=64):
        """Update the task center in the CLIP router."""
        if self.clip_router is not None:
            self.clip_router.update_task_center(
                task_name, texts, device, encode_batch_size=encode_batch_size
            )

    def update_task_center_from_embeddings(self, task_name, embeddings, device):
        """Update task center in CLIP router from precomputed embeddings."""
        if self.clip_router is not None:
            self.clip_router.update_task_center_from_embeddings(task_name, embeddings, device)

    @torch.no_grad()
    def cache_fused_embeddings(self, task_name, route_texts, clip_image_features):
        """Cache fused embeddings on GPU during training to avoid post-train re-extraction."""
        if self.clip_router is None:
            return
        if route_texts is None or clip_image_features is None:
            return
        if not torch.is_tensor(clip_image_features) or clip_image_features.dim() != 2:
            return
        if not isinstance(route_texts, (list, tuple)):
            return
        if len(route_texts) == 0 or clip_image_features.size(0) != len(route_texts):
            return

        cache_dtype = clip_image_features.dtype
        device = clip_image_features.device
        text_emb = self.clip_router.encode_text(list(route_texts), device=device)
        text_emb = text_emb.to(dtype=cache_dtype)
        img_emb = F.normalize(clip_image_features.to(dtype=cache_dtype), dim=-1)
        fused = torch.cat([text_emb, img_emb], dim=-1)
        fused = F.normalize(fused, dim=-1).detach()
        fused = fused.to(dtype=torch.float32, device="cpu")

        self._task_fused_cache.setdefault(task_name, []).append(fused)
        self._task_fused_cache_counts[task_name] = self._task_fused_cache_counts.get(task_name, 0) + fused.size(0)

        # Keep memory bounded: keep only the most recent max_samples.
        count = self._task_fused_cache_counts[task_name]
        if count > self._task_fused_cache_max_samples:
            merged = torch.cat(self._task_fused_cache[task_name], dim=0)
            merged = merged[-self._task_fused_cache_max_samples :]
            self._task_fused_cache[task_name] = [merged]
            self._task_fused_cache_counts[task_name] = merged.size(0)

    def pop_cached_fused_embeddings(self, task_name):
        """Pop cached CPU fused embeddings for a task."""
        chunks = self._task_fused_cache.pop(task_name, None)
        self._task_fused_cache_counts.pop(task_name, None)
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _build_zero_clip_image_features(self, route_texts, device):
        """
        Build zero image embeddings so fused feature construction matches
        key_router behavior when an image is missing.
        """
        if self.clip_router is None:
            return None
        if not isinstance(route_texts, (list, tuple)) or len(route_texts) == 0:
            return None
        try:
            proj_dim = int(self.clip_router.text_encoder.config.projection_dim)
        except Exception:
            return None
        if proj_dim <= 0:
            return None
        base_dtype = next(self.model.parameters()).dtype
        return torch.zeros((len(route_texts), proj_dim), dtype=base_dtype, device=device)

    def set_forced_task_name(self, task_name: Optional[str]):
        """Force routing to a specified task generator (oracle mode)."""
        self.forced_task_name = task_name

    def _get_generator_for_inference(self, texts, device, images=None, image_embeddings=None):
        """
        During inference, use the CLIP router to select the appropriate
        task-specific generator for each sample.

        Returns:
            generator: the DrapeGenerator to use
            task_name: the selected task name
        """
        if self.forced_task_name:
            task_name = self.forced_task_name
            if task_name == self.current_task_name:
                return self.drape_generator, task_name
            if task_name in self.previous_generators:
                return self.previous_generators[task_name], task_name
            return self.drape_generator, self.current_task_name

        if self.clip_router is None or self.clip_router.get_num_tasks() == 0:
            return self.drape_generator, self.current_task_name

        best_tasks, similarities = self.clip_router(
            texts, device, images=images, image_embeddings=image_embeddings
        )
        if best_tasks is None:
            return self.drape_generator, self.current_task_name

        # For batch processing, pick the most common task
        # (In practice, samples in a batch usually belong to the same task)
        task_name = best_tasks[0] if isinstance(best_tasks, list) else best_tasks

        if task_name == self.current_task_name:
            return self.drape_generator, task_name
        elif task_name in self.previous_generators:
            return self.previous_generators[task_name], task_name
        else:
            # Fallback to current generator
            return self.drape_generator, self.current_task_name

    def _build_text_only_inputs_with_prompts(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        """Build prompt-prepended inputs for text-only samples."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        inputs_embeds = self.model.get_model().embed_tokens(input_ids)
        prompts = self.get_prompt(
            input_ids=input_ids,
            image_features=None,
            labels=labels,
            attention_mask=attention_mask,
        )
        # Separate mode with no image degrades to a single prompt stream.
        if isinstance(prompts, tuple):
            prompts = prompts[1]

        batch_size, num_gen_tokens, _ = prompts.shape
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, num_gen_tokens),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat((prefix_labels, labels), dim=1)

        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        prefix_attention_mask = torch.ones(
            batch_size,
            num_gen_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        return inputs_embeds, attention_mask, labels

    def prepare_prompt_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        return_dict=None,
        **kwargs,
    ):
        _labels = labels
        _attention_mask = attention_mask
        route_texts = kwargs.pop("route_texts", None)
        kwargs.pop("image_sizes", None)

        # Text-only always uses a single prompt stream (including separate mode fallback).
        if images is None:
            if self.training and self.enable_cl and route_texts is not None:
                zero_clip_image_features = self._build_zero_clip_image_features(
                    route_texts=route_texts,
                    device=input_ids.device,
                )
                if zero_clip_image_features is not None:
                    self.cache_fused_embeddings(
                        task_name=self.current_task_name,
                        route_texts=route_texts,
                        clip_image_features=zero_clip_image_features,
                    )
            inputs_embeds, attention_mask, labels = self._build_text_only_inputs_with_prompts(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=_labels,
            )
        elif self.drape_generator.separate:
            (
                _input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features,
            ) = self.prepare_inputs_labels_for_multimodal_du(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                _labels,
                images,
            )
        else:
            (
                _input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features,
                _clip_image_features,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                _labels,
                images,
            )
            if self.training and self.enable_cl and route_texts is not None:
                self.cache_fused_embeddings(
                    task_name=self.current_task_name,
                    route_texts=route_texts,
                    clip_image_features=_clip_image_features,
                )

            prompt_source_embeds = None
            prompt_source_labels = None
            prompt_source_attention_mask = None
            if getattr(self.drape_generator, "query_mode", "segpool") == "learnable_mm":
                prompt_source_embeds = inputs_embeds
                prompt_source_labels = labels
                prompt_source_attention_mask = attention_mask

            prompts = self.get_prompt(
                input_ids=input_ids,
                image_features=image_features,
                labels=_labels,
                attention_mask=_attention_mask,
                prompt_source_embeds=prompt_source_embeds,
                prompt_source_labels=prompt_source_labels,
                prompt_source_attention_mask=prompt_source_attention_mask,
            )
            batch_size, num_gen_tokens, _ = prompts.shape
            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, num_gen_tokens),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat((prefix_labels, labels), dim=1)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(
                    batch_size,
                    num_gen_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1
                )

        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )
        kwargs.update(inputs_embeds=inputs_embeds, position_ids=None)
        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_prompt_inputs(**kwargs))

    def get_prompt(
        self, input_ids=None, image_features=None, labels=None, attention_mask=None,
        generator_override=None, prompt_source_embeds=None,
        prompt_source_labels=None, prompt_source_attention_mask=None,
        return_attn: bool = False,
    ):
        """
        Generate prompts using the Drape generator.

        Args:
            generator_override: if provided, use this generator instead of the default.
                Used during inference when the router selects a previous task's generator.
        """
        generator = generator_override if generator_override is not None else self.drape_generator

        if (
            not generator.separate
            and getattr(generator, "query_mode", "segpool") == "learnable_mm"
            and prompt_source_embeds is not None
        ):
            visual_features = None
            text_features = prompt_source_embeds
            if prompt_source_attention_mask is None:
                mask = torch.ones(
                    (text_features.size(0), text_features.size(1)),
                    dtype=torch.bool,
                    device=text_features.device,
                )
            else:
                mask = prompt_source_attention_mask.bool().to(text_features.device)
            if prompt_source_labels is not None:
                # Avoid answer leakage: only non-target positions can be attended by prompt queries.
                mask = mask & (prompt_source_labels == IGNORE_INDEX).to(mask.device)
        else:
            visual_features = image_features
            infer = False if labels is not None else True
            mask_labels = None
            if labels is not None:
                mask_labels = labels == IGNORE_INDEX
            _, _, s_mask, u_mask, n_label_mask = split_system_user(
                input_ids, inference=infer
            )
            texts_ids = input_ids
            mask = input_ids != IMAGE_TOKEN_INDEX
            if attention_mask is not None:
                mask = mask & attention_mask.bool()
            texts_ids = texts_ids.masked_fill(~mask, 0)
            if mask_labels is not None:
                mask = mask & mask_labels

            text_emdeds = self.word_embeddings(texts_ids)
            text_features = text_emdeds

        device = next(generator.parameters()).device
        text_features = text_features.to(device)
        mask = mask.to(device)
        if visual_features is not None:
            visual_features = visual_features.to(device)

        if generator.separate:
            generated = generator(
                vision_features=visual_features,
                text_features=text_features,
                mask=mask,
                return_attn=return_attn,
            )
            return generated
        else:
            prompts = generator(
                vision_features=visual_features,
                text_features=text_features,
                mask=mask,
                return_attn=return_attn,
            )
            return prompts

    def _get_generator_by_task_name(self, task_name: Optional[str]):
        """Resolve the generator used for visualization or oracle routing."""
        if task_name is None or task_name == self.current_task_name:
            return self.drape_generator, self.current_task_name
        if task_name in self.previous_generators:
            return self.previous_generators[task_name], task_name
        raise KeyError(f"Unknown task generator: {task_name}")

    @torch.no_grad()
    def extract_prompt_image_attention(
        self,
        input_ids,
        images,
        attention_mask=None,
        labels=None,
        route_texts=None,
        task_name: Optional[str] = None,
    ):
        """Return generated prompt embeddings and prompt-to-image attention weights."""
        if images is None:
            raise ValueError("images must be provided for prompt-to-image attention extraction")
        if self.drape_generator.separate:
            raise NotImplementedError("Prompt attention visualization currently supports non-separate DRAPE only")

        (
            _,
            _,
            mm_attention_mask,
            _,
            inputs_embeds,
            mm_labels,
            image_features,
            clip_image_features,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            labels,
            images,
        )

        selected_task_name = self.current_task_name
        if task_name is not None:
            generator, selected_task_name = self._get_generator_by_task_name(task_name)
        elif self.enable_cl and route_texts is not None and self.clip_router is not None:
            device = next(self.model.parameters()).device
            generator, selected_task_name = self._get_generator_for_inference(
                route_texts,
                device,
                images=None,
                image_embeddings=clip_image_features,
            )
        else:
            generator = self.drape_generator

        prompt_source_embeds = None
        prompt_source_labels = None
        prompt_source_attention_mask = None
        if getattr(generator, "query_mode", "segpool") == "learnable_mm":
            prompt_source_embeds = inputs_embeds
            prompt_source_labels = mm_labels
            prompt_source_attention_mask = mm_attention_mask

        prompt_outputs = self.get_prompt(
            input_ids=input_ids,
            image_features=image_features,
            labels=labels,
            attention_mask=attention_mask,
            generator_override=generator,
            prompt_source_embeds=prompt_source_embeds,
            prompt_source_labels=prompt_source_labels,
            prompt_source_attention_mask=prompt_source_attention_mask,
            return_attn=True,
        )

        if not isinstance(prompt_outputs, tuple) or len(prompt_outputs) != 2:
            raise RuntimeError("Expected get_prompt(return_attn=True) to return (prompts, attention_weights)")
        prompts, attention_weights = prompt_outputs
        if attention_weights is None:
            raise RuntimeError(
                "Prompt-to-image attention weights are unavailable for the selected DRAPE configuration"
            )

        return {
            "prompts": prompts,
            "attention_weights": attention_weights,
            "image_features": image_features,
            "selected_task_name": selected_task_name,
            "image_features_shape": list(image_features.shape) if torch.is_tensor(image_features) else None,
            "clip_image_features_shape": list(clip_image_features.shape) if torch.is_tensor(clip_image_features) else None,
        }

    @torch.no_grad()
    def generate(self, input_ids=None, images=None, texts=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if input_ids is None:
            raise ValueError("Drape generation requires input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError("Drape routes each sample independently; use inference batch_size=1")
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("Pass token IDs and images to Drape, not precomputed inputs_embeds")
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("image_sizes", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        original_generator = self.drape_generator
        try:
            if images is None:
                generator, _ = self._get_generator_for_inference(texts, input_ids.device)
                self.drape_generator = generator
                embeds, mask, _ = self._build_text_only_inputs_with_prompts(
                    input_ids, attention_mask=attention_mask, labels=None,
                )
            else:
                (_, _, mask, _, embeds, _, features, clip_features) = (
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids, None, attention_mask, None, None, images,
                    )
                )
                generator, _ = self._get_generator_for_inference(
                    texts, input_ids.device, image_embeddings=clip_features,
                )
                prompts = self.get_prompt(
                    input_ids=input_ids, image_features=features,
                    attention_mask=attention_mask, generator_override=generator,
                    prompt_source_embeds=embeds, prompt_source_attention_mask=mask,
                ).to(embeds.dtype)
                embeds = torch.cat((prompts, embeds), dim=1)
                mask = torch.cat((mask.new_ones((1, prompts.shape[1])), mask), dim=1)
            # Keep the original token prefix in the output: Prism's decoders slice it off.
            # Negative image placeholders are metadata, never decoder embedding indices.
            return self.model.generate(
                input_ids=input_ids.clamp_min(0), inputs_embeds=embeds,
                attention_mask=mask, **kwargs,
            )
        finally:
            self.drape_generator = original_generator

    def generate_with_routing(self, texts=None, **kwargs):
        return self.generate(texts=texts, **kwargs)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.model.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
                None,
            )

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            clip_image_features, image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            clip_image_features = torch.stack([chunk.mean(0) for chunk in torch.split(clip_image_features, split_sizes)])
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            clip_image_features, image_features = self.encode_images(images)
            if self.enable_cl and self.clip_router is not None and hasattr(self.clip_router, "_shared_refs") and self.clip_router._shared_refs.get("vision_tower", None) is None:
                router_device = next(self.model.parameters()).device
                router_dtype = None
                try:
                    if self.clip_router.vision_encoder is not None:
                        router_dtype = next(self.clip_router.vision_encoder.parameters()).dtype
                    elif self.clip_router.visual_projection is not None:
                        router_dtype = next(self.clip_router.visual_projection.parameters()).dtype
                except Exception:
                    router_dtype = None
                clip_image_features = self.clip_router.encode_image(images, device=router_device, dtype=router_dtype)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.model.config, "tune_mm_mlp_adapter", False) and getattr(
            self.model.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.model.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.model.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            image_features,
            clip_image_features,
        )

    def prepare_inputs_labels_for_multimodal_du(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        # image_prompts=None,text_prompts=None
    ):
        vision_tower = self.model.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                None,
            )

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.model.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            clip_image_features = torch.stack([chunk.mean(0) for chunk in torch.split(clip_image_features, split_sizes)])
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [
                x.flatten(0, 1).to(self.model.device) for x in image_features
            ]
        else:
            image_features = self.model.encode_images(images).to(self.model.device)

        # =====================================================================
        generated_prompts = self.get_prompt(
            input_ids=input_ids, image_features=image_features, labels=labels
        )
        if isinstance(generated_prompts, tuple):
            image_prompts, text_prompts = generated_prompts
        else:
            # Separate text-only fallback (should rarely happen when image_features exists).
            image_prompts, text_prompts = generated_prompts, generated_prompts

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.model.config, "tune_mm_mlp_adapter", False) and getattr(
            self.model.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        # --- 核心循环修改开始 ---
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 获取当前样本的 Prompts 和 Label (-100)
            cur_img_prompt = image_prompts[batch_idx]
            cur_txt_prompt = text_prompts[batch_idx]

            cur_img_prompt_lbl = torch.full(
                (cur_img_prompt.shape[0],),
                IGNORE_INDEX,
                device=labels[batch_idx].device,
                dtype=labels[batch_idx].dtype,
            )
            cur_txt_prompt_lbl = torch.full(
                (cur_txt_prompt.shape[0],),
                IGNORE_INDEX,
                device=labels[batch_idx].device,
                dtype=labels[batch_idx].dtype,
            )

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            # Case 1: 无图样本
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.model.get_model().embed_tokens(cur_input_ids)

                # 即使无图，也在最前面插入 Text Prompt (作为纯文本任务的 Prefix)
                # 结构: [Text Prompt] -> [Original Input]
                cur_input_embeds = torch.cat(
                    [cur_txt_prompt, cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)

                cur_labels_with_prompt = torch.cat(
                    [cur_txt_prompt_lbl, labels[batch_idx]], dim=0
                )
                new_labels.append(cur_labels_with_prompt)

                cur_image_idx += 1
                continue

            # Case 2: 有图样本
            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            # 提取所有非图像部分的 Token ID 和 Label
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.model.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            # 遍历每一个片段 (Segment)
            # 结构逻辑:
            # Segment 0 (User Role) -> [Img Prompt] -> Image 1 -> Image 2... -> [Txt Prompt] -> Segment N (User Text)

            for i in range(num_images + 1):
                # 【逻辑点 2】：Text Prompt 在用户prompt前面插入
                # 此时，刚插入完图片，后面紧接着就是 Segment[i+1] (即用户文本)
                if i == num_images:
                    cur_new_input_embeds.append(
                        cur_input_embeds_no_im[i][:1]
                    )  # 这里插入文本的token，第一个是 '\n'
                    cur_new_labels.append(
                        cur_labels_noim[i][:1]
                    )  # 先加入进去再加入prompt

                    cur_new_input_embeds.append(cur_txt_prompt)
                    cur_new_labels.append(cur_txt_prompt_lbl)

                    cur_new_input_embeds.append(cur_input_embeds_no_im[i][1:])
                    cur_new_labels.append(cur_labels_noim[i][1:])
                else:
                    # 1. 放入当前的文本片段 (原始文本)
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])

                # 2. 如果还有图片要处理
                if i < num_images:
                    if i == 0:
                        cur_new_input_embeds.append(cur_img_prompt)
                        cur_new_labels.append(cur_img_prompt_lbl)

                    # 放入真实的图片特征
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        # --- 核心循环修改结束 ---

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.model.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.model.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            image_features,
        )

    @property
    def active_peft_config(self):
        if isinstance(self.peft_config, DrapeConfig):
            return self.peft_config
        return self.peft_config[self.active_adapter]

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the Drape generator weights, config, and continual learning state.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)

        # 1. Save Config
        self.active_peft_config.save_pretrained(save_directory)

        # 2. Save Current Generator Weights
        output_state_dict = self.drape_generator.state_dict()
        torch.save(output_state_dict, os.path.join(save_directory, "adapter_model.bin"))

        # 3. Save mm_projector weights
        non_peft_trainables = {}
        try:
            base_state = self.model.state_dict()
        except Exception:
            base_state = self.model.get_model().state_dict()
        for k, v in base_state.items():
            if "mm_projector" in k:
                non_peft_trainables[k] = v.detach().cpu()
        if len(non_peft_trainables) > 0:
            torch.save(
                non_peft_trainables,
                os.path.join(save_directory, "non_peft_trainables.bin"),
            )

        # 4. Save Continual Learning State
        if self.enable_cl and kwargs.get("save_continual_state", True):
            self._save_continual_learning_state(save_directory)

    def _save_continual_learning_state(self, save_directory):
        """Save all continual learning artifacts."""
        # 4a. Save router state
        if self.clip_router is not None:
            router_dir = os.path.join(save_directory, "router")
            self.clip_router.save_state(router_dir)
            print(f"[CL] Saved router state to {router_dir}")

        # 4b. Save all task generators (previous + current)
        generators_dir = os.path.join(save_directory, "task_generators")
        os.makedirs(generators_dir, exist_ok=True)

        all_task_names = list(self.previous_generators.keys())
        if self.current_task_name:
            all_task_names.append(self.current_task_name)

        for task_name in self.previous_generators:
            gen = self.previous_generators[task_name]
            torch.save(
                gen.state_dict(),
                os.path.join(generators_dir, f"{task_name}.bin"),
            )

        # Save current task generator
        if self.current_task_name:
            torch.save(
                self.drape_generator.state_dict(),
                os.path.join(generators_dir, f"{self.current_task_name}.bin"),
            )

        # Save generator config
        with open(os.path.join(generators_dir, "generators_config.json"), "w") as f:
            json.dump({"task_names": all_task_names}, f)
        print(f"[CL] Saved {len(all_task_names)} task generators to {generators_dir}")

        # 4c. Save null-space projection
        if self.apply_null_space_projection and (self.null_space_projections or self.null_space_stats):
            torch.save(
                self._pack_null_space_state(),
                os.path.join(save_directory, "null_space_projection.pt"),
            )
            print(f"[CL] Saved null-space projection")

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", **kwargs):
        """
        Loads the Drape model from a pretrained checkpoint.
        Supports loading continual learning state (router, previous generators, null-space).
        """
        # Load Config
        config = DrapeConfig.from_pretrained(model_id)

        # Instantiate Model
        drape_model = cls(model, config, adapter_name)

        # Load Weights
        filename = os.path.join(model_id, "adapter_model.bin")
        if os.path.exists(filename):
            try:
                # drape_model.drape_generator.load_state_dict(
                #     torch.load(filename, map_location="cpu"), strict=True
                # )
                state_dict = torch.load(filename, map_location="cpu")
                model_keys = set(drape_model.drape_generator.state_dict().keys())
                filtered = {k: v for k, v in state_dict.items() if k in model_keys}
                skipped = set(state_dict.keys()) - model_keys
                if skipped:
                    print(f"[CL] Skipped loading keys (removed from model): {skipped}")
                drape_model.drape_generator.load_state_dict(filtered, strict=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Drape generator weights: {filename}"
                )
        else:
            raise FileNotFoundError(f"Could not find adapter weights at {filename}")

        # Load Continual Learning state if available
        if config.enable_continual_learning:
            device = next(model.parameters()).device
            # Router and previous generators are loaded in __init__ via _load_previous_task_state
            # But we should also load from the current checkpoint's own CL state
            router_dir = os.path.join(model_id, "router")
            if os.path.exists(router_dir) and drape_model.clip_router is not None:
                drape_model.clip_router.load_state((router_dir), device=device)

            generators_dir = os.path.join(model_id, "task_generators")
            if os.path.exists(generators_dir):
                gen_config_path = os.path.join(generators_dir, "generators_config.json")
                if os.path.exists(gen_config_path):
                    with open(gen_config_path, "r") as f:
                        gen_cfg = json.load(f)
                    for task_name in gen_cfg.get("task_names", []):
                        if task_name == config.current_task_name:
                            continue  # Current task uses drape_generator
                        task_weight_path = os.path.join(generators_dir, f"{task_name}.bin")
                        if os.path.exists(task_weight_path):
                            prev_gen = DrapeGenerator(config)
                            prev_gen.load_state_dict(
                                torch.load(task_weight_path, map_location="cpu"),strict=False
                            )
                            # CHANGED(fp32-generator): keep checkpoint historical generators in fp32.
                            prev_gen = prev_gen.to(device=device, dtype=torch.float32)
                            for param in prev_gen.parameters():
                                param.requires_grad = False
                            drape_model.previous_generators[task_name] = prev_gen

        return drape_model

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class DrapePeftModel(PeftModel):
    """DRAPE adapter implementing Prism's existing PEFT model contract.

    CLModel expects six values from multimodal preparation. The tuner owns
    prompt/image expansion, while this adapter exposes that standard interface.
    Loading/saving uses load_adapter/save_pretrained; lifecycle state is handled
    separately by DrapeIntegration.load_extra_state/save_extra_state.
    """

    def routing_texts(self, input_ids, attention_mask=None, labels=None):
        if input_ids is None:
            raise ValueError("DRAPE requires input_ids to construct instruction prompts")
        tokenizer = getattr(self.base_model.model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("DRAPE requires Prism's LLaVA tokenizer")
        valid = input_ids >= 0
        if attention_mask is not None:
            valid &= attention_mask.bool()
        if labels is not None:
            valid &= labels == IGNORE_INDEX
        rows = [row[mask].detach().cpu().tolist() for row, mask in zip(input_ids, valid)]
        return tokenizer.batch_decode(rows, skip_special_tokens=True)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,
    ):
        prepared = self.base_model.prepare_prompt_inputs(
            input_ids=input_ids, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=past_key_values,
            labels=labels, images=images,
            route_texts=self.routing_texts(input_ids, attention_mask, labels),
        )
        return (
            None, None, prepared["attention_mask"], past_key_values,
            prepared["inputs_embeds"], prepared["labels"],
        )

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        kwargs.pop("cl_raw_example", None)
        kwargs.pop("image_sizes", None)
        if inputs_embeds is not None:
            # Image expansion and prompt construction already ran through
            # CLModel.prepare_inputs_labels_for_multimodal above.
            return self.base_model.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                labels=labels, **kwargs,
            )
        return self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            route_texts=self.routing_texts(input_ids, attention_mask, labels), **kwargs,
        )

    def generate(self, *args, **kwargs):
        if args:
            if len(args) != 1 or "input_ids" in kwargs or "inputs" in kwargs:
                raise ValueError("Pass one input_ids argument to DRAPE.generate")
            kwargs["input_ids"] = args[0]
        input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
        texts = self.routing_texts(input_ids, kwargs.get("attention_mask"))
        return self.base_model.generate(texts=texts, **kwargs)

    def load_adapter(self, model_id, adapter_name="default", **kwargs):
        if adapter_name != self.active_adapter:
            raise ValueError("DRAPE stores task generators within one default adapter")
        directory = os.path.abspath(model_id)
        for name in ("adapter_config.json", "adapter_model.bin", "non_peft_trainables.bin"):
            if not os.path.isfile(os.path.join(directory, name)):
                raise FileNotFoundError(f"Incomplete DRAPE checkpoint: {directory}/{name}")
        with open(os.path.join(directory, "adapter_config.json")) as stream:
            saved = json.load(stream)
        if saved.get("peft_type") != "DRAPETUNING":
            raise ValueError(f"Not a DRAPE checkpoint: {directory}")
        raw = self.base_model
        hidden = int(raw.model.config.hidden_size)
        if any(int(saved[name]) != hidden for name in ("token_dim", "text_input_dim", "vision_input_dim")):
            raise ValueError("DRAPE checkpoint hidden size does not match the LLaVA backbone")
        incremental = bool(getattr(self._drape_method_config, "previous_task_model_path", None))
        if not incremental:
            # Inference reconstructs the saved generator architecture through
            # the adapter interface, without a method-specific core argument.
            from dataclasses import fields
            names = {field.name for field in fields(DrapeConfig)}
            config = raw.active_peft_config
            for name, value in saved.items():
                if name in names and name not in ("clip_model_name", "previous_task_model_dir", "task_type"):
                    setattr(config, name, value)
            config.__post_init__()
            raw.drape_generator = DrapeGenerator(config).to(
                device=next(raw.model.parameters()).device, dtype=torch.float32,
            )
            raw.apply_null_space_projection = config.apply_null_space_projection
        raw.drape_generator.load_state_dict(
            torch.load(os.path.join(directory, "adapter_model.bin"), map_location="cpu"), strict=True,
        )
        projector = torch.load(os.path.join(directory, "non_peft_trainables.bin"), map_location="cpu")
        expected = {name for name in raw.model.state_dict() if "mm_projector" in name}
        if set(projector) != expected:
            raise ValueError("DRAPE checkpoint projector keys do not match the LLaVA model")
        raw.model.load_state_dict(projector, strict=False)
        self._drape_loaded_adapter = directory
        self._drape_saved_config = saved
        return self

    def save_pretrained(self, save_directory, **kwargs):
        # Prism saves router/history/NSP through save_extra_state after this.
        self.base_model.save_pretrained(save_directory, save_continual_state=False)
        self.config.save_pretrained(save_directory)
