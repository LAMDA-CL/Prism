# -*- encoding: utf-8 -*-
r"""CRAM LoRA: variable-rank expert slots plus a residual buf."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ...import_utils import is_bnb_4bit_available, is_bnb_available
from ...utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from ..standard.lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

if is_bnb_available():
    import bitsandbytes as bnb

_CRAM_ROUTE_N: int = 1
_CRAM_ROUTE_A_BUF: List[bool] = [False]
_CRAM_ROUTE_B_BUF: List[bool] = [False]
_CRAM_ROUTE_KA: List[int] = [0]
_CRAM_ROUTE_KB: List[int] = [0]
_CRAM_ROUTE_WA: List[float] = [1.0]
_CRAM_ROUTE_WB: List[float] = [0.0]
_CRAM_ROUTE_WA_TENSOR: List[Optional[torch.Tensor]] = [None]
_CRAM_ROUTE_WB_TENSOR: List[Optional[torch.Tensor]] = [None]

_CRAM_ROUTE_INFER_FULL: List[bool] = [False]
_CRAM_ROUTE_INFER_PI: List[Optional[torch.Tensor]] = [None]

def cram_set_route_num_layers(n: int) -> None:
    global _CRAM_ROUTE_N, _CRAM_ROUTE_KA, _CRAM_ROUTE_KB, _CRAM_ROUTE_A_BUF, _CRAM_ROUTE_B_BUF
    global _CRAM_ROUTE_WA, _CRAM_ROUTE_WB, _CRAM_ROUTE_WA_TENSOR, _CRAM_ROUTE_WB_TENSOR
    global _CRAM_ROUTE_INFER_FULL, _CRAM_ROUTE_INFER_PI
    n = max(1, int(n))
    if n == _CRAM_ROUTE_N and len(_CRAM_ROUTE_KA) == n:
        return
    _CRAM_ROUTE_N = n
    _CRAM_ROUTE_KA = [0] * n
    _CRAM_ROUTE_KB = [0] * n
    _CRAM_ROUTE_A_BUF = [False] * n
    _CRAM_ROUTE_B_BUF = [False] * n
    _CRAM_ROUTE_WA = [1.0] * n
    _CRAM_ROUTE_WB = [0.0] * n
    _CRAM_ROUTE_WA_TENSOR = [None] * n
    _CRAM_ROUTE_WB_TENSOR = [None] * n
    _CRAM_ROUTE_INFER_FULL = [False] * n
    _CRAM_ROUTE_INFER_PI = [None] * n

def clear_cram_infer_group_softmax_routing() -> None:

    global _CRAM_ROUTE_INFER_FULL, _CRAM_ROUTE_INFER_PI
    for li in range(_CRAM_ROUTE_N):
        _CRAM_ROUTE_INFER_FULL[li] = False
        _CRAM_ROUTE_INFER_PI[li] = None

def set_cram_infer_group_softmax_layer(layer_id: Any, pi_per_slot: Optional[torch.Tensor]) -> None:

    global _CRAM_ROUTE_INFER_FULL, _CRAM_ROUTE_INFER_PI
    li = _cram_route_li(layer_id)
    if pi_per_slot is None:
        _CRAM_ROUTE_INFER_FULL[li] = False
        _CRAM_ROUTE_INFER_PI[li] = None
        return
    _CRAM_ROUTE_INFER_FULL[li] = True
    if bool(getattr(pi_per_slot, "requires_grad", False)):
        _CRAM_ROUTE_INFER_PI[li] = pi_per_slot
    else:
        _CRAM_ROUTE_INFER_PI[li] = pi_per_slot.detach().float().cpu().clone()
    _CRAM_ROUTE_KA[li] = 0
    _CRAM_ROUTE_KB[li] = 0
    _CRAM_ROUTE_A_BUF[li] = False
    _CRAM_ROUTE_B_BUF[li] = False
    _CRAM_ROUTE_WA[li] = 1.0
    _CRAM_ROUTE_WB[li] = 0.0
    _CRAM_ROUTE_WA_TENSOR[li] = None
    _CRAM_ROUTE_WB_TENSOR[li] = None

def _cram_route_li(layer_id: Any) -> int:
    li = int(layer_id)
    return max(0, min(li, _CRAM_ROUTE_N - 1))

def set_cram_budget_route_layer(
    layer_id: Any,
    ka: int,
    kb: int,
    unit_a_is_buf: bool,
    unit_b_is_buf: bool,
    wa: float,
    wb: float,
    wa_tensor: Optional[torch.Tensor] = None,
    wb_tensor: Optional[torch.Tensor] = None,
) -> None:
    li = _cram_route_li(layer_id)
    _CRAM_ROUTE_INFER_FULL[li] = False
    _CRAM_ROUTE_INFER_PI[li] = None
    _CRAM_ROUTE_KA[li] = int(ka)
    _CRAM_ROUTE_KB[li] = int(kb)
    _CRAM_ROUTE_A_BUF[li] = bool(unit_a_is_buf)
    _CRAM_ROUTE_B_BUF[li] = bool(unit_b_is_buf)
    _CRAM_ROUTE_WA[li] = float(wa)
    _CRAM_ROUTE_WB[li] = float(wb)
    _CRAM_ROUTE_WA_TENSOR[li] = wa_tensor
    _CRAM_ROUTE_WB_TENSOR[li] = wb_tensor

def set_cram_budget_route_all_layers(
    ka: int, kb: int, ua_buf: bool, ub_buf: bool, wa: float, wb: float
) -> None:
    for li in range(_CRAM_ROUTE_N):
        set_cram_budget_route_layer(li, ka, kb, ua_buf, ub_buf, wa, wb, None, None)

def apply_expert_budget_ranks_to_module(
    module: "CramBudgetLoraLinear",
    ranks: List[int],
    *,
    r_buf_train: Optional[int] = None,
) -> None:

    Rmax = int(module.rank_max)
    cap = int(module.lora_cram_buf_A.shape[0])
    rb = int(r_buf_train) if r_buf_train is not None else int(module.cram_rank_total)
    rb = max(1, min(rb, cap))
    with torch.no_grad():
        for s in range(min(len(ranks), module.max_slots)):
            rv = int(max(0, min(ranks[s], Rmax)))
            module.lora_cram_expert_r[s] = int(rv)
            module.lora_cram_expert_mask[s] = rv > 0
        module.r_buf = rb
        module.num_committed = int(module.lora_cram_expert_mask.sum().item())
    module._recompute_buf_scaling()

def sync_cram_r_buf_fixed(module: "CramBudgetLoraLinear", r_buf_train: int) -> None:

    cap = int(module.lora_cram_buf_A.shape[0])
    module.r_buf = max(1, min(int(r_buf_train), cap))
    module._recompute_buf_scaling()

def sync_cram_r_buf_from_budget_charges(module: "CramBudgetLoraLinear", charges: List[int]) -> None:

    sync_cram_r_buf_fixed(module, int(module.cram_rank_total))

class CramLoraAdapterWeights(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_buf: int,
        max_slots: int,
        rank_max: int,
        *,
        init_lora_weights: bool = True,
    ) -> None:
        super().__init__()
        self.register_parameter("buf_A", nn.Parameter(torch.zeros(r_buf, in_features)))
        self.register_parameter("buf_B", nn.Parameter(torch.zeros(out_features, r_buf)))
        self.register_parameter(
            "expert_A",
            nn.Parameter(torch.zeros(max_slots, rank_max, in_features)),
        )
        self.register_parameter(
            "expert_B",
            nn.Parameter(torch.zeros(max_slots, out_features, rank_max)),
        )
        self.register_buffer("expert_mask", torch.zeros(max_slots, dtype=torch.bool))
        self.register_buffer("expert_r", torch.zeros(max_slots, dtype=torch.long))

        if init_lora_weights:
            nn.init.kaiming_uniform_(self.buf_A, a=math.sqrt(5))
            nn.init.zeros_(self.buf_B)
            nn.init.zeros_(self.expert_A)
            nn.init.zeros_(self.expert_B)

@dataclass
class CramBudgetLoraConfig(LoraConfig):
    cram_rank_total: int = field(default=48)
    cram_expert_rank_max: int = field(default=9)
    cram_max_expert_slots: int = field(default=10)
    expert_num: int = field(default=10)
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CRAM
        if self.r <= 0:
            self.r = int(self.cram_rank_total)
        if int(self.expert_num) != int(self.cram_max_expert_slots):
            self.expert_num = int(self.cram_max_expert_slots)

class CramBudgetLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _prepare_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("CramBudgetLoraModel supports only 1 adapter with bias.")

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, layer = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer):
                raise ValueError("CRAM expects plain Linear targets.")
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, self.model.training, layer)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {lora_config.target_modules} not found in base model.")

    def _create_new_module(self, lora_config, adapter_name, target, training, layer):
        bias = hasattr(target, "bias") and target.bias is not None

        kwargs = {
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "cram_rank_total": int(lora_config.cram_rank_total),
            "cram_expert_rank_max": int(getattr(lora_config, "cram_expert_rank_max", 9)),
            "cram_max_expert_slots": int(lora_config.cram_max_expert_slots),
            "layer_id": int(layer),
        }
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            raise ValueError("CRAM budget LoRA + 8bit is not supported.")
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            raise ValueError("CRAM budget LoRA + 4bit is not supported.")
        elif isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn("fan_in_fan_out set to True for Linear layer; setting to False.")
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(f"Target module {target} not supported.")
        return CramBudgetLoraLinear(
            adapter_name,
            in_features,
            out_features,
            bias=bias,
            train_signal=training,
            **kwargs,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

class CramBudgetLoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, training: bool, layer_id: int):
        super().__init__(in_features, out_features)
        self.layer_id = layer_id

class CramBudgetLoraLinear(nn.Linear, CramBudgetLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        train_signal: bool,
        layer_id: int = 0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        cram_rank_total: int = 48,
        cram_expert_rank_max: int = 9,
        cram_max_expert_slots: int = 10,
        **kwargs,
    ):
        self.is_target_conv_1d_layer = bool(kwargs.pop("is_target_conv_1d_layer", False))
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        CramBudgetLoraLayer.__init__(self, in_features, out_features, train_signal, layer_id)
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)

        self.cram_rank_total = int(cram_rank_total)
        self.rank_max = int(max(1, cram_expert_rank_max))
        self.max_slots = int(cram_max_expert_slots)
        self.r_buf = int(self.cram_rank_total)
        self.num_committed = 0

        self.lora_cram = nn.ModuleDict(
            {
                adapter_name: CramLoraAdapterWeights(
                    in_features,
                    out_features,
                    int(self.cram_rank_total),
                    self.max_slots,
                    self.rank_max,
                    init_lora_weights=init_lora_weights,
                )
            }
        )
        self.lora_dropout_layer = nn.Dropout(p=0.05)
        self.active_adapter = adapter_name
        self._recompute_buf_scaling()

        self.merged = False
        self.disable_adapters = False

    @property
    def lora_cram_buf_A(self) -> nn.Parameter:
        return self.lora_cram[self.active_adapter].buf_A

    @property
    def lora_cram_buf_B(self) -> nn.Parameter:
        return self.lora_cram[self.active_adapter].buf_B

    @property
    def lora_cram_expert_A(self) -> nn.Parameter:
        return self.lora_cram[self.active_adapter].expert_A

    @property
    def lora_cram_expert_B(self) -> nn.Parameter:
        return self.lora_cram[self.active_adapter].expert_B

    @property
    def lora_cram_expert_mask(self) -> torch.Tensor:
        return self.lora_cram[self.active_adapter].expert_mask

    @property
    def lora_cram_expert_r(self) -> torch.Tensor:
        return self.lora_cram[self.active_adapter].expert_r

    def _recompute_buf_scaling(self) -> None:
        rb = max(1, int(self.r_buf))
        self.scaling_buf = float(2 * rb) / float(rb)

    @staticmethod
    def _scaling_expert_for_rank(r_e: int) -> float:
        r_e = max(1, int(r_e))
        return float(2 * r_e) / float(r_e)

    def set_r_buf(self, r_new: int) -> None:
        r_new = int(max(1, min(r_new, int(self.lora_cram_buf_A.shape[0]))))
        self.r_buf = r_new
        self._recompute_buf_scaling()

    def _expert_branch(self, x_d: torch.Tensor, slot: int) -> torch.Tensor:
        out_shape = x_d.shape[:-1] + (self.out_features,)
        if not bool(self.lora_cram_expert_mask[slot].item()):
            return x_d.new_zeros(out_shape)
        r_e = int(self.lora_cram_expert_r[slot].item())
        if r_e <= 0:
            return x_d.new_zeros(out_shape)
        A = self.lora_cram_expert_A[slot, :r_e]
        B = self.lora_cram_expert_B[slot, :, :r_e]
        xa = x_d.to(dtype=A.dtype)
        h = F.linear(xa, A)
        out = F.linear(h, B)
        return out * self._scaling_expert_for_rank(r_e)

    def expert_delta(self, x: torch.Tensor, slot: int) -> torch.Tensor:

        x_d = self.lora_dropout_layer(x).to(self.lora_cram_expert_A.dtype)
        return self._expert_branch(x_d, int(slot))

    def _buf_branch(self, x_d: torch.Tensor) -> torch.Tensor:
        r = int(self.r_buf)
        A = self.lora_cram_buf_A[:r]
        B = self.lora_cram_buf_B[:, :r]
        xa = x_d.to(dtype=A.dtype)
        h = F.linear(xa, A)
        return F.linear(h, B) * self.scaling_buf

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.disable_adapters or self.merged:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias).to(previous_dtype)

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        li = _cram_route_li(self.layer_id)
        if li < _CRAM_ROUTE_N and _CRAM_ROUTE_INFER_FULL[li] and _CRAM_ROUTE_INFER_PI[li] is not None:
            pi = _CRAM_ROUTE_INFER_PI[li].to(device=x.device, dtype=self.lora_cram_buf_A.dtype)
            x_d = self.lora_dropout_layer(x).to(self.lora_cram_buf_A.dtype)
            ms = min(int(pi.numel()), int(self.max_slots))
            acc = None
            for s in range(ms):
                w = pi[s]
                if not torch.is_tensor(w):
                    w = x_d.new_tensor(float(w))
                if float(w.detach().abs().item()) < 1e-12:
                    continue
                b = self._expert_branch(x_d, s)
                term = b * w
                acc = term if acc is None else acc + term
            if acc is None:
                acc = x_d.new_zeros(x_d.shape[:-1] + (self.out_features,))
            result = result + acc.to(dtype=result.dtype)
            return result.to(previous_dtype)

        ka = max(0, min(int(_CRAM_ROUTE_KA[li]), self.max_slots - 1))
        kb = max(0, min(int(_CRAM_ROUTE_KB[li]), self.max_slots - 1))
        ua_buf = bool(_CRAM_ROUTE_A_BUF[li])
        ub_buf = bool(_CRAM_ROUTE_B_BUF[li])
        waf = float(_CRAM_ROUTE_WA[li])
        wbf = float(_CRAM_ROUTE_WB[li])

        x_d = self.lora_dropout_layer(x).to(self.lora_cram_buf_A.dtype)

        def pure_unit(is_buf: bool, slot: int) -> torch.Tensor:
            if is_buf:
                return self._buf_branch(x_d)
            return self._expert_branch(x_d, slot)

        wa_t = _CRAM_ROUTE_WA_TENSOR[li]
        wb_t = _CRAM_ROUTE_WB_TENSOR[li]
        same_unit = ua_buf == ub_buf and (ua_buf or ka == kb)
        if wa_t is not None and wb_t is not None and not same_unit:
            b0 = pure_unit(ua_buf, ka)
            b1 = pure_unit(ub_buf, kb)
            t0 = wa_t.to(device=b0.device, dtype=b0.dtype)
            t1 = wb_t.to(device=b1.device, dtype=b1.dtype)
            lora_m = t0 * b0 + t1 * b1
        else:
            if same_unit:
                lora_m = pure_unit(ua_buf, ka)
            else:
                lora_m = waf * pure_unit(ua_buf, ka) + wbf * pure_unit(ub_buf, kb)

        result = result + lora_m.to(dtype=result.dtype)
        return result.to(previous_dtype)

def cram_hist_expert_output_basis(
    module: CramBudgetLoraLinear,
    slot: int,
    *,
    device: torch.device,
    hist_expert_slots: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:

    cols: List[torch.Tensor] = []
    if not hist_expert_slots:
        return None
    seen: set = set()
    for s in hist_expert_slots:
        s = int(s)
        if s in seen or s == int(slot) or s < 0 or s >= int(module.max_slots):
            continue
        seen.add(s)
        if not bool(module.lora_cram_expert_mask[s].item()):
            continue
        re = int(module.lora_cram_expert_r[s].item())
        if re <= 0:
            continue
        cols.append(module.lora_cram_expert_B[s, :, :re].detach().float())
    if not cols:
        return None
    M = torch.cat(cols, dim=1).to(device=device, dtype=torch.float32)
    if M.numel() == 0 or M.shape[1] == 0:
        return None
    try:
        Q, _ = torch.linalg.qr(M, mode="reduced")
    except RuntimeError:
        Q, _ = torch.linalg.qr(M.cpu(), mode="reduced")
        Q = Q.to(device=device, dtype=torch.float32)
    return Q.to(dtype=torch.float32)

def _cram_hist_expert_a_row_basis(
    module: CramBudgetLoraLinear,
    slot: int,
    *,
    device: torch.device,
    hist_expert_slots: Optional[Sequence[int]] = None,
) -> Optional[torch.Tensor]:

    rows: List[torch.Tensor] = []
    if not hist_expert_slots:
        return None
    seen: set = set()
    for s in hist_expert_slots:
        s = int(s)
        if s in seen or s == int(slot) or s < 0 or s >= int(module.max_slots):
            continue
        seen.add(s)
        if not bool(module.lora_cram_expert_mask[s].item()):
            continue
        re = int(module.lora_cram_expert_r[s].item())
        if re <= 0:
            continue
        rows.append(module.lora_cram_expert_A[s, :re].detach().float().reshape(re, -1))
    if not rows:
        return None
    M = torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)
    if M.numel() == 0 or M.shape[0] == 0:
        return None

    try:
        Q, _ = torch.linalg.qr(M.T, mode="reduced")
    except RuntimeError:
        Q, _ = torch.linalg.qr(M.T.cpu(), mode="reduced")
        Q = Q.to(device=device, dtype=torch.float32)
    return Q.to(dtype=torch.float32)

def _cram_svd_scaled_BA(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    *,
    need_U: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """SVD of ``scale * B @ A`` without forming the full ``[out, in]`` matrix.

    ``A`` is ``[r, in]``, ``B`` is ``[out, r]``. Rank is at most ``r``, so
    ``QR(B)`` + ``SVD(R @ A)`` matches ``svd(B @ A)`` and is far cheaper
    (4096×4096 SVD was taking ~15 minutes for 128 attn layers).
    """

    def _run(Aa: torch.Tensor, Bb: torch.Tensor):
        Qb, Rb = torch.linalg.qr(Bb, mode="reduced")
        mid = (float(scale) * Rb) @ Aa
        if need_U:
            Us, S, Vh = torch.linalg.svd(mid, full_matrices=False)
            return Qb @ Us, S, Vh
        _, S, Vh = torch.linalg.svd(mid, full_matrices=False)
        return None, S, Vh

    try:
        return _run(A, B)
    except RuntimeError:
        U, S, Vh = _run(A.cpu(), B.cpu())
        if U is not None:
            U = U.to(device=A.device, dtype=A.dtype)
        S = S.to(device=A.device, dtype=A.dtype)
        Vh = Vh.to(device=A.device, dtype=A.dtype)
        return U, S, Vh

def consolidate_buf_into_expert_slot(
    module: CramBudgetLoraLinear,
    slot: int,
    tau_alloc: float,
    rank_max: int,
    *,
    tau_novel: float = 0.0,
    hist_expert_slots: Optional[Sequence[int]] = None,
    rank_min: int = 6,
) -> int:

    if slot < 0 or slot >= module.max_slots:
        raise ValueError(f"bad slot {slot}")
    r = int(module.r_buf)
    if r <= 0:
        return 0
    A = module.lora_cram_buf_A[:r].detach().float()
    B = module.lora_cram_buf_B[:, :r].detach().float()
    scale = float(module.scaling_buf)
    U, S, Vh = _cram_svd_scaled_BA(A, B, scale, need_U=True)
    assert U is not None

    n = int(S.numel())
    if n == 0:
        return 0
    thr = float(tau_alloc)
    spectral_mask = S > thr
    cand = torch.where(spectral_mask)[0]
    if int(cand.numel()) == 0:
        cand = torch.tensor([0], device=S.device, dtype=torch.long)

    order = torch.argsort(S[cand], descending=True)
    cand_sorted = cand[order]
    cap = int(min(int(rank_max), n, int(cand_sorted.numel())))
    cand_sorted = cand_sorted[:cap]

    Q = None
    if float(tau_novel) > 0.0:
        Q = _cram_hist_expert_a_row_basis(
            module, slot, device=Vh.device, hist_expert_slots=hist_expert_slots
        )

    sel_list: List[int] = []
    if Q is None or float(tau_novel) <= 0.0:
        sel_list = [int(cand_sorted[i].item()) for i in range(int(cand_sorted.numel()))]
    else:
        Vc = Vh[cand_sorted.long(), :].float()
        Qf = Q.float() if Q is not None else None
        proj = (Vc @ Qf) @ Qf.T
        perp = Vc - proj
        vnorm = Vc.norm(dim=1).clamp(min=1e-12)
        alpha = perp.norm(dim=1) / vnorm
        for j in range(int(cand_sorted.numel())):
            if float(alpha[j].item()) > float(tau_novel):
                sel_list.append(int(cand_sorted[j].item()))
        if not sel_list:
            sel_list = [int(cand_sorted[0].item())]

    rmin = int(rank_min)
    if rmin > 0:
        target = min(max(rmin, len(sel_list)), int(rank_max), n)
        pool = torch.argsort(S, descending=True)[: min(int(rank_max), n)]
        seen: set = set()
        new_sel: List[int] = []
        for i in sorted(sel_list, key=lambda k: float(S[k].item()), reverse=True):
            if i not in seen:
                seen.add(i)
                new_sel.append(i)
        for j in range(int(pool.numel())):
            if len(new_sel) >= target:
                break
            p = int(pool[j].item())
            if p not in seen:
                seen.add(p)
                new_sel.append(p)
        sel_list = new_sel

    idx_t = torch.tensor(sel_list, device=S.device, dtype=torch.long)
    Uu = U[:, idx_t]
    Ss = S[idx_t]
    Vraw = Vh[idx_t, :]

    if Q is not None and float(tau_novel) > 0.0:
        Qf = Q.float()
        Vf = Vraw.float()
        proj2 = (Vf @ Qf) @ Qf.T
        Vorth = Vf - proj2
        row_norm = Vorth.norm(dim=1, keepdim=True).clamp(min=1e-12)
        too_flat = (row_norm.squeeze(-1) < 1e-6 * Vf.norm(dim=1).clamp(min=1e-12))
        if bool(too_flat.any().item()):
            Vorth = torch.where(too_flat.unsqueeze(-1), Vf, Vorth)
            row_norm = Vorth.norm(dim=1, keepdim=True).clamp(min=1e-12)
        V_use = Vorth / row_norm
    else:
        V_use = Vraw

    sqrt_s = torch.sqrt(torch.clamp(Ss, min=1e-12))
    B_new = Uu * sqrt_s.unsqueeze(0)
    A_new = sqrt_s.unsqueeze(1) * V_use

    dt = module.lora_cram_expert_A.dtype
    r_t = int(A_new.shape[0])
    with torch.no_grad():
        module.lora_cram_expert_A[slot].zero_()
        module.lora_cram_expert_B[slot].zero_()
        module.lora_cram_expert_A[slot, :r_t].copy_(A_new.to(dt))
        module.lora_cram_expert_B[slot, :, :r_t].copy_(B_new.to(dt))
        module.lora_cram_expert_mask[slot] = True
        module.lora_cram_expert_r[slot] = int(r_t)

    with torch.no_grad():
        module.lora_cram_buf_A.zero_()
        module.lora_cram_buf_B.zero_()

    return int(r_t)

def sync_num_committed_from_mask(module: CramBudgetLoraLinear) -> None:
    module.num_committed = int(module.lora_cram_expert_mask.sum().item())

def repair_cram_expert_mask_and_r_from_weights(
    module: CramBudgetLoraLinear,
    *,
    abs_eps: float = 1e-5,
    rel_eps: float = 1e-4,
) -> None:

    w = module.lora_cram[module.active_adapter]
    rm = int(module.rank_max)
    with torch.no_grad():
        for s in range(int(module.max_slots)):
            A = w.expert_A[s].detach().float()
            B = w.expert_B[s].detach().float()
            ra = A.abs().sum(-1)
            rb = B.abs().sum(0)
            mx = max(float(ra.max().item()), float(rb.max().item()), 1e-12)
            thr = max(abs_eps, rel_eps * mx)
            nr = int((ra > thr).sum().item())
            nc = int((rb > thr).sum().item())
            if nr <= 0 and nc <= 0:
                w.expert_mask[s] = False
                w.expert_r[s] = 0
                continue
            if nr > 0 and nc > 0:
                r_eff = min(nr, nc)
            else:
                r_eff = max(nr, nc)
            r_eff = max(1, min(int(r_eff), rm))
            w.expert_mask[s] = True
            w.expert_r[s] = int(r_eff)
    sync_num_committed_from_mask(module)

def repair_all_cram_expert_mask_and_r_from_weights(root: nn.Module) -> int:

    n = 0
    for m in root.modules():
        if isinstance(m, CramBudgetLoraLinear):
            repair_cram_expert_mask_and_r_from_weights(m)
            n += 1
    return n
