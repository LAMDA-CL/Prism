r"""CRAM: level-1 text prototypes and level-2 visual routing."""
from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from config.backbone.constants import IMAGE_TOKEN_INDEX
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from method.base.routing_utils import (
    extract_routing_image_features,
    resolve_clip_tokenizer,
    resolve_mllm_vision_tower,
    resolve_text_tower,
)
from method.factory import CLMethodFactory

_LOG = logging.getLogger(__name__)
_PEFT_EXT_REGISTERED = False
_CRAM_DS_WARN_NO_REFRESH = False

def _cram_call_deepspeed_refresh_fp32_from_lp(engine: Optional[Any]) -> None:

    global _CRAM_DS_WARN_NO_REFRESH
    if engine is None:
        return
    opt = getattr(engine, "optimizer", None)
    if opt is None:
        return
    if not hasattr(opt, "refresh_fp32_params"):
        if not _CRAM_DS_WARN_NO_REFRESH:
            _CRAM_DS_WARN_NO_REFRESH = True
            _LOG.warning(
                "CRAM centroid sync: optimizer=%s has no refresh_fp32_params; skipping.",
                type(opt).__name__,
            )
        return
    try:
        opt.refresh_fp32_params()
    except Exception as e:
        _LOG.warning("CRAM centroid sync: refresh_fp32_params failed: %s", e)

maybe_deepspeed_refresh_fp32_params_from_lp = _cram_call_deepspeed_refresh_fp32_from_lp

def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return
    from PEFT.peft_model import PeftModelForCausalLMLORAMOE
    from PEFT.tuners.custom.cram_lora import CramBudgetLoraConfig, CramBudgetLoraModel

    register_peft_extension(
        peft_type="MOE_LORA_CRAM",
        config_cls=CramBudgetLoraConfig,
        tuner_model_cls=CramBudgetLoraModel,
        task_type="CAUSAL_LM_CRAM",
        task_peft_model_cls=PeftModelForCausalLMLORAMOE,
    )
    _PEFT_EXT_REGISTERED = True

def _dist_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

def _dist_cram_training_multi_gpu(model: nn.Module) -> bool:

    if not model.training or not torch.is_grad_enabled():
        return False
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
    except Exception:
        return False

def _prep_image_feat_batches_dm(image_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    x = image_feat.float()
    while x.dim() > 2:
        x = x.mean(dim=1)
    if x.dim() != 2:
        raise RuntimeError(f"CRAM: image_feat expected [B, D], got {tuple(x.shape)}")
    b = int(x.shape[0])
    if b <= 0:
        d = int(x.shape[-1])
        return x.new_zeros((0, d)), x.new_zeros((d,)), 0
    return x, x.mean(dim=0), b

def _build_visual_token_mask_expanded(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    num_patches: int = 576,
) -> torch.Tensor:

    if input_ids is None:
        raise ValueError("input_ids required for visual token mask")
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    bsz = int(input_ids.shape[0])
    masks: List[torch.Tensor] = []
    for b in range(bsz):
        cur_ids = input_ids[b][attention_mask[b]]
        num_images = int((cur_ids == IMAGE_TOKEN_INDEX).sum().item())
        image_token_indices = (
            [-1]
            + torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [int(cur_ids.shape[0])]
        )
        parts: List[torch.Tensor] = []
        for i in range(len(image_token_indices) - 1):
            text_len = image_token_indices[i + 1] - image_token_indices[i] - 1
            if text_len > 0:
                parts.append(torch.zeros(text_len, dtype=torch.bool))
            if i < num_images:
                parts.append(torch.ones(num_patches, dtype=torch.bool))
        mask_b = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.bool)
        masks.append(mask_b)
    max_len = max(int(m.shape[0]) for m in masks) if masks else 0
    out = torch.zeros(bsz, max_len, dtype=torch.bool, device=device)
    for b, m in enumerate(masks):
        if m.numel() > 0:
            out[b, : int(m.shape[0])] = m.to(device)
    return out

def _resolve_visual_token_patch_count(model: Any) -> int:
    """Resolve the actual MLLM patch count instead of assuming the LLaVA 576-patch layout."""
    tower = resolve_mllm_vision_tower(model)
    for attr in ("num_patches", "num_image_patches", "image_size"):
        value = getattr(tower, attr, None) if tower is not None else None
        if isinstance(value, int) and value > 0:
            if attr == "image_size":
                patch_size = getattr(tower, "patch_size", None)
                if isinstance(patch_size, int) and patch_size > 0:
                    return max(1, (value // patch_size) ** 2)
            else:
                return int(value)
    return 576

@CLMethodFactory.register("cram")
class CramIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.max_slots: int = 10
        self.feature_dim: int = int(getattr(config, "clip_feature_dim", 768))
        _bm = str(getattr(config, "benchmark", "") or "").strip().lower()
        self.max_groups: int = 5 if _bm == "ucit" else 10 if _bm == "trigap" else max(1, self.task_num)
        self.theta: float = float(getattr(config, "cram_delta_threshold", 0.1))
        self._model_ref: Any = None
        self._prep_invoke: int = 0
        self._cram_optimizer_steps_done: int = 0
        self._centroid_param: Optional[nn.Parameter] = None
        self._buf_centroid_param: Optional[torch.Tensor] = None
        self._prep_task_id_for_grad: int = int(getattr(config, "cur_task", 0))

        self._cram_forward_train_hard: bool = False

        self.group_prototypes: List[torch.Tensor] = []
        self.group_counts: List[int] = []
        self.group_tasks: List[List[int]] = []
        self.group_experts: List[List[int]] = []
        self._next_expert_id: int = 0
        self._task_expert_lock: Dict[int, int] = {}
        self._task_group_lock: Dict[int, int] = {}

        self.centroid_n: torch.Tensor = torch.zeros(0, dtype=torch.long)
        self.expert_budget_charge: List[int] = [0] * int(self.max_slots)
        self._cram_last_centroid_reset_task: Optional[int] = None

        self._cram_buf_svd_done_for_task: Dict[int, bool] = {}
        self._cram_stable_train_expert_slot: Optional[int] = None

        self._cram_centroid_block_entire_grad: bool = False

        self._cram_lora_grad_hook_handles: List[Any] = []

        self._cram_ds_engine_ref: Optional[Any] = None

        self._cram_resolved_visual_warmup_steps: Optional[int] = None
        self._cram_cached_task_max_steps: int = 0

        self._cram_dec_visual_mask: Optional[torch.Tensor] = None
        self._cram_dec_image_feat: Optional[torch.Tensor] = None
        self._cram_dec_route_ctx: Optional[Dict[str, Any]] = None
        self._cram_dec_layer_inputs: Dict[int, torch.Tensor] = {}
        self._cram_dec_hook_handles: List[Any] = []

    def apply_config_hyperparameters(self, *, tag: str = "config") -> None:

        cfg = self.config
        self.task_num = int(getattr(cfg, "task_num", getattr(cfg, "expert_num", self.task_num)))
        self.max_slots = 10
        _bm = str(getattr(cfg, "benchmark", "") or "").strip().lower()
        self.max_groups = 5 if _bm == "ucit" else 10 if _bm == "trigap" else max(1, self.task_num)
        self.theta = float(getattr(cfg, "cram_delta_threshold", 0.1))
        self.feature_dim = int(getattr(cfg, "clip_feature_dim", 768))
        while len(self.expert_budget_charge) < int(self.max_slots):
            self.expert_budget_charge.append(0)
        if len(self.expert_budget_charge) > int(self.max_slots):
            self.expert_budget_charge = self.expert_budget_charge[: int(self.max_slots)]
        self._cram_resolved_visual_warmup_steps = None

    def _resolve_cram_task_max_optimizer_steps(self, trainer: Any = None) -> int:

        if trainer is not None:
            st = getattr(trainer, "state", None)
            if st is not None:
                ms = int(getattr(st, "max_steps", 0) or 0)
                if ms > 0:
                    return ms
            args = getattr(trainer, "args", None)
            if args is not None:
                ms = int(getattr(args, "max_steps", 0) or 0)
                if ms > 0:
                    return ms
                try:
                    dl = trainer.get_train_dataloader()
                    n = len(dl)
                    gas = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
                    epochs = float(getattr(args, "num_train_epochs", 1) or 1)
                    return max(1, int(n // gas * epochs))
                except Exception:
                    pass
        return int(getattr(self, "_cram_cached_task_max_steps", 0) or 0)

    def _refresh_cram_visual_warmup_steps(self, trainer: Any = None) -> int:

        ratio = getattr(self.config, "cram_visual_warmup_ratio", 0.05)
        try:
            rf = float(ratio) if ratio is not None else -1.0
        except (TypeError, ValueError):
            rf = -1.0
        if rf <= 0.0:
            self._cram_resolved_visual_warmup_steps = -1
            return -1

        max_steps = self._resolve_cram_task_max_optimizer_steps(trainer)
        self._cram_cached_task_max_steps = int(max_steps)
        if max_steps <= 0:
            self._cram_resolved_visual_warmup_steps = None
            return -1
        nw = max(1, int(round(rf * float(max_steps))))
        self._cram_resolved_visual_warmup_steps = nw
        return nw

    def _cram_visual_warmup_steps(self) -> int:

        cached = getattr(self, "_cram_resolved_visual_warmup_steps", None)
        if cached is not None:
            return int(cached)
        return self._refresh_cram_visual_warmup_steps(None)

    def _resolve_deepspeed_engine_for_centroid_sync(self) -> Optional[Any]:

        ref = getattr(self, "_cram_ds_engine_ref", None)
        if ref is not None:
            opt = getattr(ref, "optimizer", None)
            if opt is not None and hasattr(opt, "refresh_fp32_params"):
                return ref
            self._cram_ds_engine_ref = None
        for fr in inspect.stack()[2:36]:
            self_obj = fr.frame.f_locals.get("self")
            if self_obj is None:
                continue
            opt = getattr(self_obj, "optimizer", None)
            if opt is not None and hasattr(opt, "refresh_fp32_params"):
                self._cram_ds_engine_ref = self_obj
                return self_obj
        return None

    def _maybe_refresh_deepspeed_fp32_after_centroid_data_write(self) -> None:

        eng = self._resolve_deepspeed_engine_for_centroid_sync()
        if eng is None:
            return
        _cram_call_deepspeed_refresh_fp32_from_lp(eng)

    @staticmethod
    def _unwrap_training_model(model: nn.Module) -> nn.Module:

        m: Any = model
        for _ in range(16):
            nxt = getattr(m, "module", None)
            if nxt is None:
                break
            m = nxt
        return m

    def _task_id_for_expert_slot(self, slot: int) -> Optional[int]:
        for tasks, exps in zip(self.group_tasks, self.group_experts):
            for t, e in zip(tasks, exps):
                if int(e) == int(slot):
                    return int(t)
        return None

    def _group_index_for_task(self, task_id: int) -> Optional[int]:
        tid = int(task_id)
        if tid in self._task_group_lock:
            return int(self._task_group_lock[tid])
        for g, tasks in enumerate(self.group_tasks):
            if tid in tasks:
                return int(g)
        return None

    def _hist_expert_slots_same_semantic_pool(self, task_id: int, slot: int) -> List[int]:

        g = self._group_index_for_task(task_id)
        if g is None or g < 0 or g >= len(self.group_experts):
            return []
        s_tgt = int(slot)
        return [int(e) for e in self.group_experts[g] if int(e) != s_tgt]

    def _centroid_expert_row_frozen(self, slot: int, t_cur: int) -> bool:
        if not getattr(self, "_cram_forward_train_hard", False):
            return False
        tid = self._task_id_for_expert_slot(slot)
        if tid is None:
            return False
        return tid < int(t_cur)

    def _reset_buf_and_current_expert_centroids_for_task(self, model: nn.Module, e_slot: int) -> None:

        self._ensure_centroid_tables(model)
        if self._buf_sum is not None:
            self._buf_sum.zero_()
        if self._buf_count is not None:
            self._buf_count.zero_()
        if self._buf_centroid_param is not None:
            with torch.no_grad():
                self._buf_centroid_param.data.zero_()
        if self._centroid_sum is not None and 0 <= int(e_slot) < int(self._centroid_sum.shape[1]):
            with torch.no_grad():
                self._centroid_sum[:, int(e_slot), :].zero_()
        if self._centroid_count is not None and int(e_slot) < self._centroid_count.numel():
            self._centroid_count[int(e_slot)] = 0
        if self._centroid_param is not None and 0 <= int(e_slot) < int(self._centroid_param.shape[1]):
            with torch.no_grad():
                self._centroid_param.data[:, int(e_slot), :].zero_()
        if int(e_slot) < self.centroid_n.numel():
            self.centroid_n[int(e_slot)] = 0
        self._maybe_refresh_deepspeed_fp32_after_centroid_data_write()

    def _register_centroid_vis_grad_freeze_hook(self, bm: nn.Module) -> None:
        if self._centroid_param is None or getattr(bm, "_cram_centroid_vis_grad_hook_registered", False):
            return
        if not isinstance(self._centroid_param, nn.Parameter):
            return
        owner = self

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if getattr(owner, "_cram_centroid_block_entire_grad", False):

                grad.zero_()
                return grad
            t_cur = int(getattr(owner, "_prep_task_id_for_grad", 0))
            S = int(grad.shape[1])
            for s in range(S):
                if owner._centroid_expert_row_frozen(s, t_cur):
                    grad[:, s, :].zero_()
            return grad

        self._centroid_param.register_hook(_hook)
        bm._cram_centroid_vis_grad_hook_registered = True

    def _buf_centroid_vector_for_route(self, layer_idx: int, need_detach: bool) -> torch.Tensor:

        li = int(layer_idx)
        d_vis = self._centroid_vis_width(self._buf_centroid_param)
        bc = int(self._buf_count.item()) if self._buf_count is not None else 0
        if bc <= 0 and self._buf_sum is not None:
            v = self._buf_sum[li, :d_vis].float()
        elif self._buf_centroid_param is not None:
            v = self._buf_centroid_param[li, :d_vis].float()
        else:
            dev = self._buf_sum.device if self._buf_sum is not None else torch.device("cpu")
            return torch.zeros(d_vis, dtype=torch.float32, device=dev)
        if need_detach:
            v = v.detach()
        return v

    def _route_rbf_sigma(self) -> float:
        return max(float(getattr(self.config, "cram_route_rbf_sigma", 2.0)), 1e-8)

    @staticmethod
    def _secondary_route_logits_gemm(
        q_1d: torch.Tensor, W_ke: torch.Tensor, *, sigma: float
    ) -> torch.Tensor:

        q = q_1d.reshape(-1).to(device=W_ke.device, dtype=W_ke.dtype)
        sig2 = max(float(sigma), 1e-8) ** 2
        d2 = (W_ke * W_ke).sum(dim=1) + (q * q).sum() - 2.0 * (W_ke @ q)
        return -0.5 * d2.clamp_min(0.0) / sig2

    @staticmethod
    def _route_softmax_weights(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=0)

    def _route_topk(self) -> int:
        return 0

    def _in_stable_training_phase(self) -> bool:
        nw = self._cram_visual_warmup_steps()
        if nw <= 0:
            return True
        return int(self._cram_optimizer_steps_done) >= int(nw)

    def _group_expert_pool_meta(
        self,
        g_star: int,
        layer_idx: int,
        route_model: Optional[nn.Module],
        *,
        include_buf: bool,
        need_tensors: bool,
    ) -> Tuple[List[int], List[bool], torch.Tensor]:

        p = self._centroid_param
        pbuf = self._buf_centroid_param
        ct = self._centroid_count
        if p is None or (include_buf and pbuf is None):
            return [], [], torch.zeros(0, 0)
        d_vis = self._centroid_vis_width(p)
        li = int(layer_idx)
        mask = list(self.group_experts[g_star]) if g_star < len(self.group_experts) else []
        slots: List[int] = []
        is_buf: List[bool] = []
        rows: List[torch.Tensor] = []
        for k in mask:
            nk = (
                int(ct[k].item())
                if ct is not None and k < ct.numel()
                else int(self.centroid_n[k].item() if k < self.centroid_n.numel() else 0)
            )
            in_pool = nk > 0 or (not include_buf and self._expert_slot_committed(route_model, int(k)))
            if not in_pool:
                continue
            w_row = p[li, int(k), :d_vis].float()
            if need_tensors and self._expert_frozen_for_grad(int(k)):
                w_row = w_row.detach()
            slots.append(int(k))
            is_buf.append(False)
            rows.append(w_row)
        if include_buf:
            rbuf = int(self._buf_ranks_ok(route_model if route_model is not None else self._model_ref))
            if rbuf > 0:
                b_vec = self._buf_centroid_vector_for_route(
                    li, need_detach=(need_tensors and not self._buf_trainable())
                )
                slots.append(-1)
                is_buf.append(True)
                rows.append(b_vec)
        if not rows:
            return [], [], torch.zeros(0, d_vis)
        return slots, is_buf, torch.stack(rows, dim=0)

    def _group_expert_pi_dense_tensor(
        self,
        g_star: int,
        q_vis: torch.Tensor,
        layer_idx: int,
        route_model: nn.Module,
        max_slots: int,
        *,
        include_buf: bool,
        route_topk: int,
        fallback_slot: int,
        need_tensors: bool = False,
    ) -> torch.Tensor:

        out = torch.zeros(int(max_slots), dtype=torch.float32)
        slots, _, w_rows = self._group_expert_pool_meta(
            g_star, layer_idx, route_model, include_buf=include_buf, need_tensors=need_tensors
        )
        if w_rows.numel() == 0:
            if 0 <= int(fallback_slot) < int(max_slots):
                out[int(fallback_slot)] = 1.0
            return out.to(device=q_vis.device)
        if q_vis.dim() == 2:
            q = q_vis[0].float().reshape(-1)[: w_rows.shape[1]]
            if not need_tensors:
                q = q.detach()
        else:
            q = q_vis.float().reshape(-1)[: w_rows.shape[1]]
            if not need_tensors:
                q = q.detach()
        sig = self._route_rbf_sigma()
        logits = self._secondary_route_logits_gemm(q, w_rows, sigma=sig)
        n = int(logits.numel())
        k_req = int(route_topk)
        if k_req <= 0 or k_req >= n:
            sel = list(range(n))
        else:
            _, topi = torch.topk(logits, int(k_req), largest=True, sorted=True)
            sel = [int(i) for i in topi.tolist()]
        log_sel = logits[sel]
        w = self._route_softmax_weights(log_sel)
        for j, idx in enumerate(sel):
            slot = int(slots[idx])
            if slot >= 0 and slot < int(max_slots):
                out[slot] = w[j]
        return out.to(device=q_vis.device, dtype=torch.float32)

    def _group_expert_pi_batch_tensor(
        self,
        g_star: int,
        q_batch: torch.Tensor,
        layer_idx: int,
        route_model: nn.Module,
        max_slots: int,
        *,
        include_buf: bool,
        route_topk: int,
        need_tensors: bool,
    ) -> torch.Tensor:

        bsz = int(q_batch.shape[0])
        device = q_batch.device
        out = torch.zeros(bsz, int(max_slots), dtype=torch.float32, device=device)
        slots, _, w_rows = self._group_expert_pool_meta(
            g_star, layer_idx, route_model, include_buf=include_buf, need_tensors=need_tensors
        )
        if w_rows.numel() == 0 or bsz == 0:
            return out
        d_vis = int(w_rows.shape[1])
        q = q_batch.float()[:, :d_vis]
        sig = self._route_rbf_sigma()
        q2 = (q * q).sum(dim=1, keepdim=True)
        w2 = (w_rows * w_rows).sum(dim=1)
        logits = -0.5 * (q2 + w2.unsqueeze(0) - 2.0 * (q @ w_rows.t())).clamp_min(0.0) / (sig**2)
        n = int(w_rows.shape[0])
        k_req = int(route_topk)
        for bi in range(bsz):
            lb = logits[bi]
            if k_req <= 0 or k_req >= n:
                sel = list(range(n))
            else:
                _, topi = torch.topk(lb, int(k_req), largest=True, sorted=True)
                sel = [int(i) for i in topi.tolist()]
            w = self._route_softmax_weights(lb[sel])
            for j, idx in enumerate(sel):
                slot = int(slots[idx])
                if slot >= 0:
                    out[bi, slot] = w[j]
        return out

    def _expert_slot_committed(self, model: Optional[nn.Module], slot: int) -> bool:

        if model is None:
            return False
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        for m in self._iter_cram_linears(model):
            if isinstance(m, CramBudgetLoraLinear) and int(slot) < int(m.lora_cram_expert_mask.numel()):
                if bool(m.lora_cram_expert_mask[int(slot)].item()):
                    return True
        return False

    def _infer_group_expert_pi_dense(
        self,
        g_star: int,
        c_pooled: torch.Tensor,
        layer_idx: int,
        model: nn.Module,
        max_slots: int,
        fallback_slot: int,
        *,
        route_topk: int,
    ) -> torch.Tensor:

        return self._group_expert_pi_dense_tensor(
            g_star,
            c_pooled,
            layer_idx,
            model,
            max_slots,
            include_buf=False,
            route_topk=int(route_topk),
            fallback_slot=int(fallback_slot),
            need_tensors=False,
        ).detach().cpu()

    @staticmethod
    def _pi_expert_buf_for_slot(
        e_slot: int,
        ua_buf: bool,
        ka: int,
        ub_buf: bool,
        kb: int,
        wa: float,
        wb: float,
    ) -> Tuple[float, float]:

        pi_buf = (float(wa) if ua_buf else 0.0) + (float(wb) if ub_buf else 0.0)
        pi_e = 0.0
        if not ua_buf and int(ka) == int(e_slot):
            pi_e += float(wa)
        if not ub_buf and int(kb) == int(e_slot):
            pi_e += float(wb)
        return pi_e, pi_buf

    def _centroid_vis_width(self, p: Optional[torch.Tensor]) -> int:
        if p is None or p.ndim < 2:
            return int(self.feature_dim)
        return min(int(self.feature_dim), int(p.shape[-1]))

    def _load_group_prototypes_from_blob(self, gp: Any) -> None:

        self.group_prototypes = []
        if not isinstance(gp, list):
            return
        for t in gp:
            if isinstance(t, torch.Tensor):
                self.group_prototypes.append(t.detach().cpu().float().clone())
            else:
                try:
                    self.group_prototypes.append(torch.as_tensor(t, dtype=torch.float32).detach().cpu().clone())
                except Exception:
                    _LOG.warning("CRAM load: skip unparseable group prototype type=%s", type(t).__name__)

    def _validate_level1_state_after_load(self) -> None:

        Gp = len(self.group_prototypes)
        for tid, gi in list(self._task_group_lock.items()):
            if int(gi) < 0 or int(gi) >= Gp:
                raise RuntimeError(
                    f"CRAM: level-1 state mismatch — task_group_lock task={tid} -> group={gi}, "
                    f"but only G={Gp} prototypes are loaded."
                )

    def _assert_group_lists_aligned(self, phase: str) -> None:
        Lp = len(self.group_prototypes)
        Lc = len(self.group_counts)
        Lt = len(self.group_tasks)
        Le = len(self.group_experts)
        if Lp == Lc == Lt == Le:
            return
        msg = (
            f"CRAM {phase}: level-1 table lengths mismatch "
            f"Lp={Lp} L_count={Lc} L_tasks={Lt} L_experts={Le} (must be equal)."
        )
        raise RuntimeError(msg)

    def _text_vector(self, text_feat: torch.Tensor) -> torch.Tensor:
        x = text_feat.float().mean(dim=0)
        return F.layer_norm(x, (x.shape[-1],))

    def _assign_level1(self, t_i: torch.Tensor, *, mutate: bool = True) -> Tuple[int, bool, bool]:

        t_i = t_i.detach().float()
        G = len(self.group_prototypes)
        if G == 0:
            if mutate:
                self.group_prototypes.append(t_i.cpu().clone())
                self.group_counts.append(1)
                self.group_tasks.append([])
                self.group_experts.append([])
                return 0, True, True
            return 0, False, False
        sims: List[float] = []
        for g in range(G):
            m = self.group_prototypes[g].to(t_i.device).float()
            sims.append(float(F.cosine_similarity(t_i.unsqueeze(0), m.unsqueeze(0), dim=-1).item()))
        sims_t = torch.tensor(sims, device=t_i.device, dtype=torch.float32)
        topv, topi = torch.topk(sims_t, k=min(2, G))
        s1, g1 = float(topv[0]), int(topi[0])
        new_group = False
        if G == 1:
            thr_hi = 0.8
            if s1 > thr_hi:
                g_star = g1
                lock_now = True
            elif G < self.max_groups:
                if mutate:
                    self.group_prototypes.append(t_i.cpu().clone())
                    self.group_counts.append(1)
                    self.group_tasks.append([])
                    self.group_experts.append([])
                    g_star = len(self.group_prototypes) - 1
                    new_group = True
                    lock_now = True
                else:
                    g_star = g1
                    lock_now = False
            else:
                g_star = g1
                lock_now = False
        else:
            s2 = float(topv[1])
            delta = s1 - s2
            if delta > self.theta:
                g_star = g1
                lock_now = True
            elif G < self.max_groups:
                if mutate:
                    self.group_prototypes.append(t_i.cpu().clone())
                    self.group_counts.append(1)
                    self.group_tasks.append([])
                    self.group_experts.append([])
                    g_star = len(self.group_prototypes) - 1
                    new_group = True
                    lock_now = True
                else:
                    g_star = g1
                    lock_now = False
            else:
                g_star = g1
                lock_now = False
        if mutate and (not new_group and lock_now):
            n_old = self.group_counts[g_star]
            m_old = self.group_prototypes[g_star].float()
            self.group_prototypes[g_star] = (m_old * n_old + t_i.cpu()) / (n_old + 1)
            self.group_counts[g_star] = n_old + 1
        return g_star, new_group, lock_now

    def _pick_level1_group_semantic_infer(self, t_i: torch.Tensor) -> int:

        t_i = t_i.detach().float()
        G = len(self.group_prototypes)
        if G <= 1:
            return 0
        sims: List[float] = []
        for g in range(G):
            m = self.group_prototypes[g].to(t_i.device).float()
            sims.append(float(F.cosine_similarity(t_i.unsqueeze(0), m.unsqueeze(0), dim=-1).item()))
        sims_t = torch.tensor(sims, device=t_i.device, dtype=torch.float32)
        return int(torch.argmax(sims_t).item())

    def _infer_primary_expert_slot_for_group(self, g_star: int) -> int:

        g_star = int(g_star)
        if g_star < 0 or g_star >= len(self.group_experts):
            return 0
        slots = self.group_experts[g_star]
        return int(slots[0]) if slots else 0

    def _assign_level1_with_task_group_lock(
        self, t_i: torch.Tensor, t_cur: int, *, mutate: bool = True
    ) -> Tuple[int, bool]:
        t_cur = int(t_cur)
        if t_cur in self._task_group_lock:
            g_star = int(self._task_group_lock[t_cur])
            if mutate:
                n_old = int(self.group_counts[g_star])
                m_old = self.group_prototypes[g_star].float()
                self.group_prototypes[g_star] = (m_old * n_old + t_i.detach().float().cpu()) / (n_old + 1)
                self.group_counts[g_star] = n_old + 1
            return g_star, False
        g_star, new_group, lock_now = self._assign_level1(t_i, mutate=mutate)
        if mutate and lock_now:
            self._task_group_lock[t_cur] = int(g_star)
        return g_star, new_group

    def _ensure_task_expert(self, g_star: int, t_cur: int, *, mutate: bool = True) -> int:
        if t_cur in self._task_expert_lock:
            e = int(self._task_expert_lock[t_cur])
            if mutate:
                self._attach_task_expert_to_group(g_star, t_cur, e)
            return e
        tasks = self.group_tasks[g_star]
        exps = self.group_experts[g_star]
        if t_cur in tasks:
            e_old = int(exps[tasks.index(t_cur)])
            if mutate:
                self._task_expert_lock[t_cur] = e_old
                self._prune_task_from_other_groups(t_cur, g_star)
            return e_old
        if not mutate:
            raise RuntimeError(
                f"CRAM DDP: non-rank0 still has no expert lock for task={t_cur} after sync "
                f"(g_star={g_star})."
            )
        if self._next_expert_id >= self.max_slots:
            raise RuntimeError(f"CRAM: expert slots exhausted (max_slots={self.max_slots})")
        e_new = int(self._next_expert_id)
        self._next_expert_id += 1
        self._task_expert_lock[t_cur] = e_new
        self._prune_task_from_other_groups(t_cur, g_star)
        tasks.append(int(t_cur))
        exps.append(e_new)
        return e_new

    def _pack_cram_level1_broadcast_payload(self) -> Dict[str, Any]:
        return {
            "group_prototypes": [t.detach().cpu().float().clone() for t in self.group_prototypes],
            "group_counts": [int(x) for x in self.group_counts],
            "group_tasks": [[int(t) for t in x] for x in self.group_tasks],
            "group_experts": [[int(e) for e in x] for x in self.group_experts],
            "task_group_lock": {int(k): int(v) for k, v in self._task_group_lock.items()},
            "task_expert_lock": {int(k): int(v) for k, v in self._task_expert_lock.items()},
            "next_expert_id": int(self._next_expert_id),
        }

    def _apply_cram_level1_broadcast_payload(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self._load_group_prototypes_from_blob(payload.get("group_prototypes"))
        self.group_counts = [int(x) for x in payload.get("group_counts", [])]
        self.group_tasks = [list(x) for x in payload.get("group_tasks", [])]
        self.group_experts = [list(x) for x in payload.get("group_experts", [])]
        self._task_group_lock = {int(k): int(v) for k, v in payload.get("task_group_lock", {}).items()}
        self._task_expert_lock = {int(k): int(v) for k, v in payload.get("task_expert_lock", {}).items()}
        self._next_expert_id = int(payload.get("next_expert_id", self._next_expert_id))

    def _broadcast_cram_level1_tables_training(self, model: nn.Module) -> None:

        if not _dist_cram_training_multi_gpu(model):
            return
        import torch.distributed as dist

        objl: List[Any]
        if _dist_rank0():
            objl = [self._pack_cram_level1_broadcast_payload()]
        else:
            objl = [None]
        dist.broadcast_object_list(objl, src=0)
        if not _dist_rank0():
            self._apply_cram_level1_broadcast_payload(objl[0])

    def _prune_task_from_other_groups(self, t_cur: int, keep_g: int) -> None:

        t_cur = int(t_cur)
        keep_g = int(keep_g)
        for g, tasks in enumerate(self.group_tasks):
            if g == keep_g:
                continue
            exps = self.group_experts[g]
            while t_cur in tasks:
                i = tasks.index(t_cur)
                tasks.pop(i)
                exps.pop(i)

    def _attach_task_expert_to_group(self, g_star: int, t_cur: int, e: int) -> None:
        if g_star < 0 or g_star >= len(self.group_tasks):
            return
        self._prune_task_from_other_groups(t_cur, g_star)
        tasks = self.group_tasks[g_star]
        exps = self.group_experts[g_star]
        if t_cur in tasks:
            i = tasks.index(t_cur)
            exps[i] = int(e)
            return
        tasks.append(int(t_cur))
        exps.append(int(e))

    def _extract_clip_features(self, model, images, input_ids, clip_tokenizer, text_tower):
        device = images.device if images is not None else next(model.parameters()).device
        image_feat = None
        if images is not None:
            image_feat = extract_routing_image_features(model, images)
            if image_feat is not None and int(image_feat.shape[-1]) != int(self.feature_dim):
                raise RuntimeError(
                    f"CRAM: routing image_feat last dim {int(image_feat.shape[-1])} != "
                    f"clip_feature_dim {int(self.feature_dim)}. Use routing_vision_tower "
                    "(CLIP 768-d image_embeds), not the MLLM vision tower patch features."
                )
        if input_ids is None:
            text_feat = torch.randn(1, self.feature_dim, device=device)
        else:
            main_tokenizer = getattr(model, "tokenizer", None)
            tok = main_tokenizer or clip_tokenizer
            input_pad = np.where(
                input_ids.cpu().detach().numpy() != -200,
                input_ids.cpu().detach().numpy(),
                tok.pad_token_id,
            )
            decoded = tok.batch_decode(input_pad, skip_special_tokens=True)
            decoded_hidden = ["\n".join(d.split("\n")[1:]) for d in decoded]
            decoded_clip = [d.split(" ASSISTANT")[0] for d in decoded_hidden]
            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_feat = text_tower(clip_inputs)
                text_feat = text_feat[0] if isinstance(text_feat, tuple) else text_feat
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        return image_feat, text_feat

    def _ensure_centroid_tables(self, cl_model: nn.Module) -> None:
        root = self._unwrap_training_model(cl_model)
        bm = getattr(root, "_base_model", None)
        if bm is None:
            return
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        L = 0
        for m in root.modules():
            if isinstance(m, CramBudgetLoraLinear):
                L = max(L, int(m.layer_id) + 1)
        if L <= 0:
            L = 1
        d = int(self.feature_dim)
        S = int(self.max_slots)

        v_ex = getattr(bm, "cram_centroid_vis", None)
        if v_ex is None:
            bm.register_parameter(
                "cram_centroid_vis",
                nn.Parameter(torch.zeros(L, S, d, dtype=torch.float32), requires_grad=True),
            )
        elif isinstance(v_ex, nn.Parameter):
            pass
        elif "cram_centroid_vis" in bm._buffers:
            w = bm._buffers["cram_centroid_vis"].detach().clone()
            del bm._buffers["cram_centroid_vis"]
            bm.register_parameter("cram_centroid_vis", nn.Parameter(w, requires_grad=True))
        v_bf = getattr(bm, "cram_buf_centroid", None)
        if v_bf is None:
            bm.register_buffer("cram_buf_centroid", torch.zeros(L, d, dtype=torch.float32))
        elif isinstance(v_bf, nn.Parameter):
            w = v_bf.detach().clone()
            if "cram_buf_centroid" in bm._parameters:
                del bm._parameters["cram_buf_centroid"]
            bm.register_buffer("cram_buf_centroid", w)
        if getattr(bm, "cram_centroid_sum", None) is None:
            bm.register_buffer("cram_centroid_sum", torch.zeros(L, S, d, dtype=torch.float32))
        if getattr(bm, "cram_centroid_count", None) is None:
            bm.register_buffer("cram_centroid_count", torch.zeros(S, dtype=torch.long))
        if getattr(bm, "cram_buf_centroid_sum", None) is None:
            bm.register_buffer("cram_buf_centroid_sum", torch.zeros(L, d, dtype=torch.float32))
        if getattr(bm, "cram_buf_centroid_count", None) is None:
            bm.register_buffer("cram_buf_centroid_count", torch.zeros((), dtype=torch.long))

        if int(bm.cram_centroid_vis.shape[1]) != int(S):
            raise RuntimeError("CRAM: max_slots does not match existing cram_centroid_vis width.")

        self._centroid_param = bm.cram_centroid_vis
        self._buf_centroid_param = bm.cram_buf_centroid
        self._centroid_sum = bm.cram_centroid_sum
        self._centroid_count = bm.cram_centroid_count
        self._buf_sum = bm.cram_buf_centroid_sum
        self._buf_count = bm.cram_buf_centroid_count

        if self.centroid_n.numel() != self.max_slots:
            self.centroid_n = torch.zeros(self.max_slots, dtype=torch.long)

    def _iter_cram_linears(self, model: nn.Module):
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        root = self._unwrap_training_model(model)
        for m in root.modules():
            if isinstance(m, CramBudgetLoraLinear):
                yield m

    def _buf_train_rank(self) -> int:

        Rtot = int(getattr(self.config, "cram_rank_total", 48))
        return max(1, Rtot)

    def _sync_r_buf_all_layers(self, model: nn.Module) -> None:
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear, sync_cram_r_buf_fixed

        r_train = self._buf_train_rank()
        for m in self._iter_cram_linears(model):
            if isinstance(m, CramBudgetLoraLinear):
                sync_cram_r_buf_fixed(m, r_train)

    def _buf_ranks_ok(self, model: Optional[nn.Module]) -> int:
        if model is None:

            return int(self._buf_train_rank())
        r = None
        for m in self._iter_cram_linears(model):
            r = int(m.r_buf) if r is None else min(r, int(m.r_buf))
        return int(r or 0)

    def _centroid_agg_mode(self) -> str:

        return "image_sum"

    def _flush_buf_centroid_from_accumulators(self) -> None:
        p = self._buf_centroid_param
        sm = self._buf_sum
        ct = self._buf_count
        if p is None or sm is None or ct is None:
            return
        d_vis = self._centroid_vis_width(p)
        L = int(p.shape[0])
        c = max(int(ct.item()), 1)
        with torch.no_grad():
            for ell in range(L):
                p.data[ell, :d_vis].copy_((sm[ell, :d_vis] / float(c)).to(device=p.device, dtype=p.dtype))

    def _flush_expert_centroid_from_accumulators(self, slot: int) -> None:
        p = self._centroid_param
        sm = self._centroid_sum
        ct = self._centroid_count
        if p is None or sm is None or ct is None or slot < 0 or slot >= int(ct.numel()):
            return
        d_vis = self._centroid_vis_width(p)
        L = int(p.shape[0])
        c = max(int(ct[slot].item()), 1)
        with torch.no_grad():
            for ell in range(L):
                p.data[ell, slot, :d_vis].copy_((sm[ell, slot, :d_vis] / float(c)).to(device=p.device, dtype=p.dtype))
        if slot < self.centroid_n.numel():
            self.centroid_n[slot] = int(ct[slot].item())

    def _sync_centroid_sum_from_vis_for_slot(self, slot: int) -> None:

        p, sm, ct = self._centroid_param, self._centroid_sum, self._centroid_count
        if p is None or sm is None or ct is None:
            return
        slot = int(slot)
        if slot < 0 or slot >= int(ct.numel()):
            return
        d_vis = self._centroid_vis_width(p)
        L = int(p.shape[0])
        c = max(int(ct[slot].item()), 1)
        with torch.no_grad():
            for ell in range(L):
                sm[ell, slot, :d_vis].copy_(
                    (p.data[ell, slot, :d_vis].float() * float(c)).to(device=sm.device, dtype=sm.dtype)
                )

    def _repair_expert_centroid_vis_columns_from_sum_if_degenerate(self) -> int:

        p = self._centroid_param
        sm = self._centroid_sum
        ct = self._centroid_count
        if p is None or sm is None or ct is None:
            return 0
        d_vis = self._centroid_vis_width(p)
        eps = 1e-5
        nfixed = 0
        for k in range(int(ct.numel())):
            if int(ct[k].item()) <= 0:
                continue
            col = p[:, k, :d_vis].detach().float()
            if float(torch.norm(col)) >= eps:
                continue
            smcol = sm[:, k, :d_vis].detach().float()
            if float(torch.norm(smcol)) < eps:

                continue
            self._flush_expert_centroid_from_accumulators(int(k))
            nfixed += 1
        return nfixed

    def _repair_buf_centroid_from_sum_if_degenerate(self) -> bool:

        p = self._buf_centroid_param
        sm = self._buf_sum
        ct = self._buf_count
        if p is None or sm is None or ct is None:
            return False
        if int(ct.item()) <= 0:
            return False
        d_vis = self._centroid_vis_width(p)
        col = p[:, :d_vis].detach().float()
        if float(torch.norm(col)) >= 1e-5:
            return False
        self._flush_buf_centroid_from_accumulators()
        return True

    def _accumulate_buf_centroid_exact(self, x_bd: torch.Tensor, c_mean: torch.Tensor, img_b: int) -> None:

        p = self._buf_centroid_param
        sm = self._buf_sum
        ct = self._buf_count
        if p is None or sm is None or ct is None or img_b <= 0:
            return
        d_vis = self._centroid_vis_width(p)
        L = int(p.shape[0])
        mode = self._centroid_agg_mode()
        sm0 = sm.detach().clone()
        ct0 = ct.detach().clone()
        with torch.no_grad():
            if mode == "image_sum":
                c_sum_vis = x_bd[:, :d_vis].sum(dim=0).detach().float()
                add = int(img_b)
            else:
                c_sum_vis = c_mean.detach().float().reshape(-1)[:d_vis]
                add = 1
            for ell in range(L):
                sm[ell, :d_vis].add_(c_sum_vis.cpu() if sm.device.type == "cpu" else c_sum_vis.to(sm.device))
            ct.add_(add)
        self._dist_allreduce_buf_centroid_delta_from(sm0, ct0)
        self._flush_buf_centroid_from_accumulators()

    def _accumulate_expert_centroid_exact(self, slot: int, x_bd: torch.Tensor, c_mean: torch.Tensor, img_b: int) -> None:
        p = self._centroid_param
        sm = self._centroid_sum
        ct = self._centroid_count
        if p is None or sm is None or ct is None or slot < 0 or img_b <= 0:
            return
        d_vis = self._centroid_vis_width(p)
        L = int(p.shape[0])
        mode = self._centroid_agg_mode()
        s0 = sm[:, int(slot), :].detach().clone()
        c0v = ct[int(slot)].detach().clone()
        with torch.no_grad():
            if mode == "image_sum":
                c_sum_vis = x_bd[:, :d_vis].sum(dim=0).detach().float()
                add = int(img_b)
            else:
                c_sum_vis = c_mean.detach().float().reshape(-1)[:d_vis]
                add = 1
            for ell in range(L):
                sm[ell, slot, :d_vis].add_(c_sum_vis.cpu() if sm.device.type == "cpu" else c_sum_vis.to(sm.device))
            ct[slot] = int(ct[slot].item()) + add
        self._dist_allreduce_expert_centroid_delta_from(int(slot), s0, c0v)
        self._flush_expert_centroid_from_accumulators(slot)

    def _dist_allreduce_buf_centroid_delta_from(self, sm0: torch.Tensor, ct0: torch.Tensor) -> None:

        try:
            import torch.distributed as dist
        except Exception:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        sm, ct = self._buf_sum, self._buf_count
        if sm is None or ct is None:
            return
        dsm = sm - sm0
        dct = ct.float() - ct0.float()
        dist.all_reduce(dsm, op=dist.ReduceOp.SUM)
        dist.all_reduce(dct, op=dist.ReduceOp.SUM)
        sm.copy_(sm0 + dsm)
        ct.copy_((ct0.float() + dct).to(dtype=ct.dtype))

    def _dist_allreduce_expert_centroid_delta_from(
        self, slot: int, s0: torch.Tensor, c0: torch.Tensor
    ) -> None:

        try:
            import torch.distributed as dist
        except Exception:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        sm, ct = self._centroid_sum, self._centroid_count
        if sm is None or ct is None or slot < 0 or slot >= int(sm.shape[1]):
            return
        d = sm[:, slot, :] - s0
        dc = (ct[slot] - c0).view(1).to(dtype=torch.int64)
        dist.all_reduce(d, op=dist.ReduceOp.SUM)
        dist.all_reduce(dc, op=dist.ReduceOp.SUM)
        sm[:, slot, :].copy_(s0 + d)
        merged = (c0.to(torch.int64).reshape(1) + dc).to(dtype=ct.dtype).reshape(())
        ct[slot].copy_(merged)

    def _route_pool_top2(
        self,
        g_star: int,
        c_pooled: torch.Tensor,
        layer_idx: int,
        *,
        need_tensors: bool,
        route_model: Optional[nn.Module] = None,
        include_buf: bool = True,
        fallback_expert_slot: Optional[int] = None,
    ) -> Tuple[bool, int, bool, int, float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        p = self._centroid_param
        pbuf = self._buf_centroid_param
        ct = self._centroid_count
        if p is None or (include_buf and pbuf is None):
            return False, 0, False, 0, 1.0, 0.0, None, None
        d_vis = self._centroid_vis_width(p)
        c_vis = c_pooled.detach().float().reshape(-1)[:d_vis]
        li = int(layer_idx)
        mask = list(self.group_experts[g_star]) if g_star < len(self.group_experts) else []
        meta: List[Tuple[bool, int]] = []
        rows: List[torch.Tensor] = []
        for k in mask:
            nk = int(ct[k].item()) if ct is not None and k < ct.numel() else int(self.centroid_n[k].item() if k < self.centroid_n.numel() else 0)
            in_pool = nk > 0 or (not include_buf and self._expert_slot_committed(route_model, int(k)))
            if not in_pool:
                continue
            w_row = p[li, int(k), :d_vis].float()
            if need_tensors and self._expert_frozen_for_grad(int(k)):
                w_row = w_row.detach()
            meta.append((False, int(k)))
            rows.append(w_row)
        if include_buf:
            rbuf = int(self._buf_ranks_ok(route_model if route_model is not None else self._model_ref))
            if rbuf > 0:
                b_vec = self._buf_centroid_vector_for_route(li, need_detach=(need_tensors and not self._buf_trainable()))
                meta.append((True, -1))
                rows.append(b_vec)
        if not meta:
            fb = fallback_expert_slot
            if fb is not None and int(fb) >= 0:
                k = int(fb)
                return False, k, False, k, 1.0, 0.0, None, None
            return False, 0, False, 0, 1.0, 0.0, None, None

        Wm = torch.stack(rows, dim=0)
        sig = self._route_rbf_sigma()
        logits = self._secondary_route_logits_gemm(c_vis, Wm, sigma=sig)
        n = int(logits.numel())
        if n == 1:
            ua, ka = meta[0]
            return ua, ka, ua, ka, 1.0, 0.0, None, None

        topv, topi = torch.topk(logits, min(2, n), largest=True, sorted=True)
        i0, i1 = int(topi[0]), int(topi[1])
        ua, ka = meta[i0]
        ub, kb = meta[i1]
        st0, st1 = logits[i0], logits[i1]

        pair = torch.stack([st0, st1])
        w_pair = self._route_softmax_weights(pair)
        w0, w1 = w_pair[0], w_pair[1]
        if need_tensors:
            return (
                ua,
                ka,
                ub,
                kb,
                float(w0.detach().item()),
                float(w1.detach().item()),
                w0,
                w1,
            )
        return ua, ka, ub, kb, float(w0), float(w1), None, None

    def _expert_frozen_for_grad(self, expert_idx: int) -> bool:
        if not getattr(self, "_cram_forward_train_hard", False):
            return False
        tid = None
        for ts, es in zip(self.group_tasks, self.group_experts):
            for t, e in zip(ts, es):
                if int(e) == int(expert_idx):
                    tid = int(t)
                    break
        if tid is None:
            return True
        return tid < int(self._prep_task_id_for_grad)

    def _sync_task_centroid_display_from_buf_for_slot(self, slot: int) -> None:

        bp = self._buf_centroid_param
        cp = self._centroid_param
        if bp is None or cp is None:
            return
        slot = int(slot)
        d_vis = self._centroid_vis_width(cp)
        with torch.no_grad():
            cp.data[:, slot, :d_vis].copy_(bp[:, :d_vis].to(device=cp.data.device, dtype=cp.data.dtype))

    def _buf_trainable(self) -> bool:

        if self._buf_centroid_param is None:
            return False
        nw = self._cram_visual_warmup_steps()
        if nw <= 0:
            return False
        return int(self._cram_optimizer_steps_done) < nw

    def initialize_model(self, model) -> None:
        self._model_ref = model
        for _, p in model.named_parameters():
            p.requires_grad = False
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraConfig, cram_set_route_num_layers, set_cram_budget_route_all_layers
        from PEFT.utils.config import TaskType

        target_modules = collect_peft_target_linear_suffixes(model, self.config)
        Rtot = int(getattr(self.config, "cram_rank_total", 48))
        rmax = 10 if str(getattr(self.config, "benchmark", "") or "").strip().lower() == "trigap" else 9
        peft_config = CramBudgetLoraConfig(
            target_modules=target_modules,
            r=Rtot,
            lora_alpha=2 * Rtot,
            lora_dropout=0.05,
            cram_rank_total=Rtot,
            cram_expert_rank_max=rmax,
            cram_max_expert_slots=self.max_slots,
            expert_num=self.max_slots,
            cur_task=int(getattr(self.config, "cur_task", 0)),
            task_type=TaskType.CAUSAL_LM_CRAM,
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
            model._modules["_base_model"] = peft_model
        else:
            peft_model = get_peft_model(model, peft_config)
            if peft_model is not model:
                object.__setattr__(model, "_base_model", peft_model)
                model._modules["_base_model"] = peft_model

        self._ensure_centroid_tables(model)
        bm = getattr(model, "_base_model", None)
        if self._centroid_param is not None:
            self._centroid_param.requires_grad_(True)
        if bm is not None:
            self._register_centroid_vis_grad_freeze_hook(bm)

        L = 0
        for m in self._iter_cram_linears(model):
            L = max(L, int(m.layer_id) + 1)
        cram_set_route_num_layers(max(1, L))
        set_cram_budget_route_all_layers(0, 0, True, True, 1.0, 0.0)
        self._apply_cram_lora_deepspeed_safe_grad_phase(model, "warm", None)
        self._sync_r_buf_all_layers(model)

    def _clear_cram_lora_grad_hooks(self) -> None:
        for h in self._cram_lora_grad_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._cram_lora_grad_hook_handles.clear()
        self._cram_stable_train_expert_slot = None

    def _apply_cram_lora_deepspeed_safe_grad_phase(
        self,
        model: nn.Module,
        phase: str,
        stable_expert_slot: Optional[int],
    ) -> None:

        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        self._clear_cram_lora_grad_hooks()
        ph = (phase or "warm").lower().strip()
        for m in model.modules():
            if not isinstance(m, CramBudgetLoraLinear):
                continue
            m.lora_cram_buf_A.requires_grad_(True)
            m.lora_cram_buf_B.requires_grad_(True)
            m.lora_cram_expert_A.requires_grad_(True)
            m.lora_cram_expert_B.requires_grad_(True)

        owner = self

        def _zero_full_grad(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            grad.zero_()
            return grad

        if ph == "warm":
            for m in model.modules():
                if not isinstance(m, CramBudgetLoraLinear):
                    continue
                self._cram_lora_grad_hook_handles.append(m.lora_cram_expert_A.register_hook(_zero_full_grad))
                self._cram_lora_grad_hook_handles.append(m.lora_cram_expert_B.register_hook(_zero_full_grad))
            return

        if ph == "stable" and stable_expert_slot is not None:
            ts = int(stable_expert_slot)
            self._cram_stable_train_expert_slot = ts

            def _mask_expert_rows(grad: torch.Tensor) -> torch.Tensor:
                if grad is None:
                    return grad
                s = owner._cram_stable_train_expert_slot
                if s is None:
                    return grad
                for i in range(int(grad.shape[0])):
                    if i != int(s):
                        grad[i].zero_()
                return grad

            for m in model.modules():
                if not isinstance(m, CramBudgetLoraLinear):
                    continue
                self._cram_lora_grad_hook_handles.append(m.lora_cram_buf_A.register_hook(_zero_full_grad))
                self._cram_lora_grad_hook_handles.append(m.lora_cram_buf_B.register_hook(_zero_full_grad))
                self._cram_lora_grad_hook_handles.append(m.lora_cram_expert_A.register_hook(_mask_expert_rows))
                self._cram_lora_grad_hook_handles.append(m.lora_cram_expert_B.register_hook(_mask_expert_rows))
            return

        _LOG.warning("CRAM: unknown lora grad phase %r, fallback to warm", phase)
        self._apply_cram_lora_deepspeed_safe_grad_phase(model, "warm", None)

    def _freeze_expert_lora_train_only_buf(self, model: nn.Module) -> None:

        self._apply_cram_lora_deepspeed_safe_grad_phase(model, "warm", None)

    def _freeze_buf_train_expert_slot_only(self, model: nn.Module, expert_slot: int) -> None:

        self._apply_cram_lora_deepspeed_safe_grad_phase(model, "stable", int(expert_slot))

    def _run_buf_svd_into_expert_slot(
        self,
        model: nn.Module,
        task_id: int,
        slot: int,
    ) -> None:

        from PEFT.tuners.custom.cram_lora import consolidate_buf_into_expert_slot, sync_num_committed_from_mask

        self._ensure_centroid_tables(model)
        slot = int(slot)
        self._sync_task_centroid_display_from_buf_for_slot(slot)
        tau_a = float(getattr(self.config, "cram_svd_tau_alloc", 0.08))
        rmax = 10 if str(getattr(self.config, "benchmark", "") or "").strip().lower() == "trigap" else 9
        tau_novel = float(getattr(self.config, "cram_svd_tau_novel", 0.99))
        rank_min = 4
        hist_slots = self._hist_expert_slots_same_semantic_pool(int(task_id), slot)

        buf_snap = None
        if self._buf_centroid_param is not None:
            buf_snap = self._buf_centroid_param.detach().float().cpu().clone()

        mods = sorted(self._iter_cram_linears(model), key=lambda m: int(m.layer_id))
        r_ts: List[Tuple[int, int]] = []
        for m in mods:
            r_t = consolidate_buf_into_expert_slot(
                m,
                slot,
                tau_a,
                rmax,
                tau_novel=tau_novel,
                hist_expert_slots=hist_slots,
                rank_min=rank_min,
            )
            r_ts.append((int(m.layer_id), int(r_t)))
            sync_num_committed_from_mask(m)

        r_slot_max = max((rt for _, rt in r_ts), default=0)
        if 0 <= slot < len(self.expert_budget_charge):
            self.expert_budget_charge[slot] = int(r_slot_max)

        if buf_snap is not None and self._centroid_param is not None:
            d_vis = self._centroid_vis_width(self._centroid_param)
            with torch.no_grad():
                self._centroid_param.data[:, slot, :d_vis].copy_(
                    buf_snap[:, :d_vis].to(
                        device=self._centroid_param.data.device,
                        dtype=self._centroid_param.data.dtype,
                    )
                )

        if self._centroid_count is not None and slot < self._centroid_count.numel():
            self._centroid_count[slot] = max(int(self._centroid_count[slot].item()), 1)
        if slot < self.centroid_n.numel():
            self.centroid_n[slot] = max(int(self.centroid_n[slot].item()), 1)
        self._sync_centroid_sum_from_vis_for_slot(slot)

        if self._buf_sum is not None:
            self._buf_sum.zero_()
        if self._buf_count is not None:
            self._buf_count.zero_()
        if self._buf_centroid_param is not None:
            with torch.no_grad():
                self._buf_centroid_param.data.zero_()

        self._reinit_buf_weights(model)
        self._sync_r_buf_all_layers(model)
        self._maybe_refresh_deepspeed_fp32_after_centroid_data_write()

    def on_input_prep(self, model: Any, args: tuple, kwargs: dict, context: CLContext) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None
        attention_mask = args[2] if args and len(args) > 2 else None
        self._apply_routing(model, images, input_ids, context)
        self._cram_dec_visual_mask = None
        if (
            self._dec_lambda() > 0
            and getattr(self, "_cram_forward_train_hard", False)
            and self._in_stable_training_phase()
            and input_ids is not None
        ):
            try:
                self._cram_dec_visual_mask = _build_visual_token_mask_expanded(
                    input_ids,
                    attention_mask,
                    num_patches=_resolve_visual_token_patch_count(model),
                )
            except Exception as e:
                _LOG.warning("CRAM L_dec: failed to build visual token mask: %s", e)
                self._cram_dec_visual_mask = None

    def pre_generate_hook(self, model: Any, input_ids: Any, images: Any, context: CLContext) -> bool:
        self._apply_routing(model, images, input_ids, context)
        return True

    def _apply_routing(self, model: Any, images: Any, input_ids: Any, context: CLContext) -> None:
        from PEFT.tuners.custom.cram_lora import (
            clear_cram_infer_group_softmax_routing,
            cram_set_route_num_layers,
            set_cram_budget_route_layer,
            set_cram_infer_group_softmax_layer,
        )

        self._prep_invoke += 1
        self._ensure_centroid_tables(model)
        clip_tokenizer = resolve_clip_tokenizer(model)
        text_tower = resolve_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return
        if input_ids is None or (hasattr(input_ids, "shape") and input_ids.shape[1] <= 1):
            return

        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
        train_hard = model.training and torch.is_grad_enabled()
        self._cram_forward_train_hard = bool(train_hard)
        if train_hard:
            tid = getattr(context, "task_id", None)
            if tid is None:
                tid = int(getattr(self.config, "cur_task", 0))
            t_cur = int(tid)
            self._prep_task_id_for_grad = t_cur
        else:
            t_cur = -1

        t_i = self._text_vector(text_feat)
        if not train_hard:
            g_star = self._pick_level1_group_semantic_infer(t_i)
            new_grp = False
            e_map = self._infer_primary_expert_slot_for_group(g_star)
        else:
            ddp_train = _dist_cram_training_multi_gpu(model)
            if ddp_train and not _dist_rank0():
                self._broadcast_cram_level1_tables_training(model)
                g_star, new_grp = self._assign_level1_with_task_group_lock(t_i, t_cur, mutate=False)
                e_map = self._ensure_task_expert(g_star, t_cur, mutate=False)
            else:
                g_star, new_grp = self._assign_level1_with_task_group_lock(t_i, t_cur, mutate=True)
                e_map = self._ensure_task_expert(g_star, t_cur, mutate=True)
                if ddp_train and _dist_rank0():
                    self._broadcast_cram_level1_tables_training(model)

        if train_hard:
            if self._cram_last_centroid_reset_task is None or int(self._cram_last_centroid_reset_task) != int(t_cur):
                self._reset_buf_and_current_expert_centroids_for_task(model, int(e_map))
                self._cram_last_centroid_reset_task = int(t_cur)

        L = 0
        for m in self._iter_cram_linears(model):
            L = max(L, int(m.layer_id) + 1)
        cram_set_route_num_layers(max(1, L))
        clear_cram_infer_group_softmax_routing()

        nw_early = self._cram_visual_warmup_steps()
        self._cram_centroid_block_entire_grad = bool(
            train_hard and nw_early > 0 and self._cram_optimizer_steps_done < nw_early
        )
        if self._centroid_param is not None and isinstance(self._centroid_param, nn.Parameter):
            self._centroid_param.requires_grad_(bool(train_hard))

        if image_feat is None:
            for ell in range(L):
                set_cram_budget_route_layer(ell, e_map, e_map, False, False, 1.0, 0.0)
            return

        x_bd, c_pooled, img_b = _prep_image_feat_batches_dm(image_feat)
        nw = self._cram_visual_warmup_steps()

        in_warm = nw > 0 and self._cram_optimizer_steps_done < nw

        if train_hard and in_warm:
            self._cram_dec_route_ctx = None
            self._cram_dec_image_feat = None
            for ell in range(L):
                set_cram_budget_route_layer(ell, 0, 0, True, True, 1.0, 0.0)
            self._accumulate_buf_centroid_exact(x_bd, c_pooled, img_b)
            self._accumulate_expert_centroid_exact(e_map, x_bd, c_pooled, img_b)
        elif train_hard:

            need_mid_svd = (
                nw > 0
                and self._cram_optimizer_steps_done >= nw
                and not self._cram_buf_svd_done_for_task.get(t_cur, False)
            )
            if need_mid_svd and self._expert_slot_committed(model, int(e_map)):
                self._cram_buf_svd_done_for_task[t_cur] = True
                self._freeze_buf_train_expert_slot_only(model, int(e_map))
            elif need_mid_svd:
                self._run_buf_svd_into_expert_slot(model, t_cur, int(e_map))
                self._cram_buf_svd_done_for_task[t_cur] = True
                self._freeze_buf_train_expert_slot_only(model, int(e_map))
            include_buf = nw <= 0
            want_t = train_hard and self._centroid_param is not None and bool(self._centroid_param.requires_grad)
            route_topk = self._route_topk()
            hist_slots = self._hist_expert_slots_same_semantic_pool(int(t_cur), int(e_map))
            self._cram_dec_image_feat = x_bd
            self._cram_dec_route_ctx = {
                "g_star": int(g_star),
                "e_slot": int(e_map),
                "t_cur": int(t_cur),
                "hist_slots": list(hist_slots),
                "include_buf": bool(include_buf),
            }
            for ell in range(L):
                pi_t = self._group_expert_pi_dense_tensor(
                    g_star,
                    c_pooled,
                    ell,
                    model,
                    int(self.max_slots),
                    include_buf=include_buf,
                    route_topk=route_topk,
                    fallback_slot=int(e_map),
                    need_tensors=bool(want_t),
                )
                set_cram_infer_group_softmax_layer(ell, pi_t if want_t else pi_t.detach())
        else:
            infer_topk = self._route_topk()
            for ell in range(L):
                pi_dense = self._infer_group_expert_pi_dense(
                    g_star,
                    c_pooled,
                    ell,
                    model,
                    int(self.max_slots),
                    int(e_map),
                    route_topk=infer_topk,
                )
                set_cram_infer_group_softmax_layer(ell, pi_dense)

    def _dec_lambda(self) -> float:
        return 1.0

    def _should_compute_dec_loss(self) -> bool:
        return (
            self._dec_lambda() > 0
            and bool(getattr(self, "_cram_forward_train_hard", False))
            and self._in_stable_training_phase()
        )

    def _clear_cram_dec_capture(self) -> None:
        for h in self._cram_dec_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._cram_dec_hook_handles.clear()
        self._cram_dec_layer_inputs.clear()

    def _register_cram_dec_hooks(self, model: nn.Module) -> None:
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        self._clear_cram_dec_capture()

        def _hook_fn(module: CramBudgetLoraLinear, inp: Tuple[Any, ...], _out: Any) -> None:
            if not inp or inp[0] is None:
                return
            self._cram_dec_layer_inputs[int(module.layer_id)] = inp[0]

        for m in self._iter_cram_linears(model):
            self._cram_dec_hook_handles.append(m.register_forward_hook(_hook_fn))

    @staticmethod
    def _visual_token_mean_hidden(x: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:

        seq = int(x.shape[1])
        m = visual_mask[:, :seq].to(device=x.device, dtype=x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def _compute_cram_dec_loss(self, model: nn.Module) -> Optional[torch.Tensor]:
        ctx = self._cram_dec_route_ctx
        if ctx is None or self._cram_dec_visual_mask is None or self._cram_dec_image_feat is None:
            return None
        hist_slots: List[int] = list(ctx.get("hist_slots") or [])
        if not hist_slots:
            return None
        e_slot = int(ctx["e_slot"])
        g_star = int(ctx["g_star"])
        route_topk = self._route_topk()
        image_feat = self._cram_dec_image_feat
        if image_feat is None or int(image_feat.shape[0]) == 0:
            return None
        if not self._cram_dec_layer_inputs:
            return None

        from PEFT.tuners.custom.cram_lora import (
            CramBudgetLoraLinear,
            _cram_hist_expert_a_row_basis,
            cram_hist_expert_output_basis,
        )

        layer_losses: List[torch.Tensor] = []
        for ell, x_in in sorted(self._cram_dec_layer_inputs.items(), key=lambda t: t[0]):
            mod: Optional[CramBudgetLoraLinear] = None
            for m in self._iter_cram_linears(model):
                if int(m.layer_id) == int(ell):
                    mod = m
                    break
            if mod is None:
                continue
            h_vis = self._visual_token_mean_hidden(x_in, self._cram_dec_visual_mask)
            z_new = mod.expert_delta(h_vis, e_slot)
            if int(mod.in_features) == int(mod.out_features):
                q_basis = _cram_hist_expert_a_row_basis(
                    mod, e_slot, device=z_new.device, hist_expert_slots=hist_slots
                )
            else:
                q_basis = cram_hist_expert_output_basis(
                    mod, e_slot, device=z_new.device, hist_expert_slots=hist_slots
                )
            if q_basis is None:
                continue
            # QR basis is float32; expert_delta follows LoRA dtype (bf16 in this run).
            z = z_new.to(dtype=q_basis.dtype)
            proj = z @ q_basis
            energy = (proj * proj).sum(dim=1)
            pi_b = self._group_expert_pi_batch_tensor(
                g_star,
                image_feat,
                int(ell),
                model,
                int(self.max_slots),
                include_buf=False,
                route_topk=route_topk,
                need_tensors=True,
            )
            gamma = pi_b[:, hist_slots].sum(dim=1).to(dtype=energy.dtype)
            layer_losses.append((gamma * energy).mean())

        if not layer_losses:
            return None
        return self._dec_lambda() * torch.stack(layer_losses).mean()

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        self._clear_cram_dec_capture()
        if self._should_compute_dec_loss():
            self._register_cram_dec_hooks(model)

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        try:
            if not model.training:
                return outputs
            loss = getattr(outputs, "loss", None)
            if loss is None or not self._should_compute_dec_loss():
                return outputs
            l_dec = self._compute_cram_dec_loss(model)
            if l_dec is not None:
                outputs.loss = loss + l_dec
        finally:
            self._clear_cram_dec_capture()
        return outputs

    def on_step_end(
        self,
        model: nn.Module,
        context: CLContext,
        loss: Optional[torch.Tensor] = None,
        *,
        global_step: Optional[int] = None,
    ) -> None:
        if global_step is not None:
            self._cram_optimizer_steps_done = int(global_step)

    def on_train_begin(self, model: nn.Module, global_step: Optional[int], trainer: Any = None, **kwargs) -> None:
        if global_step is not None:
            self._cram_optimizer_steps_done = int(global_step)
        self._refresh_cram_visual_warmup_steps(trainer)

    def on_train_task_finished(
        self,
        model: nn.Module,
        context: CLContext,
        task_id: int,
        trainer: Any = None,
    ) -> None:
        self._ensure_centroid_tables(model)
        slot = int(self._task_expert_lock.get(int(task_id), self._next_expert_id - 1))

        if self._cram_buf_svd_done_for_task.get(int(task_id), False):
            self._freeze_expert_lora_train_only_buf(model)
            return

        self._run_buf_svd_into_expert_slot(model, int(task_id), slot)
        self._freeze_expert_lora_train_only_buf(model)

    def _reinit_buf_weights(self, model: nn.Module) -> None:
        from PEFT.tuners.custom.cram_lora import CramBudgetLoraLinear

        for m in self._iter_cram_linears(model):
            if not isinstance(m, CramBudgetLoraLinear):
                continue
            r = int(m.r_buf)
            with torch.no_grad():
                m.lora_cram_buf_A.zero_()
                m.lora_cram_buf_B.zero_()
                nn.init.kaiming_uniform_(m.lora_cram_buf_A[:r], a=math.sqrt(5))
                m.lora_cram_buf_B[:, :r].zero_()

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        return

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        if model is not None:
            self._ensure_centroid_tables(model)
        cv = torch.zeros(0)
        bc = torch.zeros(0)
        if self._centroid_param is not None:
            cv = self._centroid_param.detach().cpu().float().clone()
            if self._centroid_sum is not None and self._centroid_count is not None:
                d_vis = self._centroid_vis_width(self._centroid_param)
                sm = self._centroid_sum.detach().cpu().float()
                cc = self._centroid_count.detach().cpu().float()
                eps = 1e-5
                for k in range(int(cc.numel())):
                    if float(cc[k]) <= 0:
                        continue
                    if float(torch.norm(cv[:, k, :d_vis])) >= eps:
                        continue
                    c = max(int(cc[k].item()), 1)
                    cv[:, k, :d_vis] = sm[:, k, :d_vis] / float(c)
            cv = cv.to(self._centroid_param.dtype)
        if self._buf_centroid_param is not None:
            bc = self._buf_centroid_param.detach().cpu().float().clone()
            if self._buf_sum is not None and self._buf_count is not None:
                d_vis = self._centroid_vis_width(self._buf_centroid_param)
                sm = self._buf_sum.detach().cpu().float()
                c = max(int(self._buf_count.detach().cpu().item()), 1)
                if int(self._buf_count.detach().cpu().item()) > 0 and float(torch.norm(bc[:, :d_vis])) < 1e-5:
                    bc[:, :d_vis] = sm[:, :d_vis] / float(c)
            bc = bc.to(self._buf_centroid_param.dtype)
        csum = self._centroid_sum.detach().cpu().clone() if self._centroid_sum is not None else torch.zeros(0)
        cct = self._centroid_count.detach().cpu().clone() if self._centroid_count is not None else torch.zeros(0)
        bsum = self._buf_sum.detach().cpu().clone() if self._buf_sum is not None else torch.zeros(0)
        bct = self._buf_count.detach().cpu().clone() if self._buf_count is not None else torch.zeros(0)
        self._assert_group_lists_aligned("save_extra_state")
        blob = {
            "group_prototypes": [t.detach().cpu().float().clone() for t in self.group_prototypes],
            "group_counts": list(self.group_counts),
            "group_tasks": [list(x) for x in self.group_tasks],
            "group_experts": [list(x) for x in self.group_experts],
            "next_expert_id": self._next_expert_id,
            "centroid_vis": cv,
            "buf_centroid": bc,
            "centroid_sum": csum,
            "centroid_count": cct,
            "buf_centroid_sum": bsum,
            "buf_centroid_count": bct,
            "centroid_n": self.centroid_n.cpu().clone(),
            "expert_budget_charge": list(self.expert_budget_charge),
            "expert_budget_ranks": list(self.expert_budget_charge),
            "theta": self.theta,
            "max_groups": self.max_groups,
            "task_group_lock": dict(self._task_group_lock),

            "task_expert_lock": dict(self._task_expert_lock),
            "cram_last_centroid_reset_task": self._cram_last_centroid_reset_task,
            "cram_buf_svd_done_for_task": {int(k): bool(v) for k, v in self._cram_buf_svd_done_for_task.items()},
        }
        torch.save(blob, os.path.join(output_dir, "cram_state.bin"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        p = os.path.join(load_dir, "cram_state.bin")
        if not os.path.isfile(p):
            return False
        blob = torch.load(p, map_location="cpu")
        if not isinstance(blob, dict):
            return False
        if model is not None:
            self._ensure_centroid_tables(model)
        gp = blob.get("group_prototypes")
        self._load_group_prototypes_from_blob(gp)
        self.group_counts = list(blob.get("group_counts", []))
        self.group_tasks = [list(x) for x in blob.get("group_tasks", [])]
        self.group_experts = [list(x) for x in blob.get("group_experts", [])]
        self._assert_group_lists_aligned("load_extra_state")
        self._next_expert_id = int(blob.get("next_expert_id", 0))
        self.centroid_n = blob.get("centroid_n", torch.zeros(0)).long().clone()
        ebc = blob.get("expert_budget_charge")
        if ebc is None:
            ebc = blob.get("expert_budget_ranks")
        if isinstance(ebc, list):
            self.expert_budget_charge = [int(x) for x in ebc][: self.max_slots]
            while len(self.expert_budget_charge) < self.max_slots:
                self.expert_budget_charge.append(0)
        else:
            self.expert_budget_charge = [0] * int(self.max_slots)

        self._task_group_lock = {int(k): int(v) for k, v in blob.get("task_group_lock", {}).items()}

        for gi, tasks in enumerate(self.group_tasks):
            for t in tasks:
                tid = int(t)
                if tid not in self._task_group_lock:
                    self._task_group_lock[tid] = int(gi)
        self._task_expert_lock = {int(k): int(v) for k, v in blob.get("task_expert_lock", {}).items()}
        self._validate_level1_state_after_load()
        if "cram_last_centroid_reset_task" in blob:
            lr = blob.get("cram_last_centroid_reset_task")
            self._cram_last_centroid_reset_task = int(lr) if lr is not None else None
        bsd = blob.get("cram_buf_svd_done_for_task")
        if isinstance(bsd, dict):
            self._cram_buf_svd_done_for_task = {int(k): bool(v) for k, v in bsd.items()}
        else:
            self._cram_buf_svd_done_for_task = {}
        if self._centroid_param is not None and blob.get("centroid_vis") is not None:
            t = blob["centroid_vis"].to(self._centroid_param.device, dtype=self._centroid_param.dtype)
            if t.shape == self._centroid_param.shape:
                self._centroid_param.data.copy_(t)
                self._maybe_refresh_deepspeed_fp32_after_centroid_data_write()
        if self._buf_centroid_param is not None and blob.get("buf_centroid") is not None:
            t = blob["buf_centroid"].to(self._buf_centroid_param.device, dtype=self._buf_centroid_param.dtype)
            if t.shape == self._buf_centroid_param.shape:
                self._buf_centroid_param.data.copy_(t)
        if self._centroid_sum is not None and blob.get("centroid_sum") is not None:
            ts = blob["centroid_sum"].to(self._centroid_sum.device, dtype=self._centroid_sum.dtype)
            if ts.shape == self._centroid_sum.shape:
                self._centroid_sum.copy_(ts)
        if self._centroid_count is not None and blob.get("centroid_count") is not None:
            tc = blob["centroid_count"].to(self._centroid_count.device, dtype=self._centroid_count.dtype)
            if tc.shape == self._centroid_count.shape:
                self._centroid_count.copy_(tc)
        if self._buf_sum is not None and blob.get("buf_centroid_sum") is not None:
            ts = blob["buf_centroid_sum"].to(self._buf_sum.device, dtype=self._buf_sum.dtype)
            if ts.shape == self._buf_sum.shape:
                self._buf_sum.copy_(ts)
        if self._buf_count is not None and blob.get("buf_centroid_count") is not None:
            tc = blob["buf_centroid_count"].to(self._buf_count.device, dtype=self._buf_count.dtype)
            if tc.shape == self._buf_count.shape:
                self._buf_count.copy_(tc)

        if model is not None:
            self._repair_expert_centroid_vis_columns_from_sum_if_degenerate()
            self._repair_buf_centroid_from_sum_if_degenerate()
            root_ld = self._unwrap_training_model(model)
            bm_ld = getattr(root_ld, "_base_model", None) or root_ld
            if self._centroid_param is not None:
                self._centroid_param.requires_grad_(True)
            if bm_ld is not None and not getattr(bm_ld, "_cram_centroid_vis_grad_hook_registered", False):
                self._register_centroid_vis_grad_freeze_hook(bm_ld)

        if model is not None:
            from PEFT.tuners.custom.cram_lora import repair_all_cram_expert_mask_and_r_from_weights

            repair_all_cram_expert_mask_and_r_from_weights(model)
        if model is not None:
            self._sync_r_buf_all_layers(model)
        if model is not None and bool(getattr(model, "training", False)):
            ct = int(getattr(self.config, "cur_task", 0))
            if self._cram_buf_svd_done_for_task.get(ct, False):
                slot = int(self._task_expert_lock.get(ct, self._next_expert_id - 1))
                self._freeze_buf_train_expert_slot_only(model, slot)
        return True

    def get_inference_config(self) -> Dict:
        return {"expert_num": self.max_slots, "clip_feature_dim": self.feature_dim}
