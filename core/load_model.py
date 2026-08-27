import torch
import transformers
from typing import Any, Tuple, Optional
import os
import sys
import importlib
from backbone.shared.model_loading import (
    setup_quantization,
    load_pretrained_model,
    load_tokenizer,
    setup_tokenizer,
    load_clip_tokenizer,
    initialize_multimodal_modules,
    apply_mm_projector_trainability,
    adjust_precision,
    prepare_model_for_kbit,
    setup_gradient_checkpointing,
    quiet_hf_checkpoint_load_messages,
)
from .config_loader import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    merge_benchmark_task_num_into,
    merge_method_config_into,
)
from config.backbone.registry import (
    get_backbone_display_name,
    get_checkpoint_suffix,
    get_load_vision_tower_separately,
    get_routing_feature_dim,
    import_mllm_model_class,
    infer_precision_flags,
    is_multimodal_model_name,
    resolve_backbone_id,
    resolve_infer_precision,
)
from utils.infer_paths import (
    resolve_method_infer_tower_paths,
    should_load_routing_vision_tower,
)
from config.backbone.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import warnings

from utils.rank import is_local_main_process


def _train_log(*args, **kwargs) -> None:
    """Log only on local main process (LOCAL_RANK 0 or unset)."""
    if is_local_main_process():
        print(*args, **kwargs)


def _resolve_train_cuda_device(training_args) -> torch.device:
    """Map this process to ``cuda:{local_rank}``; single-process launch uses ``cuda``."""
    rank = getattr(training_args, "local_rank", None)
    if rank is None:
        env = os.environ.get("LOCAL_RANK", "-1")
        try:
            rank = int(env)
        except ValueError:
            rank = -1
    else:
        rank = int(rank)
    if rank < 0:
        return torch.device("cuda")
    return torch.device(f"cuda:{rank}")


def _move_model_to_train_device(model: Any, training_args) -> Any:
    """Place the full model on this rank's GPU (avoid 4 ranks stacking on physical GPU0)."""
    device = _resolve_train_cuda_device(training_args)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _train_log(
        f"Moving model to {device} (training_args.local_rank={getattr(training_args, 'local_rank', None)})"
    )
    return model.to(device)


def _cli_sets_training_flag(flag: str) -> bool:
    """True if child ``sys.argv`` contains ``--flag`` or ``--flag=...`` (preserve CLI overrides)."""
    prefix = f"{flag}="
    for a in sys.argv:
        if a == flag or a.startswith(prefix):
            return True
    return False


# Deferred imports for CLModel / CLIntegration (avoid circular imports)
def _try_import_cl_components():
    """Import CLModel / CLIntegration lazily."""
    try:
        from method.base.cl_model import CLModel
        from method.base.integration import CLIntegration
        return CLModel, CLIntegration
    except ImportError as e:
        _train_log(f"CL component import failed: {e}")
        return None, None


def _maybe_repair_cram_expert_metadata(lora_target) -> None:
    """Restore CRAM expert_mask / expert_r from A/B weights after adapter load.

    Original CRAM (grac) did this in ``load_from_checkpoint``: PEFT may omit
    ``expert_mask`` / ``expert_r`` buffers, and the forward path then skips every
    expert branch (inference looks like zeroshot).
    """
    try:
        from PEFT.tuners.custom.cram_lora import repair_all_cram_expert_mask_and_r_from_weights

        repair_all_cram_expert_mask_and_r_from_weights(lora_target)
    except Exception as e:
        _train_log(f"    [CRAM] expert_mask/expert_r restore skipped: {e}")


def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
    """Load weights from a checkpoint directory."""
    import json

    _train_log(f"Loading checkpoint from {checkpoint_path}...")
    _train_log(f"  for_incremental_training: {for_incremental_training}")

    # Locate PEFT target
    lora_target = model
    if hasattr(model, '_base_model'):
        _train_log("  Detected CLModel wrapper")
        lora_target = model._base_model

    _train_log(f"  LoRA target model type: {type(lora_target)}")
    _train_log(f"  has load_adapter: {hasattr(lora_target, 'load_adapter')}")

    # non-LoRA weights
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        _train_log("  Loading non-LoRA weights...")
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')

        target = lora_target
        if hasattr(target, 'base_model'):
            target = target.base_model
        if hasattr(target, 'model'):
            target = target.model

        target.load_state_dict(non_lora_weights, strict=False)
        _train_log("    non-LoRA weights loaded")

    # LoRA weights
    if for_incremental_training:
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                _train_log("\n  Loading LoRA adapter (incremental training mode)...")
                _train_log(f"      checkpoint_path: {checkpoint_path}")
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                _train_log("    LoRA adapter loaded")
                _maybe_repair_cram_expert_metadata(lora_target)
        else:
            _train_log("  Model has no load_adapter method")

    # Inference: load_adapter
    else:
        _train_log("\n  Inference mode: trying load_adapter...")
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                _train_log(f"      load_adapter({checkpoint_path}, adapter_name='default')")

                config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    _train_log(f"      adapter_config peft_type: {config.get('peft_type', 'unknown')}")

                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                _train_log("    load_adapter finished")
                _maybe_repair_cram_expert_metadata(lora_target)
            else:
                _train_log("    Checkpoint is not a directory, falling back to manual LoRA load...")
                _manual_load_lora(lora_target, checkpoint_path)
        else:
            _train_log("    Model has no load_adapter, loading LoRA manually...")
            _manual_load_lora(lora_target, checkpoint_path)

    # Optional expert LoRA norm diagnostics
    expert_norms = {}
    for name, param in lora_target.named_parameters():
        if 'lora' in name.lower():
            import re
            match = re.search(r'loraA\.(\d+)', name)
            if match:
                expert_id = match.group(1)
                if expert_id not in expert_norms:
                    expert_norms[expert_id] = []
                expert_norms[expert_id].append(param.norm().item())

    for expert_id in sorted(expert_norms.keys())[:4]:
        if expert_norms[expert_id]:
            avg_norm = sum(expert_norms[expert_id]) / len(expert_norms[expert_id])
            _train_log(f"    Expert {expert_id}: avg_norm={avg_norm:.6f} (samples={len(expert_norms[expert_id])})")

    if '0' in expert_norms and '1' in expert_norms:
        avg0 = sum(expert_norms['0']) / len(expert_norms['0'])
        avg1 = sum(expert_norms['1']) / len(expert_norms['1'])
        if abs(avg0 - avg1) < 1e-6:
            _train_log("    Warning: Expert 0 and Expert 1 weights appear identical")
        else:
            _train_log(f"    Expert 0 vs 1 avg norm: {avg0:.6f} vs {avg1:.6f}")

    _train_log("\nCheckpoint load finished")
    return model


def _manual_load_lora(lora_target, checkpoint_path):
    """Load LoRA weights without HuggingFace ``load_adapter``."""
    lora_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')

    if os.path.exists(lora_path):
        _train_log(f"      Manual load: {lora_path}")

        if lora_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            lora_weights = load_file(lora_path)
        else:
            lora_weights = torch.load(lora_path, map_location='cpu')

        model_keys = set(lora_target.state_dict().keys())

        remapped_weights = {}
        for key, value in lora_weights.items():
            new_key = key
            if '.lora_A.loraA' in key and '.default.' not in key:
                new_key = key.replace('.lora_A.loraA', '.lora_A.default.loraA')
            elif '.lora_B.loraB' in key and '.default.' not in key:
                new_key = key.replace('.lora_B.loraB', '.lora_B.default.loraB')
            remapped_weights[new_key] = value

        matched = set(remapped_weights.keys()) & model_keys
        _train_log(f"      Keys matched after remap: {len(matched)}/{len(remapped_weights)}")

        if len(matched) > 0:
            missing, unexpected = lora_target.load_state_dict(remapped_weights, strict=False)
            _train_log("      Manual LoRA load finished")
            _maybe_repair_cram_expert_metadata(lora_target)
        else:
            _train_log("      Failed to map LoRA weights to model keys")
    else:
        _train_log("      LoRA weight file not found")


# common/load_model.py
def load_model_for_train(model_args, data_args, training_args):
    """Load model for continual-learning training."""
    if str(getattr(model_args, "method", "") or "").strip().lower() == "zeroshot":
        raise ValueError(
            "method='zeroshot' is inference-only (no continual learning training). "
            "Use eval/infer with --method zeroshot, or set method to a CL method for train."
        )
    merge_method_config_into(model_args)
    merge_benchmark_task_num_into(model_args)
    # Copy METHOD_CONFIG lora_* into training_args unless CLI overrides.
    if getattr(training_args, "lora_enable", False):
        _lora_flag = {
            "lora_r": "--lora_r",
            "lora_alpha": "--lora_alpha",
            "lora_dropout": "--lora_dropout",
        }
        for name, flag in _lora_flag.items():
            if _cli_sets_training_flag(flag):
                continue
            if hasattr(model_args, name):
                v = getattr(model_args, name)
                if v is not None:
                    setattr(training_args, name, v)
        model_args.lora_r = training_args.lora_r
        model_args.lora_alpha = training_args.lora_alpha
        model_args.lora_dropout = training_args.lora_dropout
    _m = str(getattr(model_args, "method", "") or "").lower()
    if _m not in ("", "base", "none", "zeroshot") and getattr(model_args, "task_num", None) is None:
        raise ValueError(
            "Continual learning training requires task_num: pass --benchmark (ucit / coin) "
            "or --task_num explicitly; task counts come from config/benchmarks, not config/methods."
        )

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    bnb_args = setup_quantization(training_args, compute_dtype)

    has_vision = model_args.vision_tower is not None
    backbone_id = resolve_backbone_id(getattr(model_args, "backbone", None))
    ckpt_suffix = get_checkpoint_suffix(backbone_id)
    model_args.backbone = backbone_id
    model_args.load_vision_tower_separately = get_load_vision_tower_separately(backbone_id)
    if not model_args.load_vision_tower_separately:
        from config.backbone.registry import get_embedded_vision_image_size

        model_args.embedded_vision_image_size = get_embedded_vision_image_size(backbone_id)
    model = load_pretrained_model(
        model_args.model_name_or_path,
        training_args,
        bnb_args,
        has_vision,
        backbone_id=backbone_id,
        vision_tower=model_args.vision_tower,
        routing_vision_tower=getattr(model_args, "routing_vision_tower", None),
        text_tower=getattr(model_args, "text_tower", None),
    )

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model = prepare_model_for_kbit(model, training_args, compute_dtype)
    model = setup_gradient_checkpointing(model, training_args)

    tokenizer = load_tokenizer(model_args.model_name_or_path, training_args)
    setup_tokenizer(tokenizer, model, model_args.version)

    # Multimodal (before CL wrapper)
    if model_args.vision_tower is not None:
        _train_log("Initializing multimodal modules...")
        model, data_args = initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer)
        _train_log("Multimodal modules initialized.\n")

    model = adjust_precision(model, training_args)

    if model_args.text_tower is not None:
        clip_tokenizer = load_clip_tokenizer(model_args.text_tower, training_args)
        model.set_clip_tokenizer(clip_tokenizer)

    model.set_tokenizer(tokenizer)
    if hasattr(model, 'set_cur_task'):
        model.set_cur_task(model_args.cur_task, model_args.task_num)

    # CL wrapper + method integration
    method_name = getattr(model_args, 'method', 'base')
    
    if method_name != 'base':
        _train_log(f"\n{'='*70}")
        _train_log(f"Continual learning method: {method_name}")
        if str(method_name).lower() == "zeroshot":
            _train_log("Zeroshot: plain LLaVA, no PEFT injection / no method-side extras")
        else:
            _train_log("Method integration will handle PEFT/LoRA injection")
        _train_log(f"{'='*70}\n")
        
        CLModel, CLIntegration = _try_import_cl_components()
        if CLModel is None:
            raise ImportError("Failed to import CL components (CLModel / CLIntegration)")
        
        module = __import__(f"method.custom.{method_name}.integration", fromlist=[''])
        IntegrationClass = getattr(module, f"{method_name.capitalize()}Integration")
        integration = IntegrationClass(model_args)

        model = CLModel(model, integration)
        _train_log(f"CLModel wrapper ready | method: {method_name}\n")
        
        # Resume from previous task after PEFT is injected
        prev_path = getattr(model_args, "previous_task_model_path", None)
        if prev_path is not None:
            prev_path = os.path.expanduser(str(prev_path).strip())
            tried_paths = [prev_path]
            lora_suffix = f"{ckpt_suffix}_lora"
            if not os.path.exists(prev_path) and prev_path.endswith(lora_suffix):
                alt = prev_path[: -len("_lora")]
                tried_paths.append(alt)
                if os.path.exists(alt):
                    _train_log(
                        f"[train] previous_task_model_path {prev_path!r} not found; "
                        f"using {alt!r} ({lora_suffix} vs {ckpt_suffix} naming).\n"
                    )
                    prev_path = alt
                    model_args.previous_task_model_path = alt
            elif (
                not os.path.exists(prev_path)
                and prev_path.endswith(ckpt_suffix)
                and "_lora" not in prev_path
            ):
                alt = prev_path.replace(ckpt_suffix, lora_suffix)
                tried_paths.append(alt)
                if os.path.exists(alt):
                    _train_log(
                        f"[train] previous_task_model_path {prev_path!r} not found; "
                        f"using {alt!r} ({ckpt_suffix} vs {lora_suffix} naming).\n"
                    )
                    prev_path = alt
                    model_args.previous_task_model_path = alt
            if not os.path.exists(prev_path):
                raise FileNotFoundError(
                    "[train] previous_task_model_path not found (incremental load required). Tried:\n  "
                    + "\n  ".join(repr(p) for p in tried_paths)
                    + "\nFix the path or train the missing previous task before continuing."
                )

        if model_args.previous_task_model_path is not None and os.path.exists(
            model_args.previous_task_model_path
        ):
            _train_log(f"Loading previous-task checkpoint: {model_args.previous_task_model_path}")
            model = load_from_checkpoint(
                model,
                model_args.previous_task_model_path,
                merge_lora=False,
                for_incremental_training=True
            )
            _train_log("Checkpoint load finished.\n")

            if hasattr(model, '_integration'):
                integration = model._integration
                if hasattr(integration, 'load_extra_state'):
                    _train_log("Loading method extra state from previous checkpoint...")
                    ok = integration.load_extra_state(model_args.previous_task_model_path, model=model)
                    if ok:
                        _train_log("Method extra state restored")
                    elif str(method_name).lower() == "same":
                        raise RuntimeError(
                            "SAME incremental training requires carry-over state "
                            "(same_state.bin or prism.same.* keys in adapter_model.safetensors) "
                            f"under {model_args.previous_task_model_path}; load_extra_state returned False."
                        )
                    else:
                        _train_log("Method extra state not found or failed to load (continuing training)")
    else:
        assert(0)
    if hasattr(model, "_integration"):
        _prep = getattr(model._integration, "prepare_training_data", None)
        if callable(_prep):
            _prep(data_args, model_args, training_args)

    apply_mm_projector_trainability(model, training_args)

    # LoRA is injected inside CL integration; cast adapters after PEFT wrap.
    if getattr(training_args, "lora_enable", False) and training_args.bits in (4, 8):
        model = adjust_precision(model, training_args)

    model = _move_model_to_train_device(model, training_args)
    model.train()
    
    return model, tokenizer, data_args


def _checkpoint_dir_has_peft_adapter(model_path: str) -> bool:
    return bool(model_path) and os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


def _checkpoint_has_merged_llava_weights(model_path: str) -> bool:
    """True if directory has merged LLaVA weights (no adapter_config.json)."""
    if not model_path or not os.path.isdir(model_path):
        return False
    return os.path.isfile(os.path.join(model_path, "model.safetensors")) or os.path.isfile(
        os.path.join(model_path, "pytorch_model.bin")
    )


def _load_merged_llava_weights_into_model(model: Any, checkpoint_path: str) -> None:
    """Load ``model.safetensors`` / ``pytorch_model.bin`` into inner LLaVA (``_base_model``)."""
    inner = model._base_model if hasattr(model, "_base_model") else model
    st_path = os.path.join(checkpoint_path, "model.safetensors")
    pt_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file

        weights: dict = dict(load_file(st_path))
        src = st_path
    elif os.path.isfile(pt_path):
        weights = torch.load(pt_path, map_location="cpu")
        src = pt_path
    else:
        raise FileNotFoundError(
            f"Merged LLaVA checkpoint not found under {checkpoint_path} "
            "(expected model.safetensors or pytorch_model.bin)"
        )
    dtype = next(inner.parameters()).dtype
    weights = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in weights.items()}
    bad = inner.load_state_dict(weights, strict=False)
    miss = getattr(bad, "missing_keys", None) or []
    unexp = getattr(bad, "unexpected_keys", None) or []
    _train_log(f"  Merged LLaVA weights from {src} | missing_keys={len(miss)} unexpected_keys={len(unexp)}")


def _is_zeroshot_method(method: Optional[str]) -> bool:
    return str(method or "").strip().lower() == "zeroshot"


def _resolve_model_device(model: Any) -> torch.device:
    """Pick the CUDA device where the (possibly quantized) LLM lives."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for dev in device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def _uses_hf_device_map(model: Any) -> bool:
    dm = getattr(model, "hf_device_map", None)
    return isinstance(dm, dict) and len(dm) > 0


def _module_has_meta_params(module: Any) -> bool:
    if module is None:
        return False
    return any(p.is_meta for p in module.parameters(recurse=True))


def _warn_internvl_patch_mismatch(model: Any, *, method: Optional[str] = None) -> None:
    """Warn when separate ViT resolution differs from InternVL-Chat pretraining (336px / 576 patches)."""
    try:
        vt = model.get_vision_tower()
    except Exception:
        return
    if vt is None:
        return
    patches = getattr(vt, "num_patches", None)
    if patches is None:
        return
    from config.backbone.registry import get_embedded_vision_image_size

    embedded_px = get_embedded_vision_image_size("internvl")
    expected = (embedded_px // 14) ** 2
    if int(patches) == int(expected):
        return
    ctx = f" method={method}" if method else ""
    print(
        f"[infer] WARNING: ViT outputs {patches} patch tokens but InternVL-Chat was trained "
        f"with {expected} ({embedded_px}px). Multimodal quality will be severely degraded "
        f"(garbled answers like '.' / 'Ъ' on VQA).{ctx}\n"
        f"[infer] For zeroshot baseline, set LOAD_VISION_TOWER_SEPARATELY=False in "
        f"config/paths/internvl_paths.py (use embedded ViT from BASE_MODEL_PATH).",
        flush=True,
    )


def _move_tower_for_infer(
    module: Any,
    load_device: torch.device,
    dtype: torch.dtype,
    *,
    model: Any,
    label: str = "tower",
) -> None:
    """Move a vision/text tower; separately loaded ViT is independent of LLM ``device_map``."""
    if module is None:
        return
    if _module_has_meta_params(module):
        print(
            f"[infer] {label}: skip .to() on meta tensors (weights load on first forward).",
            flush=True,
        )
        return

    load_separately = bool(getattr(module, "load_separately", False))
    tower_sharded = _uses_hf_device_map(module)
    parent_sharded = _uses_hf_device_map(model)

    try:
        if load_separately:
            module.to(device=load_device, dtype=dtype)
            print(f"[infer] {label}: loaded separately -> {load_device} ({dtype}).", flush=True)
            return
        if tower_sharded or parent_sharded:
            print(
                f"[infer] {label}: keep Accelerate device_map placement (cast dtype={dtype}).",
                flush=True,
            )
            if dtype is not None:
                module.to(dtype=dtype)
            return
        module.to(device=load_device, dtype=dtype)
    except NotImplementedError as exc:
        if "meta tensor" in str(exc).lower():
            print(
                f"[infer] {label}: skip .to() on meta tensors (weights load on first forward).",
                flush=True,
            )
            if dtype is not None:
                module.to(dtype=dtype)
            return
        raise


def _load_mm_projector_bin(model: Any, mm_path: str) -> None:
    """Load ``mm_projector.bin`` into the inner LLaVA model (works with 4/8-bit LLM wrappers)."""
    weights = torch.load(mm_path, map_location="cpu")
    inner = model.get_model() if hasattr(model, "get_model") else model
    if not hasattr(inner, "mm_projector"):
        print(f"Warning: model has no mm_projector; skip {mm_path}", flush=True)
        return
    prefix = "model.mm_projector."
    mm_w = {
        k[len(prefix):]: v.to(torch.float16)
        for k, v in weights.items()
        if k.startswith(prefix)
    }
    if not mm_w:
        mm_w = {
            k.split("mm_projector.", 1)[1]: v.to(torch.float16)
            for k, v in weights.items()
            if "mm_projector" in k
        }
    inner.mm_projector.load_state_dict(mm_w, strict=False)


def _resolve_infer_towers(
    method: Optional[str],
    *,
    vision_tower: Optional[str] = None,
    routing_vision_tower: Optional[str] = None,
    text_tower: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Merge method INFER_DEFAULTS with CLI; skip routing tower when path equals MLLM vision tower."""
    defaults = resolve_method_infer_tower_paths(method)
    routing = routing_vision_tower if routing_vision_tower is not None else defaults.get("routing_vision_tower")
    text = text_tower if text_tower is not None else defaults.get("text_tower")
    if routing and not should_load_routing_vision_tower(routing, vision_tower):
        routing = None
    return routing, text


def _tower_compute_dtype(load_8bit: bool, load_4bit: bool, llm_dtype: torch.dtype) -> torch.dtype:
    if load_8bit or load_4bit:
        return torch.float16
    return llm_dtype


def _apply_infer_tower_paths_to_config(
    cfg,
    *,
    vision_tower: Optional[str] = None,
    routing_vision_tower: Optional[str] = None,
    text_tower: Optional[str] = None,
    backbone_id: Optional[str] = None,
    method: Optional[str] = None,
) -> None:
    """Override Hub ids in a saved config with local paths from CLI (must run before model __init__)."""
    from backbone.shared.model_loading import _apply_local_tower_paths_to_config

    routing_vision_tower, text_tower = _resolve_infer_towers(
        method,
        vision_tower=vision_tower,
        routing_vision_tower=routing_vision_tower,
        text_tower=text_tower,
    )
    _apply_local_tower_paths_to_config(
        cfg,
        vision_tower=vision_tower,
        routing_vision_tower=routing_vision_tower,
        text_tower=text_tower,
    )
    load_sep = get_load_vision_tower_separately(backbone_id)
    setattr(cfg, "load_vision_tower_separately", load_sep)
    if not getattr(cfg, "load_vision_tower_separately", True):
        from config.backbone.registry import get_embedded_vision_image_size

        setattr(cfg, "embedded_vision_image_size", get_embedded_vision_image_size(backbone_id))


# common/load_model.py
def load_model_for_inference(
    model_path,
    model_base,
    model_name,
    load_8bit: bool | None = None,
    load_4bit: bool | None = None,
    device_map="auto",
    device="cuda",
    text_tower=None,
    vision_tower=None,
    routing_vision_tower=None,
    backbone: Optional[str] = None,
    method: Optional[str] = None,
    benchmark: Optional[str] = None,
    task_num: Optional[int] = None,
    **kwargs
):
    """Load model for inference."""
    backbone_id = resolve_backbone_id(backbone)
    bb_label = get_backbone_display_name(backbone_id)
    load_8bit, load_4bit, llm_dtype = infer_precision_flags(load_8bit=load_8bit, load_4bit=load_4bit)
    tower_dtype = _tower_compute_dtype(load_8bit, load_4bit, llm_dtype)
    routing_vision_tower, text_tower = _resolve_infer_towers(
        method,
        vision_tower=vision_tower,
        routing_vision_tower=routing_vision_tower,
        text_tower=text_tower,
    )
    if load_8bit or load_4bit:
        print(
            f"[infer] LLM quantization: {'4-bit' if load_4bit else '8-bit'} "
            f"(override config/run_config.INFER_PRECISION={resolve_infer_precision()!r}).",
            flush=True,
        )
    else:
        print(
            f"[infer] precision={resolve_infer_precision()} "
            f"(torch_dtype={llm_dtype}, config/run_config.INFER_PRECISION).",
            flush=True,
        )
    if backbone_id == "internvl" and load_4bit:
        print(
            "[infer] WARNING: InternVL with 4-bit LLM often yields near-zero VQA accuracy "
            "(multimodal quality collapses). Prefer INFER_PRECISION='bf16' or '8bit' in "
            "config/run_config.py for zeroshot / eval.",
            flush=True,
        )

    ModelClass = import_mllm_model_class(backbone_id)
    routing_dim = get_routing_feature_dim(backbone_id)
    kwargs = {"device_map": device_map, **kwargs}
    kwargs.pop("task_num", None)
    kwargs.pop("expert_num", None)
    kwargs.pop("benchmark", None)

    if device != "cuda":
        kwargs["device_map"] = {"": device}
    elif load_8bit or load_4bit:
        kwargs["device_map"] = "auto"
    elif kwargs.get("device_map") == "auto" and resolve_backbone_id(backbone_id) != "internvl":
        # InternVL bf16/fp16: keep device_map="auto" so LLM+ViT can shard across GPU memory.
        kwargs["device_map"] = None

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = llm_dtype

    if is_multimodal_model_name(model_name, backbone_id):
        _zs = str(method or "").strip().lower() == "zeroshot"
        if _zs:
            if not model_base:
                raise ValueError(
                    f"method=zeroshot requires model_base ({bb_label} weight directory); no CL checkpoint/adapter."
                )
            model_path = os.path.expanduser(str(model_base).strip())
            print("Inference: zeroshot — using model_base weights only (no CL checkpoint).", flush=True)

        use_peft_adapter_layout = model_base is not None and (
            _checkpoint_dir_has_peft_adapter(model_path)
            or "lora" in model_name.lower()
            or (
                str(method).lower() == "hide_llava"
                and os.path.isdir(model_path or "")
                and os.path.isfile(os.path.join(model_path, "hide_state.pt"))
                and _checkpoint_has_merged_llava_weights(model_path)
            )
        )

        if use_peft_adapter_layout:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            _apply_infer_tower_paths_to_config(
                lora_cfg_pretrained,
                vision_tower=vision_tower,
                routing_vision_tower=routing_vision_tower,
                text_tower=text_tower,
                backbone_id=backbone_id,
                method=method,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print(f"Loading {bb_label} from base model...", flush=True)
            with quiet_hf_checkpoint_load_messages(context=model_base):
                model = ModelClass.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )

            if text_tower:
                clip_tokenizer = AutoTokenizer.from_pretrained(
                    text_tower,
                    cache_dir=None,
                    model_max_length=77,
                    padding_side="right",
                    use_fast=True,
                )
                model.set_clip_tokenizer(clip_tokenizer)
            model.set_tokenizer(tokenizer)
            
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if method is not None and method != 'base':
                print(f"Inference: applying continual learning method {method}...")
                CLModel, CLIntegration = _try_import_cl_components()
                if CLModel is not None:
                    module = __import__(f"method.custom.{method}.integration", fromlist=[''])
                    if hasattr(module, "ensure_peft_extension_registered"):
                        try:
                            module.ensure_peft_extension_registered()
                        except Exception as e:
                            print(f"PEFT extension registration failed for method {method}: {e}")
                    IntegrationClass = getattr(module, f"{method.capitalize()}Integration")
                    
                    class SimpleArgs:
                        def __init__(self, **kw):
                            for k, v in kw.items():
                                setattr(self, k, v)

                    pseudo_fields = dict(
                        method=method,
                        cur_task=kwargs.get("cur_task", 0),
                        task_num=task_num,
                        benchmark=benchmark,
                        clip_feature_dim=kwargs.get("clip_feature_dim", routing_dim),
                    )
                    pseudo_args = SimpleArgs(**pseudo_fields)
                    merge_method_config_into(
                        pseudo_args,
                        method=method,
                        benchmark=benchmark,
                        backbone=backbone_id,
                    )
                    merge_benchmark_task_num_into(pseudo_args, benchmark=benchmark)
                    _is_zeroshot = str(method or "").strip().lower() == "zeroshot"
                    if getattr(pseudo_args, "task_num", None) is None and not _is_zeroshot:
                        raise ValueError(
                            "CL inference requires task_num: pass benchmark (e.g. run.py infer sets --benchmark) "
                            "or set task_num explicitly in load_model_for_inference(..., task_num=..., benchmark=...)."
                        )
                    
                    integration = IntegrationClass(pseudo_args)
                    
                    model = CLModel(model, integration)
                    print(f"Inference model wrapped with CLModel | method: {method}")
                    
                    if os.path.exists(model_path):
                        print("Loading weights from checkpoint...")
                        if _checkpoint_dir_has_peft_adapter(model_path):
                            model = load_from_checkpoint(
                                model,
                                model_path,
                                merge_lora=False,
                                for_incremental_training=False,
                            )
                        elif _checkpoint_has_merged_llava_weights(model_path):
                            _load_merged_llava_weights_into_model(model, model_path)
                        else:
                            model = load_from_checkpoint(
                                model,
                                model_path,
                                merge_lora=False,
                                for_incremental_training=False,
                            )
                        print("Checkpoint load finished")
                    
                    if hasattr(integration, 'load_extra_state'):
                        print("Loading extra state...")
                        success = integration.load_extra_state(model_path, model=model)
                        if success:
                            print(f"Extra state loaded | model has image_anchors: {hasattr(model, 'image_anchors')}")
            else:
                assert(0)
                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                    if any(k.startswith('model.model.') for k in non_lora_trainables):
                        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                    model.load_state_dict(non_lora_trainables, strict=False)

                from PEFT import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
        
        elif model_base is not None:
            print(f"Loading {bb_label} from base model...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            _apply_infer_tower_paths_to_config(
                cfg_pretrained,
                vision_tower=vision_tower,
                routing_vision_tower=routing_vision_tower,
                text_tower=text_tower,
                backbone_id=backbone_id,
                method=method,
            )
            with quiet_hf_checkpoint_load_messages(context=model_base):
                model = ModelClass.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            mm_path = os.path.join(model_path, "mm_projector.bin")
            if os.path.isfile(mm_path) and not _is_zeroshot_method(method):
                _load_mm_projector_bin(model, mm_path)
            elif _is_zeroshot_method(method) and os.path.isfile(mm_path):
                print(
                    "[infer] zeroshot: mm_projector weights already in base checkpoint; "
                    "skip mm_projector.bin reload.",
                    flush=True,
                )
            elif _checkpoint_has_merged_llava_weights(model_path):
                _load_merged_llava_weights_into_model(model, model_path)
            else:
                print(
                    f"Warning: no mm_projector.bin under {model_path}; "
                    "multimodal projector may match base only."
                )
            model.set_tokenizer(tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            with quiet_hf_checkpoint_load_messages(context=model_path):
                model = ModelClass.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            model.set_tokenizer(tokenizer)
        
        if vision_tower:
            model.config.mm_vision_tower = vision_tower
        if routing_vision_tower:
            model.config.routing_vision_tower = routing_vision_tower
        if text_tower:
            setattr(model.config, "mm_text_tower", text_tower)
            if not hasattr(model.config, "mm_text_select_layer"):
                setattr(model.config, "mm_text_select_layer", -1)

        vision_tower_model = model.get_vision_tower()
        load_vit_separately = getattr(model.config, "load_vision_tower_separately", True)
        if vision_tower_model is not None and not vision_tower_model.is_loaded:
            if load_vit_separately:
                vt_path = getattr(model.config, "mm_vision_tower", vision_tower or "")
                print(f"[infer] Loading MLLM vision tower from {vt_path}...", flush=True)
                vision_tower_model.load_model()
            else:
                vision_tower_model.mark_loaded_from_checkpoint()

        load_device = (
            _resolve_model_device(model)
            if device == "cuda" and (load_8bit or load_4bit or kwargs.get("device_map"))
            else torch.device(device)
        )

        if routing_vision_tower:
            class _RoutingArgs:
                pass

            ra = _RoutingArgs()
            ra.routing_vision_tower = routing_vision_tower
            model.get_model().initialize_routing_vision_modules(ra)
            rvt = model.get_model().get_routing_vision_tower()
            if rvt is not None:
                _move_tower_for_infer(
                    rvt, load_device, tower_dtype, model=model, label="routing_vision_tower"
                )

        text_tower_model = model.get_text_tower()
        if text_tower_model is not None and text_tower:
            if not getattr(text_tower_model, "is_loaded", True):
                text_tower_model.load_model()
            _move_tower_for_infer(
                text_tower_model, load_device, tower_dtype, model=model, label="text_tower"
            )

        if device == "cuda" and not load_8bit and not load_4bit and kwargs.get("device_map") is None:
            model.to(device)
        elif load_8bit or load_4bit:
            quant = "4-bit" if load_4bit else "8-bit"
            print(f"[infer] LLM loaded in {quant} (device_map={kwargs.get('device_map')}).", flush=True)
        elif kwargs.get("device_map"):
            print(
                f"[infer] LLM loaded with device_map={kwargs.get('device_map')!r} "
                f"(dtype={kwargs.get('torch_dtype')}).",
                flush=True,
            )

        if vision_tower_model is not None:
            _move_tower_for_infer(
                vision_tower_model,
                load_device,
                tower_dtype,
                model=model,
                label="vision_tower",
            )
            image_processor = vision_tower_model.image_processor
            if resolve_backbone_id(backbone_id) == "internvl":
                _warn_internvl_patch_mismatch(model, method=method)
        else:
            image_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        warnings.warn(f'The model is not a supported multimodal backbone ({backbone_id})')
        assert(0)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if isinstance(model, torch.nn.Module):
        model.eval()

    return tokenizer, model, image_processor, context_len