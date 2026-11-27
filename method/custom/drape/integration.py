"""Drape: task-specific cross-modal prompt generators and a shared NSP projector.

Uses Prism's CL lifecycle, benchmark order, and checkpoint directories. The
ported generator/router live alongside the other PEFT tuners.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


def ensure_peft_extension_registered():
    from method.base.peft_extension import register_peft_extension
    from PEFT.tuners.custom.drape import DrapeConfig, DrapeModel, DrapePeftModel
    register_peft_extension(
        "DRAPETUNING", DrapeConfig, DrapeModel,
        task_type="CAUSAL_LM_DRAPE", task_peft_model_cls=DrapePeftModel,
    )


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _main():
    return not dist.is_initialized() or dist.get_rank() == 0


def _task_name(config):
    benchmark = str(getattr(config, "benchmark", "")).lower()
    if benchmark in ("coin", "ucit"):
        module_name, attr = {"coin": ("CoIN", "COIN_TASKS"), "ucit": ("UCIT", "UCIT_TASKS")}[benchmark]
        module = __import__(f"config.benchmarks.{module_name}", fromlist=[attr])
        tasks = getattr(module, attr)
        return tasks[int(getattr(config, "cur_task", 0))]["name"]
    return f"task_{int(getattr(config, 'cur_task', 0))}"


@CLMethodFactory.register("drape")
class DrapeIntegration(CLIntegration):
    def __init__(self, config):
        super().__init__(config)
        self._model_ref = None
        self._finalized = False
        self._loaded = False

    def initialize_model(self, model):
        from PEFT import get_peft_model
        from PEFT.tuners.custom.drape import DrapeConfig
        from config.paths.llava_paths import CLIP_PATH

        base = model._base_model
        if (getattr(self.config, "backbone", "llava") != "llava"
                or getattr(base.config, "model_type", None) != "llava"):
            raise ValueError("DRAPE currently supports --backbone llava")
        ensure_peft_extension_registered()
        hidden = int(base.config.hidden_size)
        values = {
            field.name: getattr(self.config, field.name)
            for field in fields(DrapeConfig) if hasattr(self.config, field.name)
        }
        values.update(
            text_input_dim=hidden, vision_input_dim=hidden, token_dim=hidden,
            current_task_name=_task_name(self.config), previous_task_model_dir=None,
            clip_model_name=getattr(self.config, "clip_model_name", CLIP_PATH),
            base_model_name_or_path=getattr(self.config, "model_name_or_path", None),
        )
        wrapped = get_peft_model(base, DrapeConfig(**values))
        wrapped._drape_method_config = self.config
        wrapped._drape_loaded_adapter = None
        # The tuner freezes the backbone; also train the shared projector.
        base.get_model().mm_projector.requires_grad_(True)
        object.__setattr__(model, "_base_model", wrapped)
        model._modules["_base_model"] = wrapped
        self._model_ref = wrapped.base_model

    def _routing_texts(self, model, input_ids, attention_mask=None, labels=None):
        return model._base_model.routing_texts(input_ids, attention_mask, labels)

    def on_input_prep(self, model, args, kwargs, context):
        pass

    def on_forward_start(self, model, context):
        pass

    def on_forward_end(self, model, outputs, context):
        return outputs

    def prepare_training_data(self, data_args, model_args, training_args=None):
        if training_args is not None:
            strategy = getattr(training_args, 'save_strategy', 'no')
            strategy = getattr(strategy, 'value', strategy)
            if strategy != 'no':
                raise ValueError(
                    'Drape fits routing keys at task end. Use --save_strategy no '
                    '(the method default); Prism still saves the final task checkpoint.'
                )

    def on_task_end(self, model, context, task_id):
        if self._finalized:
            return
        drape = model._base_model.base_model
        embeddings = drape.pop_cached_fused_embeddings(drape.current_task_name)
        if _distributed():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, embeddings)
            embeddings = torch.cat([item for item in gathered if item is not None]) if any(
                item is not None for item in gathered
            ) else None
            # Sum only this task's observations. Historical statistics are merged
            # exactly once by compute_and_save_null_space below.
            device = next(drape.model.parameters()).device
            for name, collector in drape._feature_collectors.items():
                layer = drape.model.get_model().mm_projector.get_submodule(name[len("mm_projector."):])
                cov = collector.cov_matrix
                if cov is None:
                    cov = torch.zeros(layer.in_features, layer.in_features, device=device)
                count = torch.tensor(collector.num_samples, device=cov.device, dtype=torch.long)
                dist.all_reduce(cov)
                dist.all_reduce(count)
                collector.cov_matrix, collector.num_samples = cov, int(count.item())
        if embeddings is None or embeddings.numel() == 0:
            raise RuntimeError("No Drape routing features collected; refusing to save an unroutable task")
        if _main():
            # Small key learning and SVD can run on CPU on CUDA and Ascend servers.
            drape.update_task_center_from_embeddings(drape.current_task_name, embeddings, device="cpu")
            drape.clip_router.to(next(drape.model.parameters()).device)
            drape.compute_and_save_null_space(None)
        if _distributed():
            payload = [({
                "centers": {key: value.detach().cpu() for key, value in drape.clip_router.task_centers.items()},
                "counts": drape.clip_router._task_sample_counts,
                "null_space": drape._pack_null_space_state(),
            } if _main() else None)]
            dist.broadcast_object_list(payload, src=0)
            device = next(drape.model.parameters()).device
            for name, center in payload[0]["centers"].items():
                drape.clip_router.task_centers[name] = nn.Parameter(center.to(device), requires_grad=False)
            drape.clip_router._task_sample_counts = payload[0]["counts"]
            drape._load_null_space_state(payload[0]["null_space"])
        self._finalized = True

    def load_extra_state(self, load_dir, model=None):
        if model is None:
            raise ValueError("DRAPE load_extra_state requires the CLModel")
        adapter = model._base_model
        drape = adapter.base_model
        directory = Path(load_dir).resolve()
        if getattr(adapter, "_drape_loaded_adapter", None) != str(directory):
            raise RuntimeError("Load DRAPE adapter weights before loading its continual state")
        saved = adapter._drape_saved_config
        incremental = bool(getattr(self.config, "previous_task_model_path", None))
        device = next(drape.model.parameters()).device
        if not (directory / "router/task_centers.pt").is_file():
            raise FileNotFoundError(f"Missing DRAPE router state: {directory}")
        drape.clip_router.load_state(str(directory / "router"), device=device)
        history = directory / "task_generators/generators_config.json"
        if not history.is_file():
            raise FileNotFoundError(f"Missing DRAPE generator history: {history}")
        from PEFT.tuners.custom.drape import DrapeGenerator
        drape.previous_generators = nn.ModuleDict()
        for task in json.loads(history.read_text())["task_names"]:
            if not incremental and task == saved["current_task_name"]:
                continue
            generator = DrapeGenerator(drape.active_peft_config)
            generator.load_state_dict(torch.load(directory / f"task_generators/{task}.bin", map_location="cpu"), strict=True)
            generator.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
            drape.previous_generators[task] = generator
        if drape.apply_null_space_projection:
            path = directory / "null_space_projection.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing DRAPE null-space state: {path}")
            drape._load_null_space_state(torch.load(path, map_location="cpu"))
            drape._setup_null_space_hooks()
        if incremental:
            if saved["current_task_name"] == drape.current_task_name:
                raise ValueError("Previous-task checkpoint names the current task; check benchmark task order")
        else:
            drape.current_task_name = saved["current_task_name"]
            drape.active_peft_config.current_task_name = drape.current_task_name
        adapter.train(incremental)
        self._loaded = True
        return True

    def save_extra_state(self, output_dir, model=None):
        drape = model._base_model.base_model if model is not None else self._model_ref
        if drape is None:
            raise RuntimeError("Initialize the DRAPE model before saving its continual state")
        drape._save_continual_learning_state(output_dir)
        return True

    def get_inference_config(self):
        return {"batch_size": 1}
