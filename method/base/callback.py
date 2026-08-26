# method/base/callback.py
from __future__ import annotations

import inspect
from typing import Any, Optional

from transformers import TrainerCallback


def _unwrap_trainer_model_for_cl(model: Optional[Any]) -> Optional[Any]:
    """DeepSpeed / DDP wrap the user model in ``.module``; Trainer passes the wrapper to callbacks."""
    if model is None:
        return None
    seen: set[int] = set()
    cur = model
    for _ in range(16):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "cl_context"):
            return cur
        inner = getattr(cur, "module", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return model


class CLTrainerCallback(TrainerCallback):
    """CL hooks on the training loop; no-op when the model is not a ``CLModel``."""

    def __init__(self, model_args, model):
        self.model_args = model_args
        self.model = model
        unwrapped = _unwrap_trainer_model_for_cl(model)
        self.integration = getattr(unwrapped, "integration", None) if unwrapped is not None else None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """After each step: optional prototype / state updates in ``on_step_end``."""
        if self.integration is None:
            return
        m = _unwrap_trainer_model_for_cl(kwargs.get("model") or model or self.model)
        if m is None or not hasattr(m, "cl_context"):
            return
        hook = self.integration.on_step_end
        if "global_step" in inspect.signature(hook).parameters:
            hook(
                m,
                m.cl_context,
                kwargs.get("loss"),
                global_step=getattr(state, "global_step", None),
            )
        else:
            hook(m, m.cl_context, kwargs.get("loss"))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Notify integrations before the first optimizer step so warmup ratios can be resolved."""
        if self.integration is None:
            return
        m = _unwrap_trainer_model_for_cl(kwargs.get("model") or model or self.model)
        if m is None or not hasattr(m, "cl_context"):
            return
        hook = getattr(self.integration, "on_train_begin", None)
        if callable(hook):
            hook(
                m,
                getattr(state, "global_step", None),
                trainer=kwargs.get("trainer") or getattr(self, "trainer", None),
            )

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """End of training: optional ``on_train_task_finished`` then ``on_task_end``."""
        if self.integration is None:
            return
        m = _unwrap_trainer_model_for_cl(kwargs.get("model") or model or self.model)
        if m is None or not hasattr(m, "cl_context"):
            return
        task_id = getattr(self.model_args, "cur_task", 0)
        hook = getattr(self.integration, "on_train_task_finished", None)
        if callable(hook):
            hook(
                m,
                m.cl_context,
                task_id,
                trainer=getattr(self, "trainer", None),
            )
        self.integration.on_task_end(m, m.cl_context, task_id)
