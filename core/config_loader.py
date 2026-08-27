import transformers
from dataclasses import dataclass, field
from typing import Any, Optional


def merge_method_config_into(
    obj: Any,
    method: Optional[str] = None,
    benchmark: Optional[str] = None,
    backbone: Optional[str] = None,
) -> None:
    """
    Merge ``METHOD_CONFIG`` from ``config/methods/<method>.py`` into ``obj``.

    Overlay order (later wins): ``METHOD_CONFIG`` → ``METHOD_CONFIG_BY_BACKBONE[backbone]``
    → ``METHOD_CONFIG_BY_BENCHMARK[benchmark]``.

    Sets only missing attributes or attributes that are None (CLI / script wins).
    ``task_num`` / ``expert_num`` come from benchmarks via ``merge_benchmark_task_num_into``, not here.

    ``benchmark`` / ``backbone`` default to ``getattr(obj, ...)`` when omitted.
    """
    m = (method or getattr(obj, "method", None) or "").strip().lower()
    if not m or m == "base" or m == "zeroshot":
        return
    try:
        mod = __import__(f"config.methods.{m}", fromlist=["METHOD_CONFIG"])
        mc = getattr(mod, "METHOD_CONFIG", None)
        if not isinstance(mc, dict):
            return
        mc = dict(mc)
        bb_raw = backbone or getattr(obj, "backbone", None)
        if bb_raw:
            try:
                from config.backbone.registry import resolve_backbone_id

                bb = resolve_backbone_id(str(bb_raw))
            except Exception:
                bb = str(bb_raw).strip().lower()
        else:
            bb = ""
        by_bb = getattr(mod, "METHOD_CONFIG_BY_BACKBONE", None)
        if bb and isinstance(by_bb, dict):
            patch_bb = by_bb.get(bb)
            if isinstance(patch_bb, dict):
                mc = {**mc, **patch_bb}
        bm = (benchmark or getattr(obj, "benchmark", None) or "").strip().lower()
        by_bm = getattr(mod, "METHOD_CONFIG_BY_BENCHMARK", None)
        if bm and isinstance(by_bm, dict):
            patch = by_bm.get(bm)
            if isinstance(patch, dict):
                mc = {**mc, **patch}
        for key, val in mc.items():
            if key in ("expert_num", "task_num"):
                continue
            if not hasattr(obj, key):
                setattr(obj, key, val)
                continue
            if getattr(obj, key) is None:
                setattr(obj, key, val)
    except Exception:
        return


def merge_benchmark_task_num_into(obj: Any, benchmark: Optional[str] = None) -> None:
    """
    If ``task_num`` is None, set it from ``config.benchmarks.BENCHMARK_TASK_NUM[benchmark]``.
    Explicit ``--task_num`` is never overwritten.
    """
    if getattr(obj, "task_num", None) is not None:
        return
    bm = (benchmark or getattr(obj, "benchmark", None) or "").strip().lower()
    if not bm:
        return
    try:
        from config.benchmarks import BENCHMARK_TASK_NUM  # type: ignore
    except Exception:
        return
    n = BENCHMARK_TASK_NUM.get(bm)
    if n is None:
        return
    setattr(obj, "task_num", int(n))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    routing_vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": "CLIP vision tower for routing/anchors (separate from MLLM vision_tower)."},
    )
    text_tower: Optional[str] = field(default=None)
    backbone: Optional[str] = field(
        default=None,
        metadata={"help": "Backbone id: llava | internvl (defaults from PRISM_BACKBONE / run_config)."},
    )
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_text_select_layer: Optional[int] = field(default=-1)
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    task_embedding_dim: Optional[int] = field(default=64)
    benchmark: Optional[str] = field(
        default=None,
        metadata={
            "help": "Benchmark name (e.g. ucit, coin); used with config/benchmarks to fill task_num when --task_num is omitted.",
        },
    )
    task_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of CL tasks/experts; if omitted, inferred from --benchmark via config/benchmarks.",
        },
    )
    method: str = field(
        default="hide_llava",
        metadata={"help": "CL method: cram, hide_llava, same, olora, replay_lora, ft_lora, ewc, ..."},
    )
    exclude_module_path_segments: Optional[Any] = field(
        default=None,
        metadata={
            "help": "PEFT path segments to skip: None=LLaVA defaults (CLIP/mm_projector, ...); []=no filter; non-empty list=custom. "
            "May also be set in config/methods/<method>.py METHOD_CONFIG.",
        },
    )
    peft_target_modules: Any = field(
        default=None,
        metadata={
            "help": (
                "PEFT target_modules (``nn.Linear`` submodule name suffixes): "
                "presets attention(attn) / ffn(mlp) / linear(all,full), or JSON list / comma-separated names; "
                "see PEFT/utils/peft_target_modules.py. Default matches legacy behavior (attention only). "
                "Alias lora_target_modules is merged from METHOD_CONFIG.",
            ),
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    memory_data_path: str = field(default=None, metadata={"help": "Path to the memory data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    

def load_config():
    """Parse CLI into (ModelArguments, DataArguments, TrainingArguments)."""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg == "--expert_num" and i + 1 < len(sys.argv):
            sys.argv[i] = "--task_num"
        elif arg.startswith("--expert_num="):
            sys.argv[i] = "--task_num=" + arg.split("=", 1)[1]

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()
