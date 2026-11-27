"""DRAPE defaults, matching MLLM-CL's main generator/NSP/router configuration."""
from config.paths.llava_paths import CLIP_PATH

TRAIN_FLAG_OVERRIDES = {
    "--method": "drape",
    "--lora_enable": "False",
    "--freeze_mm_mlp_adapter": "False",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--weight_decay": "0",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--model_max_length": "2048",
    # Final task checkpoints carry the fitted routing key and cumulative NSP.
    "--save_strategy": "no",
}
TRAIN_EXTRA_ARGS = []
TRAIN_BATCH_SIZES = {"coin": {i: 4 for i in range(8)}, "ucit": {i: 4 for i in range(6)}}
TRAIN_GRADIENT_ACCUMULATION_STEPS = 1
INFER_DEFAULTS = {"clmethod": "drape", "batch_size": 1, "routing_vision_tower": None, "text_tower": None}
METHOD_CONFIG = {
    "generator_type": "CROSS_ATTENTION",
    "generator_hidden_size": 512,
    "generator_num_layers": 2,
    "generator_dropout": 0.1,
    "num_virtual_tokens": 10,
    "vision_feature_select_strategy": "full_patch",
    "separate": False,
    "pool_mode": "segment",
    "query_mode": "segpool",
    "enable_continual_learning": True,
    "clip_model_name": CLIP_PATH,
    "router_num_centers": 1,
    "router_kmeans_iters": 20,
    "router_k1_center_mode": "learned",
    "null_space_threshold": 0.99,
    "apply_null_space_projection": True,
}
