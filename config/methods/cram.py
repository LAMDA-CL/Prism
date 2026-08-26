"""Prism defaults for the core CRAM continual-learning method."""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION
from utils.infer_paths import DEFAULT_ROUTING_VISION_TOWER_PATH, DEFAULT_TEXT_TOWER_PATH


TRAIN_FLAG_OVERRIDES = {
    "--method": "cram",
    "--mm_projector_lr": "2e-5",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
}

TRAIN_EXTRA_ARGS: list[str] = []

INFER_DEFAULTS = {
    "clmethod": "cram",
    "batch_size": 8,
    "routing_vision_tower": DEFAULT_ROUTING_VISION_TOWER_PATH,
    "text_tower": DEFAULT_TEXT_TOWER_PATH,
}

TRAIN_BATCH_SIZES = {
    "coin": {task_id: 4 for task_id in range(8)},
    "ucit": {task_id: 12 for task_id in range(6)},
    "trigap": {task_id: 12 for task_id in range(10)},
}

METHOD_CONFIG = {
    "cram_use_checkpoint_hyperparams": False,
    "cram_svd_tau_alloc": 0.08,
    "cram_rank_total": 48,
    "lora_alpha": 96,
    "cram_visual_warmup_ratio": 0.05,
    "cram_svd_rank_min": 4,
    "cram_svd_tau_novel": 0.99,
    "lora_dropout": 0.05,
    "clip_feature_dim": 768,
    "cram_centroid_agg": "image_sum",
    "cram_route_rbf_sigma": 2,
    "cram_route_topk": 0,
    "cram_infer_route_topk": 0,
    "cram_dec_lambda": 1,
    "cram_buf_rank": 48,
    "cram_level1_single_pool_merge_min_sim": 0.8,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
    "peft_target_modules": "attn",
}

METHOD_CONFIG_BY_BENCHMARK = {
    "ucit": {
        "cram_expert_rank_max": 9,
        "cram_delta_threshold": 0.1,
        "cram_max_expert_slots": 10,
        "cram_max_groups": 5,
    },
    "trigap": {
        "cram_expert_rank_max": 10,
        "cram_delta_threshold": 0.1,
        "cram_max_expert_slots": 10,
        "cram_max_groups": 10,
    },
}
