"""全局配置 + ENV_XXX_* 环境变量覆盖。

所有可调旋钮集中在 Config；环境变量以 RF_ 前缀覆盖，便于无代码改参。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class Config:
    """RobustForge 运行配置（默认值面向 CPU 快速可复现）。"""

    random_state: int = 42
    n_classes: int = 3
    n_samples: int = 400
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 0.01
    hidden_dim: int = 32
    momentum: float = 0.9

    # 对抗训练
    pgd_at_steps: int = 7
    pgd_at_alpha: float = 0.03
    at_eps: float = 0.1

    # 评测攻击
    attack_eps: float = 0.1
    eval_pgd_steps: int = 20
    eval_pgd_alpha: float = 0.01

    # 旗舰 RobustFuse-AT
    warmup_frac: float = 0.4
    patience: int = 5
    guard_margin: float = 0.0

    enable_smoothing: bool = False
    verbose: bool = False
    n_features: int = 64  # 由数据填充

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        mapping: Dict[str, Tuple[str, Callable[[str], object]]] = {
            "RF_RANDOM_STATE": ("random_state", int),
            "RF_N_CLASSES": ("n_classes", int),
            "RF_N_SAMPLES": ("n_samples", int),
            "RF_EPOCHS": ("epochs", int),
            "RF_BATCH_SIZE": ("batch_size", int),
            "RF_LEARNING_RATE": ("learning_rate", float),
            "RF_HIDDEN_DIM": ("hidden_dim", int),
            "RF_AT_EPS": ("at_eps", float),
            "RF_ATTACK_EPS": ("attack_eps", float),
            "RF_ENABLE_SMOOTHING": (
                "enable_smoothing",
                lambda s: s.lower() in ("1", "true", "yes"),
            ),
            "RF_VERBOSE": ("verbose", lambda s: s.lower() in ("1", "true", "yes")),
        }
        for env_key, (attr, caster) in mapping.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    setattr(cfg, attr, caster(val))
                except (ValueError, TypeError):
                    # 解析失败保持默认，不做破坏性行为
                    pass
        return cfg

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
