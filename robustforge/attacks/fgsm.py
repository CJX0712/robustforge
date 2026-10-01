"""FGSM —— Fast Gradient Sign Method（Goodfellow et al., ICLR'15）。

Linf 单步攻击：x_adv = clip(x + eps * sign(∇_x L))。
不可变量：eps=0 时返回原输入（无扰动）。
"""

from __future__ import annotations

import numpy as np

from .base import check_bounds, input_gradient


def fgsm(
    model,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    check_bounds(X, clip_min, clip_max)
    X = np.asarray(X, dtype=float)
    if eps <= 0.0:
        return X.copy()  # 不可变量：零预算不改变样本
    g = input_gradient(model, X, y)
    pert = eps * np.sign(g)
    return np.clip(X + pert, clip_min, clip_max)
