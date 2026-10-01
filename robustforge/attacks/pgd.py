"""PGD —— Projected Gradient Descent（Madry et al., ICLR'18）。

Linf / L2 多步最强一阶攻击。每步投影回 eps-球并裁剪到数据范围 [clip_min, clip_max]。
不可变量：返回样本始终落在以 X 为中心、半径 eps 的球内（距离 <= eps + 1e-9）。
"""

from __future__ import annotations

import numpy as np

from .base import check_bounds, input_gradient


def pgd(
    model,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    steps: int = 40,
    alpha: float = 0.01,
    random_start: bool = True,
    norm: str = "linf",
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    check_bounds(X, clip_min, clip_max)
    X = np.asarray(X, dtype=float)
    if eps <= 0.0 or steps <= 0:
        return X.copy()  # 不可变量：无预算/无步数 -> 原输入

    x_adv = X.copy()
    if random_start:
        if norm == "linf":
            x_adv = x_adv + np.random.uniform(-eps, eps, size=X.shape)
        else:
            delta = np.random.randn(*X.shape)
            nrm = np.linalg.norm(delta, axis=1, keepdims=True)
            delta = delta / np.clip(nrm, 1e-12, None) * eps
            x_adv = x_adv + delta
        x_adv = np.clip(x_adv, clip_min, clip_max)

    for _ in range(steps):
        g = input_gradient(model, x_adv, y)
        if norm == "linf":
            step = alpha * np.sign(g)
        else:
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            step = alpha * g / np.clip(gn, 1e-12, None)
        x_adv = x_adv + step
        # 投影回 eps-球
        if norm == "linf":
            x_adv = np.clip(x_adv, X - eps, X + eps)
        else:
            delta = x_adv - X
            dn = np.linalg.norm(delta, axis=1, keepdims=True)
            over = dn > eps
            delta = np.where(over, delta * (eps / np.clip(dn, 1e-12, None)), delta)
            x_adv = X + delta
        x_adv = np.clip(x_adv, clip_min, clip_max)
    return x_adv
