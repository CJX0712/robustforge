"""攻击公共工具：输入梯度解析/数值统一入口。

策略：若模型提供 grad_logits（如 NumPyMLP）则直接用解析梯度（快）；
否则对 predict_proba 做中心差分数值梯度（模型可换、攻击照常）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import AttackError


def _ce_from_proba(proba: np.ndarray, y: np.ndarray) -> float:
    N = proba.shape[0]
    p = np.clip(proba[np.arange(N), y], 1e-12, None)
    return float(-np.log(p).mean())


def input_gradient(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """平均交叉熵损失对输入 X 的梯度，shape (N, n_features)。"""
    if hasattr(model, "grad_logits"):
        g = model.grad_logits(X, y)
        if g is not None:
            return np.asarray(g, dtype=float)
    return numerical_gradient(model, X, y)


def numerical_gradient(model, X: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    g = np.zeros_like(X)
    for j in range(d):
        xp = X.copy()
        xm = X.copy()
        xp[:, j] += eps
        xm[:, j] -= eps
        lp = _ce_from_proba(model.predict_proba(xp), y)
        lm = _ce_from_proba(model.predict_proba(xm), y)
        g[:, j] = (lp - lm) / (2.0 * eps)
    return g


def check_bounds(X: np.ndarray, clip_min: float, clip_max: float) -> None:
    if clip_min >= clip_max:
        raise AttackError("clip_min 必须小于 clip_max")
    if X.ndim != 2:
        raise AttackError("X 必须为二维 (N, d)")
