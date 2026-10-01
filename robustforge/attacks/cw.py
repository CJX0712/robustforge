"""CW-L2 —— Carlini & Wagner L2 攻击（CW, S&P'17）的忠实实现。

采用标准 tanh 变量变换将样本约束在 [0,1]，最小化
    L(w) = Σ_i (||x_i'(w) - X_i||^2 + c * f_i)，
    x'(w) = 0.5*(tanh(w)+1),  f_i = max(max_{j!=t} Z_j - Z_t + kappa, 0)
并对 c 做小范围二分搜索，挑选扰动最小且成功的样本。

梯度通过对 CW 损失做中心差分（模型无关），使 CW 可作用于任意 Classifier 后端。
注意：CW 计算较重，默认不进入端到端 benchmark；由 tests 在小样本上验证正确性。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import AttackError


def _logits_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-12, 1.0))


def _zt(model, x, y):
    return _logits_from_proba(model.predict_proba(x))[np.arange(x.shape[0]), y]


def _zmax(model, x, y):
    logits = _logits_from_proba(model.predict_proba(x)).copy()
    logits[np.arange(x.shape[0]), y] = -1e9
    return logits.max(axis=1)


def _fval(model, x, y, kappa):
    return np.maximum(_zmax(model, x, y) - _zt(model, x, y) + kappa, 0.0)


def _objective(model, x, X, y, c, kappa):
    l2 = np.sum((x - X) ** 2, axis=1)
    f = _fval(model, x, y, kappa)
    return float(np.sum(l2 + c * f))


def _cw_loss_and_grad(model, w, y, c, kappa, X, eps=1e-3):
    x = 0.5 * (np.tanh(w) + 1.0)
    loss = _objective(model, x, X, y, c, kappa)
    N, d = x.shape
    dLdx = np.zeros_like(x)
    for j in range(d):
        wp = w.copy()
        wm = w.copy()
        wp[:, j] += eps
        wm[:, j] -= eps
        xp = 0.5 * (np.tanh(wp) + 1.0)
        xm = 0.5 * (np.tanh(wm) + 1.0)
        lp = _objective(model, xp, X, y, c, kappa)
        lm = _objective(model, xm, X, y, c, kappa)
        dLdx[:, j] = (lp - lm) / (2.0 * eps)
    dLdw = dLdx * (1.0 - np.tanh(w) ** 2) / 2.0
    return loss, dLdw


def cw_l2(
    model,
    X,
    y,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 30,
    lr: float = 0.01,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    binary_search_steps: int = 5,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise AttackError("CW-L2 要求 X 为二维 (N, d)")
    N = X.shape[0]
    Xc = np.clip(X, clip_min + 1e-6, clip_max - 1e-6)
    w0 = np.arctanh(2.0 * Xc - 1.0)
    best_x = X.copy()
    best_dist = np.full(N, np.inf)
    cs = np.logspace(-2, 1, binary_search_steps) if binary_search_steps > 1 else np.array([c])
    for ci in cs:
        w = w0.copy()
        for _ in range(steps):
            _, g = _cw_loss_and_grad(model, w, y, ci, kappa, X)
            w = w - lr * g
            x_adv = np.clip(0.5 * (np.tanh(w) + 1.0), clip_min, clip_max)
            succ = model.predict(x_adv) != y
            dist = np.linalg.norm(x_adv - X, axis=1)
            for i in range(N):
                if succ[i] and dist[i] < best_dist[i]:
                    best_dist[i] = dist[i]
                    best_x[i] = x_adv[i]
    return best_x
