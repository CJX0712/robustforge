"""鲁棒性评测指标。

统一语义：分数越大越"好"（clean_acc / robust_acc 越大越好；asr 越小越好）。
所有指标对模型无关（仅依赖 predict）。
"""

from __future__ import annotations

import time

import numpy as np


def clean_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(np.asarray(X, dtype=float)) == np.asarray(y)).mean())


def robust_accuracy(model, X, y, attack_fn, **attack_kw) -> tuple[float, np.ndarray]:
    """在攻击生成的对抗样本上的精度。返回 (robust_acc, x_adv)。"""
    x_adv = attack_fn(model, np.asarray(X, dtype=float), np.asarray(y), **attack_kw)
    return float((model.predict(x_adv) == np.asarray(y)).mean()), x_adv


def attack_success_rate(model, X, y, attack_fn, **attack_kw) -> tuple[float, np.ndarray]:
    """在"原本被正确分类"的样本上的攻击成功率。返回 (asr, x_adv)。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    clean_correct = model.predict(X) == y
    x_adv = attack_fn(model, X, y, **attack_kw)
    adv_correct = model.predict(x_adv) == y
    if clean_correct.sum() == 0:
        return 0.0, x_adv
    asr = float((clean_correct & ~adv_correct).sum() / clean_correct.sum())
    return asr, x_adv


def linf_violation(x_adv: np.ndarray, X: np.ndarray, eps: float) -> float:
    """返回 max ||x_adv - X||_inf - eps（应 <= 0，允许极小浮点误差）。"""
    X = np.asarray(X, dtype=float)
    x_adv = np.asarray(x_adv, dtype=float)
    return float(np.max(np.abs(x_adv - X)) - eps)


def eval_attack(model, X, y, attack_fn, eps, steps, alpha) -> dict:
    """对单个攻击配置跑全套指标（带计时）。"""
    t0 = time.perf_counter()
    ra, _ = robust_accuracy(model, X, y, attack_fn, eps=eps, steps=steps, alpha=alpha)
    asr, _ = attack_success_rate(model, X, y, attack_fn, eps=eps, steps=steps, alpha=alpha)
    elapsed = time.perf_counter() - t0
    return {"robust_acc": ra, "asr": asr, "elapsed": elapsed}
