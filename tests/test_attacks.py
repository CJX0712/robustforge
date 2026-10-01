"""攻击不变量测试：FGSM 零预算、PGD 球面有界、攻击不提升精度。"""

from __future__ import annotations

import numpy as np

from robustforge.attacks.cw import cw_l2
from robustforge.attacks.fgsm import fgsm
from robustforge.attacks.pgd import pgd
from robustforge.eval.metrics import clean_accuracy, robust_accuracy
from robustforge.models.mlp import NumPyMLP


def _trained_model():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = NumPyMLP(6, 2, hidden_dim=16, learning_rate=0.05, n_epochs=40, random_state=8)
    m.fit(X, y)
    return m, X, y


def test_fgsm_zero_eps_returns_input():
    m, X, y = _trained_model()
    x_adv = fgsm(m, X[:5], y[:5], eps=0.0)
    assert np.allclose(x_adv, X[:5], atol=1e-12)


def test_pgd_within_eps_ball():
    # 数据落在 [0,1] 数据范围：Linf-球约束仅在 X 位于裁剪范围内成立
    rng = np.random.default_rng(7)
    X = rng.random((50, 6))
    y = rng.integers(0, 2, size=50)
    m = NumPyMLP(6, 2, random_state=8)  # 攻击几何不依赖是否已训练
    eps = 0.15
    x_adv = pgd(m, X, y, eps=eps, steps=20, alpha=0.02)
    viol = float(np.max(np.abs(x_adv - X)) - eps)
    assert viol <= 1e-6, f"PGD 越界: {viol}"


def test_attack_does_not_improve_accuracy():
    m, X, y = _trained_model()
    clean = clean_accuracy(m, X, y)
    ra, _ = robust_accuracy(m, X, y, fgsm, eps=0.1)
    assert ra <= clean + 1e-6


def test_cw_runs_and_produces_adversarial():
    m, X, y = _trained_model()
    Xs, ys = X[:4], y[:4]
    x_adv = cw_l2(m, Xs, ys, c=1.0, steps=15, lr=0.02, binary_search_steps=3)
    assert x_adv.shape == Xs.shape
    # 至少部分样本被成功误分类（CW 设计目标）
    n_mis = int((m.predict(x_adv) != ys).sum())
    assert n_mis >= 1
