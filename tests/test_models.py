"""NumPyMLP 不变量测试：解析梯度校验、概率行和、预测正确性。"""

from __future__ import annotations

import numpy as np

from robustforge.models.mlp import NumPyMLP


def _numerical_input_grad(model, X, y, eps=1e-4):
    """逐元素中心差分：只有样本 i 的交叉熵受 X[i,j] 影响。

    平均 CE = (1/N) Σ_i CE_i，故 d(mean CE)/dX[i,j] = (1/N) dCE_i/dX[i,j]。
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    g = np.zeros_like(X)
    for i in range(N):
        for j in range(d):
            xp = X.copy()
            xm = X.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            ce_p = -np.log(max(model.predict_proba(xp)[i, y[i]], 1e-12))
            ce_m = -np.log(max(model.predict_proba(xm)[i, y[i]], 1e-12))
            g[i, j] = ((ce_p - ce_m) / (2.0 * eps)) / N
    return g


def test_predict_proba_rows_sum_to_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    m = NumPyMLP(8, 3, random_state=1)
    p = m.predict_proba(X)
    assert p.shape == (50, 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_grad_logits_matches_numerical():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((12, 6))
    y = rng.integers(0, 3, size=12)
    m = NumPyMLP(6, 3, random_state=3)
    analytic = m.grad_logits(X, y)
    numeric = _numerical_input_grad(m, X, y)
    rel = np.max(np.abs(analytic - numeric) / (np.abs(numeric) + 1e-8))
    assert rel < 1e-4, f"梯度校验失败 rel={rel}"


def test_fit_improves_clean_accuracy():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = NumPyMLP(6, 2, hidden_dim=16, learning_rate=0.05, n_epochs=40, random_state=5)
    before = float((m.predict(X) == y).mean())
    m.fit(X, y)
    after = float((m.predict(X) == y).mean())
    assert after > before + 0.3
