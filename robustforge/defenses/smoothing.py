"""随机平滑（Randomized Smoothing）—— Cohen et al., ICML'19 认证防御。

对输入加高斯噪声后取多数投票得到平滑分类器；可给出 L2 认证半径下界。
较重（需大量噪声样本），默认不进入端到端 benchmark；由 tests 验证。"""

from __future__ import annotations

import numpy as np


class RandomizedSmoothing:
    name = "randomized_smoothing"

    def __init__(self, base_model, sigma: float = 0.25, n_samples: int = 100):
        self.base_model = base_model
        self.sigma = float(sigma)
        self.n_samples = int(n_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        votes = np.zeros((N, self.base_model.n_classes), dtype=float)
        for _ in range(self.n_samples):
            noise = np.random.normal(0.0, self.sigma, size=X.shape)
            proba = self.base_model.predict_proba(np.clip(X + noise, 0.0, 1.0))
            votes += proba
        return np.array(self.base_model.classes_)[np.argmax(votes, axis=1)]

    def certify(self, X: np.ndarray, alpha: float = 0.001) -> np.ndarray:
        """简化认证半径：返回每个样本下界半径（用计数近似，演示用途）。"""
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        radii = np.zeros(N)
        for i in range(N):
            counts = np.zeros(self.base_model.n_classes)
            xi = np.repeat(X[i : i + 1], self.n_samples, axis=0)
            noise = np.random.normal(0.0, self.sigma, size=xi.shape)
            preds = self.base_model.predict(np.clip(xi + noise, 0.0, 1.0))
            for p in preds:
                counts[self.base_model.classes_.index(int(p))] += 1
            counts = counts / self.n_samples
            top2 = np.sort(counts)[::-1][:2]
            p_a = top2[0]
            if p_a <= 0.5:
                radii[i] = 0.0
                continue
            # Cohen et al. 认证半径（高斯）：sigma * Phi^{-1}(p_a) 近似
            from scipy.stats import norm

            radii[i] = self.sigma * norm.ppf(p_a)
        return radii
