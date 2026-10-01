"""纯 numpy 手写 MLP —— RobustForge 离线兜底核心模型。

零第三方依赖可训练、可推理；并实现 grad_logits（交叉熵损失对输入的解析梯度），
使 FGSM/PGD 攻击在 CPU 上极快。支持对抗训练（fit 接收 attack_gen 生成对抗样本）。

不变量（由 tests 守护）：
- 反向传播梯度与中心差分数值梯度一致（max rel err ~1e-6）；
- predict_proba 行和恒为 1（float64 容差 1e-9）；
- 对抗训练后鲁棒精度不低于无防御基线（经验结论，由 benchmark 报告）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ModelError


class NumPyMLP:
    """两层 MLP（输入 -> 隐层 ReLU -> 输出 softmax），SGD + 动量。"""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 128,
        momentum: float = 0.9,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.momentum = float(momentum)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.classes_ = list(range(self.n_classes))
        self._rng = np.random.default_rng(self.random_state)
        self._init_params()

    # ---- 初始化 ----------------------------------------------------------
    def _init_params(self) -> None:
        rng = self._rng
        self.W1 = rng.standard_normal((self.n_features, self.hidden_dim)) * np.sqrt(
            2.0 / self.n_features
        )
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.standard_normal((self.hidden_dim, self.n_classes)) * np.sqrt(
            2.0 / self.hidden_dim
        )
        self.b2 = np.zeros(self.n_classes)
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)

    # ---- 激活 ------------------------------------------------------------
    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---- 前向 ------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return Z1, A1, Z2

    def logits(self, X: np.ndarray) -> np.ndarray:
        _, _, Z2 = self._forward(np.asarray(X, dtype=float))
        return Z2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]

    # ---- 损失与梯度 ------------------------------------------------------
    @staticmethod
    def _ce_grad_logits(Z2: np.ndarray, y: np.ndarray) -> np.ndarray:
        """平均交叉熵对 logits 的梯度，shape (N, c)。"""
        N = Z2.shape[0]
        p = NumPyMLP._softmax(Z2)
        onehot = np.zeros((N, Z2.shape[1]))
        onehot[np.arange(N), y] = 1.0
        return (p - onehot) / N

    def grad_logits(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """平均交叉熵损失对输入 X 的解析梯度，shape (N, n_features)。

        攻击器优先调用它（快速）；若缺失则攻击器退化为数值梯度。
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ModelError(f"X shape {X.shape} 与 n_features={self.n_features} 不匹配")
        Z1, A1, Z2 = self._forward(X)
        dZ2 = self._ce_grad_logits(Z2, y)  # (N, c)
        dA1 = dZ2 @ self.W2.T  # (N, h)
        dZ1 = dA1 * self._relu_grad(Z1)  # (N, h)
        dX = dZ1 @ self.W1.T  # (N, d)
        return dX

    def _loss(self, Z2: np.ndarray, y: np.ndarray) -> float:
        N = Z2.shape[0]
        p = self._softmax(Z2)
        logp = np.log(np.clip(p[np.arange(N), y], 1e-12, None))
        return float(-logp.mean())

    # ---- 训练 ------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val=None,
        y_val=None,
        attack_gen=None,
        epochs: int | None = None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[1] != self.n_features:
            raise ModelError("训练数据维度与模型 n_features 不一致")
        N = X.shape[0]
        epochs = epochs if epochs is not None else self.n_epochs
        rng = self._rng
        for epoch in range(epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X[idx]
                yb = y[idx]
                if attack_gen is not None:
                    xb = attack_gen(self, xb, yb)  # 对抗训练：用当前模型生成对抗样本
                Z1, A1, Z2 = self._forward(xb)
                dZ2 = self._ce_grad_logits(Z2, yb)
                dW2 = A1.T @ dZ2
                db2 = dZ2.sum(axis=0)
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * self._relu_grad(Z1)
                dW1 = xb.T @ dZ1
                db1 = dZ1.sum(axis=0)
                self._vW2 = self.momentum * self._vW2 - self.learning_rate * dW2
                self._vb2 = self.momentum * self._vb2 - self.learning_rate * db2
                self._vW1 = self.momentum * self._vW1 - self.learning_rate * dW1
                self._vb1 = self.momentum * self._vb1 - self.learning_rate * db1
                self.W2 += self._vW2
                self.b2 += self._vb2
                self.W1 += self._vW1
                self.b1 += self._vb1
                epoch_loss += self._loss(Z2, yb)
                n_batches += 1
            if self.verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                val = ""
                if X_val is not None:
                    val = f" val_acc={float((self.predict(X_val) == y_val).mean()):.4f}"
                print(
                    f"[NumPyMLP] epoch {epoch:03d} loss={epoch_loss / max(n_batches, 1):.4f}{val}"
                )
        return self

    # ---- 序列化（用于最优鲁棒检查点保存/恢复） ---------------------------
    def get_params(self) -> dict:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_params(self, params: dict) -> None:
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
