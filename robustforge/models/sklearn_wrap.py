"""可选 sklearn 后端 —— MLPClassifier 封装。

非核心依赖：缺失时 available() 返回 False，系统全程走 numpy MLP。
由于 sklearn 模型不暴露解析输入梯度，攻击器对该后端自动退化为数值梯度，
保证"模型可换、攻击照常"。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ModelError


def available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


class SklearnMLPWrapper:
    """把 sklearn.neural_network.MLPClassifier 适配为 Classifier 契约。"""

    name = "sklearn_mlp"

    def __init__(self, n_features: int, n_classes: int, random_state: int = 42, **kw):
        if not available():
            raise ModelError("sklearn 未安装，无法使用 SklearnMLPWrapper（请用 NumPyMLP）")
        from sklearn.neural_network import MLPClassifier

        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.classes_ = list(range(self.n_classes))
        self.hidden_dim = kw.get("hidden_layer_sizes", (32,))
        self.model = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            max_iter=kw.get("max_iter", 300),
            random_state=random_state,
            early_stopping=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y))
        # sklearn 的 classes_ 可能是子集；对齐对外类别列表
        self.classes_ = list(self.model.classes_.tolist())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(np.asarray(X, dtype=float))
        # 对齐到全局 classes_ 列顺序（缺失类补 0）
        full = np.zeros((proba.shape[0], self.n_classes), dtype=float)
        for i, c in enumerate(self.model.classes_):
            full[:, c] = proba[:, i]
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=float))

    # 不实现 grad_logits -> 攻击器自动退化为数值梯度
