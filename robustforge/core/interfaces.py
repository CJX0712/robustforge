"""接口契约（Protocol）。

Classifier 是攻击/防御/评测的统一抽象；任何提供 predict/predict_proba 的对象
都可被接入。grad_logits 为可选高性能梯度入口，缺失时攻击自动退化为数值梯度。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Classifier(Protocol):
    """可被攻击与评测的分类器契约。"""

    n_classes: int
    classes_: list

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def grad_logits(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """可选：平均交叉熵损失对输入 X 的解析梯度，shape (N, n_features)。"""
        ...
