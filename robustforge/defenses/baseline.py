"""无防御基线：标准（非对抗）训练。作为鲁棒性对照下界。"""

from __future__ import annotations

from ..models.mlp import NumPyMLP
from .base import Defense


class Undefended(Defense):
    name = "undefended"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = NumPyMLP(
            n_features=cfg.n_features,
            n_classes=cfg.n_classes,
            hidden_dim=cfg.hidden_dim,
            learning_rate=cfg.learning_rate,
            n_epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            momentum=cfg.momentum,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y, X_val=X_val, y_val=y_val)
        return self
