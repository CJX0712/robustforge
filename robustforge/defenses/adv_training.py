"""对抗训练（Adversarial Training）防御族。

- FGSM-AT：用单步 FGSM 对抗样本训练（快速但鲁棒性有限）。
- PGD-AT ：用多步 PGD 对抗样本训练（Madry et al. 标准强防御）。
两者均复用 NumPyMLP，通过 fit(attack_gen=...) 注入对抗样本。
"""

from __future__ import annotations

from ..attacks.fgsm import fgsm
from ..attacks.pgd import pgd
from ..models.mlp import NumPyMLP
from .base import Defense


class _BaseAT(Defense):
    attack = "pgd"
    name = "adv_training"

    def __init__(self, cfg, eps=None, steps=None, alpha=None):
        self.cfg = cfg
        self.eps = cfg.at_eps if eps is None else eps
        self.steps = cfg.pgd_at_steps if steps is None else steps
        self.alpha = cfg.pgd_at_alpha if alpha is None else alpha
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

    def _gen(self, model, X, y):
        if self.attack == "fgsm":
            return fgsm(model, X, y, eps=self.eps)
        return pgd(model, X, y, eps=self.eps, steps=self.steps, alpha=self.alpha)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y, X_val=X_val, y_val=y_val, attack_gen=self._gen)
        return self


class FGSMAT(_BaseAT):
    attack = "fgsm"
    name = "fgsm_at"


class PGDAT(_BaseAT):
    attack = "pgd"
    name = "pgd_at"
