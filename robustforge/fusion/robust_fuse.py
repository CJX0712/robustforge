"""旗舰：RobustFuse-AT —— 自适应 ε 对抗训练 + 鲁棒性感知早停 + 非劣守护。

设计取舍（公平且诚实，参照 Athalye et al. 对"混淆梯度"型防御的告诫）：
1. 训练目标就是标准 min-max 对抗训练（PGD-AT），不使用任何会掩盖真实梯度的技巧；
2. 自适应 ε 退火：前 warmup_frac 比例 epoch 用小 ε 升温，避免强扰动致训崩、提升稳定性；
3. 鲁棒性感知早停：逐 epoch 用 PGD 探针（确定性 random_start=False）测验证集鲁棒精度，
   保存最优鲁棒检查点 —— 相比"用最后一个 epoch"，通常避开过拟合、拿到更高鲁棒精度；
4. 非劣守护：若旗舰鲁棒精度低于 reference（标准 PGD-AT 末轮）超过 guard_margin，则诚实退回 reference。

全程纯 numpy；reference 可选（不传则只做自适应ε+鲁棒早停）。
"""

from __future__ import annotations

from ..attacks.pgd import pgd
from ..defenses.base import Defense
from ..models.mlp import NumPyMLP


class RobustFuseAT(Defense):
    name = "robustfuse_at"

    def __init__(
        self,
        cfg,
        eps_max=None,
        warmup_frac=None,
        patience=None,
        guard_margin=None,
        reference=None,
    ):
        self.cfg = cfg
        self.eps_max = cfg.at_eps if eps_max is None else eps_max
        self.warmup_frac = cfg.warmup_frac if warmup_frac is None else warmup_frac
        self.patience = cfg.patience if patience is None else patience
        self.guard_margin = cfg.guard_margin if guard_margin is None else guard_margin
        self.reference = reference  # 标准 PGD-AT 末轮模型（非劣守护基准）
        self.model = NumPyMLP(
            n_features=cfg.n_features,
            n_classes=cfg.n_classes,
            hidden_dim=cfg.hidden_dim,
            learning_rate=cfg.learning_rate,
            n_epochs=1,
            batch_size=cfg.batch_size,
            momentum=cfg.momentum,
            random_state=cfg.random_state,
            verbose=False,
        )
        self.best_robust_acc = -1.0
        self.epochs_run = 0
        self._fell_back = False

    def _eps_schedule(self, epoch: int, total: int) -> float:
        warmup = max(1, int(self.warmup_frac * total))
        return self.eps_max * min(1.0, (epoch + 1) / warmup)

    @staticmethod
    def _make_attack_gen(eps, steps, alpha):
        return lambda m, Xb, yb: pgd(  # noqa: E731
            m, Xb, yb, eps=eps, steps=steps, alpha=alpha
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            n = max(1, int(0.2 * len(X)))
            X_val, y_val = X[:n], y[:n]
        total = self.cfg.epochs
        wait = 0
        for epoch in range(total):
            eps = self._eps_schedule(epoch, total)
            attack_gen = self._make_attack_gen(
                eps, self.cfg.pgd_at_steps, self.cfg.pgd_at_alpha
            )
            self.model.fit(X, y, X_val=X_val, y_val=y_val, attack_gen=attack_gen, epochs=1)
            # 确定性鲁棒探针（部署强度 eps_max）
            x_probe = pgd(
                self.model,
                X_val,
                y_val,
                eps=self.eps_max,
                steps=self.cfg.eval_pgd_steps,
                alpha=self.cfg.eval_pgd_alpha,
                random_start=False,
            )
            rob = float((self.model.predict(x_probe) == y_val).mean())
            if rob > self.best_robust_acc:
                self.best_robust_acc = rob
                self._best_params = self.model.get_params()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        # 恢复最优鲁棒检查点
        self.model.set_params(self._best_params)
        self.epochs_run = epoch + 1

        # 非劣守护
        if self.reference is not None:
            x_ref = pgd(
                self.reference,
                X_val,
                y_val,
                eps=self.eps_max,
                steps=self.cfg.eval_pgd_steps,
                alpha=self.cfg.eval_pgd_alpha,
                random_start=False,
            )
            rob_ref = float((self.reference.predict(x_ref) == y_val).mean())
            if self.best_robust_acc < rob_ref - self.guard_margin:
                self.model = self.reference
                self._fell_back = True
                self.best_robust_acc = rob_ref
        return self
