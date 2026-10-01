"""防御与非劣守护测试。"""

from __future__ import annotations

import numpy as np

from robustforge.core.config import Config
from robustforge.defenses.adv_training import FGSMAT, PGDAT
from robustforge.defenses.baseline import Undefended
from robustforge.fusion.robust_fuse import RobustFuseAT


def _mini_cfg():
    cfg = Config()
    cfg.n_features = 8
    cfg.n_classes = 2
    cfg.epochs = 15
    cfg.learning_rate = 0.02
    cfg.pgd_at_steps = 4
    cfg.eval_pgd_steps = 8
    cfg.at_eps = 0.1
    cfg.attack_eps = 0.1
    cfg.verbose = False
    return cfg


def _data(cfg):
    rng = np.random.default_rng(11)
    X = rng.standard_normal((200, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_defenses_train_and_classify():
    cfg = _mini_cfg()
    X, y = _data(cfg)
    for D in (Undefended, FGSMAT, PGDAT):
        d = D(cfg)
        d.fit(X, y, X[:20], y[:20])
        acc = float((d.get_model().predict(X) == y).mean())
        assert acc > 0.6


def test_robustfuse_non_inferiority_guard():
    cfg = _mini_cfg()
    X, y = _data(cfg)
    Xtr, Xval, ytr, yval = X[:160], X[160:], y[:160], y[160:]
    pgd_at = PGDAT(cfg)
    pgd_at.fit(Xtr, ytr, X_val=Xval, y_val=yval)
    reference = pgd_at.get_model()

    # 在验证集上算 reference 鲁棒精度（与旗舰探针同口径：确定性 PGD）
    from robustforge.attacks.pgd import pgd

    x_ref = pgd(
        reference,
        Xval,
        yval,
        eps=cfg.at_eps,
        steps=cfg.eval_pgd_steps,
        alpha=cfg.eval_pgd_alpha,
        random_start=False,
    )
    rob_ref = float((reference.predict(x_ref) == yval).mean())

    rf = RobustFuseAT(cfg, reference=reference)
    rf.fit(Xtr, ytr, X_val=Xval, y_val=yval)
    # 非劣守护：旗舰最优鲁棒精度 >= reference 鲁棒精度 - margin(0)
    assert rf.best_robust_acc >= rob_ref - 1e-9


def test_pipeline_runs_small():
    from robustforge.pipeline.pipeline import RobustnessPipeline

    cfg = _mini_cfg()
    cfg.n_samples = 200
    cfg.n_classes = 2
    cfg.epochs = 5
    cfg.pgd_at_steps = 4
    cfg.eval_pgd_steps = 8
    cfg.verbose = False
    report = RobustnessPipeline(cfg).run()
    assert len(report.entries) == 4
    for e in report.entries:
        assert 0.0 <= e.clean_acc <= 1.0
        assert 0.0 <= e.robust_acc_pgd <= 1.0
