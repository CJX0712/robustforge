"""端到端管线：加载数据 -> 训练各防御 -> 公平评测 -> 产出报告。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from ..core.config import Config
from ..core.types import BenchmarkEntry, PipelineReport
from ..defenses.adv_training import FGSMAT, PGDAT
from ..defenses.baseline import Undefended
from ..eval.benchmark import evaluate_model
from ..fusion.robust_fuse import RobustFuseAT


def _load_digits_subset(cfg: Config):
    """加载 sklearn digits，按 n_classes / n_samples 筛选并切分 train/test/val。"""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 scikit-learn 提供 digits 数据集") from exc

    data = load_digits()
    X = data.images.reshape(len(data.images), -1).astype(float) / 16.0  # 归一化到 [0,1]
    y = data.target.astype(int)

    keep = y < cfg.n_classes
    X, y = X[keep], y[keep]

    rng = np.random.default_rng(cfg.random_state)
    if X.shape[0] > cfg.n_samples:
        idx = rng.choice(X.shape[0], size=cfg.n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=cfg.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=cfg.random_state, stratify=y_tr
    )
    return X_tr, y_tr, X_val, y_val, X_te, y_te


class RobustnessPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.from_env()

    def run(self) -> PipelineReport:
        cfg = self.cfg
        np.random.seed(cfg.random_state)  # 固定攻击 random_start 噪声，保证可复现
        X_tr, y_tr, X_val, y_val, X_te, y_te = _load_digits_subset(cfg)
        cfg.n_features = X_tr.shape[1]

        # 先训练标准 PGD-AT 作为旗舰的 reference（非劣守护基准）
        pgd_at = PGDAT(cfg)
        pgd_at.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

        defenses = [
            Undefended(cfg),
            FGSMAT(cfg),
            pgd_at,
            RobustFuseAT(cfg, reference=pgd_at.model),
        ]

        entries: list[BenchmarkEntry] = []
        for d in defenses:
            t0 = time.perf_counter()
            if d.name == "robustfuse_at":
                d.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            else:
                d.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            m = d.get_model()
            metrics = evaluate_model(m, X_te, y_te, cfg)
            entries.append(
                BenchmarkEntry(
                    defense=d.name,
                    clean_acc=metrics["clean_acc"],
                    robust_acc_fgsm=metrics["robust_acc_fgsm"],
                    robust_acc_pgd=metrics["robust_acc_pgd"],
                    asr_fgsm=metrics["asr_fgsm"],
                    asr_pgd=metrics["asr_pgd"],
                    train_seconds=time.perf_counter() - t0,
                )
            )

        report = PipelineReport(
            entries=entries,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            config=cfg.as_dict(),
            notes=(
                "评测攻击统一使用 eps=%s 的 FGSM 与 PGD(%d步)。"
                "robustfuse_at 采用自适应ε退火+鲁棒早停，并以 pgd_at 为非劣守护基准。"
                % (cfg.attack_eps, cfg.eval_pgd_steps)
            ),
        )
        return report

    def run_and_dump(self, out_path: str = "benchmark.json") -> PipelineReport:
        report = self.run()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return report
