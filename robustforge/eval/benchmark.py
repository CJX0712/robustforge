"""单模型跨攻击评测：clean / FGSM / PGD 全套指标。"""

from __future__ import annotations

import numpy as np

from ..attacks.fgsm import fgsm
from ..attacks.pgd import pgd
from .metrics import attack_success_rate, clean_accuracy, robust_accuracy


def evaluate_model(model, X_te, y_te, cfg) -> dict:
    """对单个防御模型跑 clean + FGSM + PGD 评测，返回指标字典。"""
    X_te = np.asarray(X_te, dtype=float)
    y_te = np.asarray(y_te)
    clean = clean_accuracy(model, X_te, y_te)
    ra_f, _ = robust_accuracy(model, X_te, y_te, fgsm, eps=cfg.attack_eps)
    ra_p, _ = robust_accuracy(
        model,
        X_te,
        y_te,
        pgd,
        eps=cfg.attack_eps,
        steps=cfg.eval_pgd_steps,
        alpha=cfg.eval_pgd_alpha,
    )
    asr_f, _ = attack_success_rate(model, X_te, y_te, fgsm, eps=cfg.attack_eps)
    asr_p, _ = attack_success_rate(
        model,
        X_te,
        y_te,
        pgd,
        eps=cfg.attack_eps,
        steps=cfg.eval_pgd_steps,
        alpha=cfg.eval_pgd_alpha,
    )
    return {
        "clean_acc": clean,
        "robust_acc_fgsm": ra_f,
        "robust_acc_pgd": ra_p,
        "asr_fgsm": asr_f,
        "asr_pgd": asr_p,
    }
