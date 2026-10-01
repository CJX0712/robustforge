"""RobustForge 数据契约（dataclass）。

跨模块统一语义：
- 分类概率列顺序与 classes_ 一致；
- 攻击结果携带真实标签与范数/eps，便于评测与不可变量校验；
- BenchmarkEntry 为跨防御公平评测的统一行格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AttackSpec:
    """一次攻击的配置。"""

    name: str
    norm: str = "linf"  # "linf" | "l2"
    eps: float = 0.1
    steps: int = 40
    alpha: float = 0.01
    random_start: bool = True


@dataclass
class AttackResult:
    """攻击产出：对抗样本 + 元信息。"""

    name: str
    x_adv: np.ndarray
    y_true: np.ndarray
    norm: str
    eps: float
    steps: int
    elapsed: float = 0.0


@dataclass
class RobustRow:
    """单个防御在某攻击下的鲁棒性结果行。"""

    attack: str
    eps: float
    robust_acc: float
    asr: float  # attack success rate（在原本被正确分类的样本上）


@dataclass
class BenchmarkEntry:
    """跨防御公平评测的统一行。"""

    defense: str
    clean_acc: float
    robust_acc_fgsm: float
    robust_acc_pgd: float
    asr_fgsm: float
    asr_pgd: float
    train_seconds: float


@dataclass
class PipelineReport:
    """端到端管线产出。"""

    entries: List[BenchmarkEntry]
    generated_at: str
    config: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "config": self.config,
            "notes": self.notes,
            "entries": [e.__dict__ for e in self.entries],
        }
