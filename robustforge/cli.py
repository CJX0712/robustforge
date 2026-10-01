"""RobustForge 命令行入口。

用法：
    python -m robustforge.cli                 # 跑端到端 benchmark，落盘 benchmark.json
    python -m robustforge.cli --epochs 8      # 调参（环境变量 RF_* 亦可）
    python -m robustforge.cli info            # 打印系统信息
"""

from __future__ import annotations

import argparse
import sys

from .core.config import Config
from .pipeline.pipeline import RobustnessPipeline


def _print_table(report) -> None:
    cols = ["defense", "clean", "rob_fgsm", "rob_pgd", "asr_fgsm", "asr_pgd", "sec"]
    head = "".join(f"{c:<14}" for c in cols)
    print(head)
    print("-" * len(head))
    for e in report.entries:
        print(
            f"{e.defense:<14}"
            f"{e.clean_acc:<14.4f}"
            f"{e.robust_acc_fgsm:<14.4f}"
            f"{e.robust_acc_pgd:<14.4f}"
            f"{e.asr_fgsm:<14.4f}"
            f"{e.asr_pgd:<14.4f}"
            f"{e.train_seconds:<14.1f}"
        )


def _cmd_run(args) -> int:
    cfg = Config.from_env()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_classes is not None:
        cfg.n_classes = args.n_classes
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.at_eps is not None:
        cfg.at_eps = args.at_eps
    if args.attack_eps is not None:
        cfg.attack_eps = args.attack_eps
    if args.verbose:
        cfg.verbose = True

    pipe = RobustnessPipeline(cfg)
    report = pipe.run_and_dump(args.out)
    _print_table(report)
    print(f"\n已落盘: {args.out}")
    print(f"生成时间: {report.generated_at}")
    return 0


def _cmd_info(_args) -> int:
    try:
        from . import __author__, __version__

        print(f"RobustForge v{__version__} (作者: {__author__})")
    except Exception:
        print("RobustForge")
    print("域: 对抗鲁棒性 / 对抗机器学习 (Adversarial Robustness)")
    print("核心: 纯 numpy MLP + FGSM/PGD/CW 攻击 + 对抗训练(AT)")
    print("旗舰: RobustFuse-AT (自适应ε退火 + 鲁棒早停 + 非劣守护)")
    print("参照开源: IBM ART / cleverhans / foolbox")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="robustforge", description="RobustForge CLI")
    sub = parser.add_subparsers(dest="cmd")
    run_p = sub.add_parser("run", help="运行端到端 benchmark")
    run_p.add_argument("--epochs", type=int, default=None)
    run_p.add_argument("--n-classes", type=int, default=None)
    run_p.add_argument("--n-samples", type=int, default=None)
    run_p.add_argument("--at-eps", type=float, default=None)
    run_p.add_argument("--attack-eps", type=float, default=None)
    run_p.add_argument("--out", default="benchmark.json")
    run_p.add_argument("--verbose", action="store_true")
    sub.add_parser("info", help="打印系统信息")
    args = parser.parse_args(argv)
    if args.cmd == "info":
        return _cmd_info(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
