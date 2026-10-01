"""端到端演示：训练四类防御并公平评测，落盘 benchmark.json。

直接运行：
    python -m robustforge.examples.run_demo
"""

from __future__ import annotations

import os
import sys

# 允许以脚本直接运行（python robustforge/examples/run_demo.py）
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from robustforge.cli import main  # noqa: E402


def _demo(argv=None) -> int:
    print("=" * 64)
    print("RobustForge 演示：对抗鲁棒性端到端评测")
    print("=" * 64)
    # 复用 CLI 的 run，输出到 examples/benchmark.json
    out = os.path.join(_HERE, "benchmark.json")
    sys.argv = ["robustforge", "run", "--out", out]
    return main()


if __name__ == "__main__":
    raise SystemExit(_demo())
