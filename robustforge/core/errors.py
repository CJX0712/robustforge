"""RobustForge 错误分类（E100–E500）。

设计约定：所有异常继承自 RobustForgeError，携带稳定 code 便于调用方程序化处理。
"""

from __future__ import annotations


class RobustForgeError(Exception):
    """所有 RobustForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class DataError(RobustForgeError):
    """数据相关问题（形状/范围/缺失）。code E100。"""

    code = "E100"


class ModelError(RobustForgeError):
    """模型构建 / 训练 / 推理问题。code E200。"""

    code = "E200"


class AttackError(RobustForgeError):
    """对抗攻击构造问题（范数/步数/越界）。code E300。"""

    code = "E300"


class DefenseError(RobustForgeError):
    """防御 / 训练流程问题。code E400。"""

    code = "E400"


class ConfigError(RobustForgeError):
    """配置 / 环境变量解析问题。code E500。"""

    code = "E500"
