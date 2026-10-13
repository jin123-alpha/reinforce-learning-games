"""
games/dodger/__init__.py
========================
导出本游戏的注册符号 `GAME`（详见 games/catcher/__init__.py 的说明）。
"""
from .dodger import DodgerGame

GAME = DodgerGame

__all__ = ["GAME", "DodgerGame"]
