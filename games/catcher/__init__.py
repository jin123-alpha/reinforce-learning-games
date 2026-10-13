"""
games/catcher/__init__.py
=========================
导出本游戏的注册符号 `GAME`。

`games/__init__.py` 会自动扫描所有子目录并读取 `GAME`，
所以新增游戏 = 建一个目录 + 导出 GAME，**不需要改任何注册表**。

约定：目录名必须等于 `CatcherGame.name`，由 games/__init__.py 断言校验。
"""
from .catcher import CatcherGame

GAME = CatcherGame

__all__ = ["GAME", "CatcherGame"]
