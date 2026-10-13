"""
games/__init__.py
=================
**游戏注册表**：扫描 games/ 下的子目录，自动发现游戏。

    games/
    ├── __init__.py      ← 本文件（自动发现 + 兼容重导出）
    ├── catcher/
    │   ├── __init__.py  ← from .catcher import CatcherGame as GAME
    │   └── catcher.py
    └── dodger/
        ├── __init__.py
        └── dodger.py

新增一个游戏只需两步：
    1. 建 games/<name>/ 目录，写一个继承 BaseGame 的类
    2. 在该目录的 __init__.py 里 `GAME = YourGame`
训练/演示代码一行都不用改，也不用回来编辑这个文件。

与旧版的兼容
------------
旧代码写 `from games import BaseGame`。接口契约已经搬到 envs/base.py，
但这里仍然重新导出 BaseGame / run_human，让老写法继续可用。
注意这只是**别名**，真正的定义只有一份（在 envs/base.py）。
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

# 契约的真正定义在 envs/base.py —— 这里只是方便老代码的重新导出
from envs.base import BaseGame, run_human

# ----------------------------------------------------------------------
_GAMES: dict[str, type] = {}
_CACHE: dict[str, object] | None = None


def _discover(force: bool = False) -> dict[str, object]:
    """扫描并导入所有游戏子包，返回 {名字: 游戏类}。

    每个子包必须导出 `GAME`（一个 BaseGame 子类）。
    目录名必须等于该类的 `name` 属性 —— 不相等就直接报错，
    因为 models/<game>/ 和 logs/<game>/ 都是按目录名组织的，
    名字不一致会导致「报告里写 catcher、实际加载 dodger 权重」。
    """
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    found: dict[str, object] = {}
    problems: list[str] = []
    pkg_dir = Path(__file__).resolve().parent

    for info in pkgutil.iter_modules([str(pkg_dir)]):
        folder = info.name
        if not info.ispkg or folder.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"{__name__}.{folder}")
        except Exception as exc:                       # noqa: BLE001
            # 一个游戏坏了不该让整个框架起不来，但要明确报出来
            problems.append(f"  {folder}/ 导入失败：{type(exc).__name__}: {exc}")
            continue

        cls = getattr(module, "GAME", None)
        if cls is None:
            problems.append(
                f"  {folder}/__init__.py 没有导出 GAME"
                f"（应该写 `from .{folder} import YourGame as GAME`）"
            )
            continue

        declared = getattr(cls, "name", None)
        if declared != folder:
            problems.append(
                f"  {folder}/ 的类 {cls.__name__}.name = {declared!r}，"
                f"与目录名 {folder!r} 不一致（必须相同）"
            )
            continue

        found[folder] = cls

    if problems:
        raise RuntimeError(
            "发现游戏时出错：\n" + "\n".join(problems) + "\n"
            "（目录名、类名、GAME 导出三者必须对得上）"
        )

    _CACHE = found
    return found


# ----------------------------------------------------------------------
# 兼容旧接口：原来是模块级常量 GAMES / 函数 list_games / make_game
# ----------------------------------------------------------------------
def __getattr__(attr: str):
    """懒加载：只有真的访问 GAMES 时才去 import 各游戏（会拉起 pygame）。

    这样 `--list-games` 这类只是想看名字的场景可以走
    common/paths.py 的 available_games()（纯目录扫描）而完全不碰 pygame。
    """
    if attr == "GAMES":
        return _discover()
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def list_games() -> list[str]:
    """所有可用游戏名（已排序）。"""
    return sorted(_discover())


def make_game(name: str, **kwargs) -> BaseGame:
    """按名字创建游戏实例。"""
    games = _discover()
    if name not in games:
        raise ValueError(f"未知游戏 {name!r}，可选：{list_games()}")
    cls = games[name]
    if "render_mode" not in kwargs:
        kwargs["render_mode"] = None
    return cls(**kwargs)


def describe(name: str) -> str:
    """一行描述某个游戏的空间信息。"""
    g = make_game(name)
    desc = (f"{name:8s} | 动作 {str(g.action_space):28s} "
            f"| 观测 {str(g.observation_space.shape):10s}")
    g.close()
    return desc


__all__ = [
    "BaseGame", "run_human", "GAMES", "list_games", "make_game", "describe",
]
