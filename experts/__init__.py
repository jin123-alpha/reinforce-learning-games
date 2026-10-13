"""
experts/__init__.py
===================
专家策略注册表 —— **按游戏一个 py 文件**。

    experts/
    ├── __init__.py     ← 本文件（注册表 + 工厂函数）
    ├── base.py         ← BasePolicy / RandomPolicy 接口
    ├── catcher.py      ← ExpertCatcher（规则 + EDF 调度）
    ├── dodger.py       ← ExpertDodger（滚动时域前瞻避碰）
    └── snake.py        ← ExpertSnake（安全过滤 + 朝食物贪心 + 留退路）

为什么是「一个游戏一个文件」而不是子目录
----------------------------------------
专家策略是**跨游戏对比**用的对照基线，体量远小于游戏本身
（一个游戏的专家通常几十到一两百行）。用扁平文件更好找、
更好 diff，也和 `--game <name>` 一一对应。

对比之下 games/ 用子目录，是因为一个游戏往往有多个文件
（逻辑、渲染、关卡……）。两处的取舍不同，是故意的。

新增专家只需两步：
    1. 写 experts/<game>.py，实现一个 BasePolicy 子类
    2. 在下面的 EXPERTS 里注册一行

前提：该游戏的 `obs_spec()` 必须声明 `kind` 与还原系数 ——
专家是纯 obs→action 的，尺度只能由游戏提供（见 envs/base.py）。

没有专家策略的游戏不会报错，compare.py 会在对应窗口显示占位提示。
"""
from __future__ import annotations

from experts.base import BasePolicy, ObsSpecMismatch, RandomPolicy
from experts.catcher import ExpertCatcher
from experts.dodger import ExpertDodger
from experts.snake import ExpertSnake

# 游戏名 -> 专家策略类
EXPERTS: dict[str, type] = {
    "catcher": ExpertCatcher,
    "dodger": ExpertDodger,
    "snake": ExpertSnake,
}


def has_expert(game_name: str) -> bool:
    return game_name in EXPERTS


def expert_games() -> list[str]:
    return sorted(EXPERTS)


def make_expert(game_name: str, action_space, spec: dict) -> BasePolicy:
    """按游戏名创建专家策略。

    参数刻意设计成 (action_space, spec)，而不是 game 实例 ——
    这样专家拿不到游戏内部状态，只能看见「观测长什么样」这件事，
    从结构上就不可能作弊（详见 experts/base.py 的说明）。
    """
    if game_name not in EXPERTS:
        raise ValueError(
            f"游戏 {game_name!r} 还没有专家策略。已实现：{expert_games()}"
        )
    return EXPERTS[game_name](action_space, spec)


__all__ = [
    "BasePolicy", "RandomPolicy", "ObsSpecMismatch",
    "ExpertCatcher", "ExpertDodger", "ExpertSnake",
    "EXPERTS", "has_expert", "expert_games", "make_expert",
]
