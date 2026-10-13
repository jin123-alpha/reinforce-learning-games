"""
envs/ —— 环境层。

职责（第 1、3 条设计决定）
--------------------------
1. **拥有接口契约**：`envs/base.py` 定义 BaseGame，
   也就是「什么算一个兼容的游戏」。
   依赖方向因此是单向的：games/* -> envs.base，envs 不认识具体游戏。

2. **暴露统一 reward / score 接口**：GameEnv 保证每个 step 的 info 里
   一定有 score / episode_return / steps，口径由环境自己维护，
   游戏不需要（也不应该）各自去算。

3. **可选的跨游戏归一化**：envs/normalize.py 提供 Wrapper，
   只在需要把不同游戏的得分放一起比的时候才用。
"""
from __future__ import annotations

from envs.base import BaseGame, run_human
from envs.game_env import GameEnv, make_env
from envs.normalize import NormalizeScore, REFERENCE_MAX_SCORE, normalize_for

__all__ = [
    "BaseGame",
    "run_human",
    "GameEnv",
    "make_env",
    "NormalizeScore",
    "REFERENCE_MAX_SCORE",
    "normalize_for",
]
