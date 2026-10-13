"""
envs/game_env.py
================
**薄适配器**：把任意 `BaseGame` 包成标准 Gymnasium 环境。

这一层不含任何游戏逻辑 —— 这正是「游戏与环境完全解耦」的关键：
游戏负责玩法，环境只负责翻译接口。

除了转发，它还负责一件**契约级**的事：**统一 reward / score 口径**。

统一口径保证（第 3 条设计决定）
------------------------------
不管游戏自己怎么写 step()，环境都保证每个 step 返回的 info 里一定有：

    score           游戏自己的得分（= game.score 的当前值）
    episode_return  本回合累计 reward（由环境累加，游戏不用管）
    steps           本回合已走步数

为什么由环境累加而不是让每个游戏自己算：
重构前 play.py 和 compare.py **各自手写**了这个累加循环，
两处逻辑迟早会漂移（比如对 truncation 的处理不一致）。
收敛到环境里之后，只有一份真相。

注意：这里**只统一访问方式，不统一数值尺度**。
catcher 的 score 是接球得分（0~30 量级），dodger 是存活步数（0~1000 量级），
两者本来就不可比。需要跨游戏比较时用 envs/normalize.py 的可选 Wrapper。
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from envs.base import BaseGame


class GameEnv(gym.Env):
    """把一个 BaseGame 包装成 gymnasium.Env，可直接喂给 Stable-Baselines3。"""

    def __init__(self, game: str = "catcher", render_mode: str | None = None,
                 **game_kwargs):
        super().__init__()

        # 延迟 import：envs 顶层不认识任何具体游戏，只认 BaseGame 契约。
        # 具体游戏由 games 包的自动发现提供。
        from games import make_game

        self.game_name = game
        self.render_mode = render_mode
        self.game: BaseGame = make_game(game, render_mode=render_mode, **game_kwargs)

        # 空间直接沿用游戏自己声明的
        self.action_space = self.game.action_space
        self.observation_space = self.game.observation_space
        self.metadata = {
            "render_modes": list(self.game.render_modes),
            "render_fps": self.game.render_fps,
        }

        # ---- 环境自己维护的统计（契约的一部分）----
        self._episode_return = 0.0
        self._steps = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self._episode_return = 0.0
        self._steps = 0
        obs, info = self.game.reset(seed=seed, options=options)
        info = dict(info or {})
        info.setdefault("score", self.score)
        info["episode_return"] = 0.0
        info["steps"] = 0
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        reward = float(reward)

        self._episode_return += reward
        self._steps += 1

        info = dict(info or {})
        info.setdefault("score", self.score)
        # 环境覆盖这两个字段，保证口径唯一（游戏自己填的可能是别的意思）
        info["episode_return"] = self._episode_return
        info["steps"] = self._steps
        info["reward"] = reward

        return (
            np.asarray(obs, dtype=np.float32),
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.game.render()

    def close(self):
        self.game.close()

    # ------------------------------------------------------------------
    @property
    def score(self) -> float:
        """游戏自己的得分。"""
        return self.game.score

    @property
    def episode_return(self) -> float:
        """本回合累计 reward（环境累加，不是游戏提供的）。"""
        return self._episode_return

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def unwrapped_game(self) -> BaseGame:
        return self.game

    def __repr__(self) -> str:
        return (f"<GameEnv game={self.game_name!r} "
                f"obs={self.observation_space.shape} act={self.action_space}>")


def make_env(game: str = "catcher", render_mode: str | None = None, **game_kwargs):
    """快捷函数：`make_env("dodger")` 直接得到一个可训练的环境。"""
    return GameEnv(game=game, render_mode=render_mode, **game_kwargs)
