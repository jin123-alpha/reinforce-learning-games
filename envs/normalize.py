"""
envs/normalize.py
=================
**可选的**得分归一化 Wrapper —— 只在需要跨游戏比较时才用。

背景
----
envs 只统一得分的**访问方式**，不统一**数值尺度**：

    catcher  score = 接球得分      典型 0 ~ 30
    dodger   score = 存活步数      典型 0 ~ 1000

同一个游戏内的三种策略（随机 / 模型 / 专家）比大小是公平的，
因为尺度相同 —— 所以 compare.py 的三窗口**不需要**这个 Wrapper。

但如果你想在同一张图里比「catcher 的专家」和「dodger 的专家」谁更强，
就必须先归一化到 [0, 1]。这就是本模块存在的唯一理由。

为什么不做进主契约
------------------
归一化需要每个游戏声明一个 `max_score`。catcher 在理论上
只要不失误就能无限接球（回合结束条件是失误次数，不是得分上限），
所以它的 max_score 只能人工拍一个 —— 这种「靠估」的数字
不适合放进框架的强制契约里。

用法
----
    from envs import make_env, NormalizeScore
    env = NormalizeScore(make_env("catcher"), max_score=30.0)
    ...
    info["score_normalized"]   # 0.0 ~ 1.0
"""
from __future__ import annotations

import gymnasium as gym


class NormalizeScore(gym.Wrapper):
    """把 info['score'] 线性映射到 [0, 1]，附加为 score_normalized。

    不改变 reward、不改变 observation，只是**多提供一个字段**。
    这样训练行为完全不变，只有展示/比较逻辑受影响。
    """

    def __init__(self, env: gym.Env, max_score: float, min_score: float = 0.0):
        super().__init__(env)
        if max_score <= min_score:
            raise ValueError(
                f"max_score({max_score}) 必须大于 min_score({min_score})"
            )
        self.min_score = float(min_score)
        self.max_score = float(max_score)

    # ------------------------------------------------------------------
    def _normalize(self, info: dict) -> dict:
        raw = info.get("score", 0.0)
        span = self.max_score - self.min_score
        value = (float(raw) - self.min_score) / span
        info["score_raw"] = float(raw)
        info["score_normalized"] = max(0.0, min(1.0, value))
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._normalize(dict(info or {}))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._normalize(dict(info or {}))


# ----------------------------------------------------------------------
# 各游戏给跨游戏比较用的参考上限（**仅供展示**，不是框架契约）
# ----------------------------------------------------------------------
# catcher：max_misses=10 时，专家实测约 17~25 分，取 30 作为参考上限。
# dodger ：回合上限 max_steps=1000，活满即满分。
# snake  ：吃满整个网格需要吃掉 n_cells - initial_len 颗食物，
#          所以 16×16 - 3 = 253 是理论满分（专家实测约 24）。
REFERENCE_MAX_SCORE = {
    "catcher": 30.0,
    "dodger": 1000.0,
    "snake": 253.0,
}


def normalize_for(game: str, env: gym.Env) -> gym.Env:
    """按游戏名套上参考归一化（没有参考值就原样返回）。"""
    if game not in REFERENCE_MAX_SCORE:
        return env
    return NormalizeScore(env, max_score=REFERENCE_MAX_SCORE[game])
