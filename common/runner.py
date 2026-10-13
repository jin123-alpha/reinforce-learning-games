"""
common/runner.py
================
**单策略运行循环** —— play.py 和 compare.py 的子进程共用这一份。

为什么单独抽出来
----------------
重构前 play.py 和 compare.py 各自手写了一遍「reset -> predict -> step ->
累加 return -> 到终止就重置」的循环，两份逻辑对截断、对回合计数的
处理并不完全一致。抽成一份之后：

* play.py   = 加载权重 + 调这里
* compare.py 的每个子进程 = 指定策略 + 调这里

行为天然一致，不会再漂移。

注意本模块**不 import stable_baselines3**：模型加载由调用方负责，
这里只要求传进来的 policy 有 `.predict(obs, deterministic=True)`，
这也是 SB3 模型的接口。所以本模块可以安全地被 human_play 那一路复用
（虽然目前 human_play 走的是更简单的 run_human）。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def run_policy(game, policy, episodes: int = 0,
               seed_base: int = 7000,
               on_episode_end: Callable[[int, dict], None] | None = None,
               before_reset: Callable[[int], None] | None = None,
               max_steps_hint: int = 0) -> dict:
    """让 policy 玩 game，跑到关窗口或跑满 episodes 个回合。

    参数
    ----
    game            BaseGame 实例（游戏层对象，不是 gym 包装）
    policy          任何有 .predict(obs, deterministic=True) 的对象
    episodes        跑多少个回合；0 表示无限（直到窗口关闭）
    seed_base       回合 k 使用 seed_base + k，保证可复现
    on_episode_end  回合结束回调 (回合序号, 统计 dict)
    before_reset    每回合 reset **之前**的回调，用于多进程对齐节奏
                    （compare.py 靠它让三个窗口同时开始同一回合）
    max_steps_hint  仅用于打印预估，不影响逻辑

    返回
    ----
    {"episodes": [...每个回合的统计...], "score_mean":..., "return_mean":...}
    """
    results: list[dict] = []
    ep = 0

    if before_reset is not None:
        before_reset(ep)
    obs, _ = game.reset(seed=seed_base + ep)
    if hasattr(policy, "reset"):
        policy.reset()

    ep_return = 0.0
    steps = 0

    while not getattr(game, "closed", False):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = game.step(int(action))
        game.render()

        ep_return += float(reward)
        steps += 1

        if terminated or truncated:
            ep += 1
            stats = {
                "episode": ep,
                "score": float(getattr(game, "score", 0.0)),
                "return": ep_return,
                "steps": steps,
            }
            results.append(stats)
            if on_episode_end is not None:
                on_episode_end(ep, stats)

            if episodes and ep >= episodes:
                break

            if before_reset is not None:
                before_reset(ep)
            obs, _ = game.reset(seed=seed_base + ep)
            if hasattr(policy, "reset"):
                policy.reset()
            ep_return = 0.0
            steps = 0

    return summarise(results)


def summarise(results: list[dict]) -> dict:
    """把逐回合统计汇总成均值。空结果返回 0，避免 nan。"""
    if not results:
        return {"episodes": [], "n": 0, "score_mean": 0.0,
                "return_mean": 0.0, "steps_mean": 0.0}

    return {
        "episodes": results,
        "n": len(results),
        "score_mean": float(np.mean([r["score"] for r in results])),
        "return_mean": float(np.mean([r["return"] for r in results])),
        "steps_mean": float(np.mean([r["steps"] for r in results])),
    }
