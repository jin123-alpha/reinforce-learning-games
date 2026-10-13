"""
experts/base.py
===============
专家策略的统一接口。

设计决定（第 4 条）
------------------
**专家是纯 `obs -> action` 的函数，不接触 game 实例。**

    predict(obs, state=None, episode_start=None, deterministic=True)
        -> (action, state)

这与 Stable-Baselines3 模型的接口**完全一致**，所以

    RandomPolicy / ExpertPolicy / SB3 model

三者在 play.py / compare.py 里是同一种东西（都只有 .predict），
调用方不需要任何 if-else 分支 —— 这正是三窗口对比能写得很干净的原因。

为什么不让专家碰 game 对象
--------------------------
重构前的 ExpertCatcher 直接读 game.WIDTH / game.num_balls /
game.paddle_speed，还调用 game.landing()。后果有两个：

1. 游戏改内部常量 -> 专家静默失效（算出的落点是错的，但不报错）；
2. 专家能读到观测里没有的状态 -> 它就不是「策略」而是「神谕」，
   拿来当基线会高估。

现在改成：

* 观测里**本来就有**的信息（落点 t、落点 x），专家直接用；
* 观测里没直接给但可以换算的（比如板中心），由游戏通过
  `obs_spec()` **声明尺度**，专家据此还原。

这样专家仍然只吃 obs，但尺度由游戏提供，改常量不会让它算错。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """所有策略（专家 / 随机 / 模型包装）的统一接口。"""

    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """返回 (action, state)，与 SB3 模型签名一致。"""

    def __call__(self, obs):
        """方便直接把策略当函数用：action = policy(obs)"""
        action, _ = self.predict(obs, deterministic=True)
        return action

    def reset(self):
        """回合开始时的钩子（无状态的策略不需要实现）。"""


class RandomPolicy(BasePolicy):
    """随机策略 —— 任何游戏通用的下界参考，也是三窗口里最左边那一个。"""

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.action_space.sample(), state


class ObsSpecMismatch(ValueError):
    """专家拿到的 obs_spec 与自己的实现不匹配。"""
