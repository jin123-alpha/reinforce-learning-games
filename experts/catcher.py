"""
experts/catcher.py
==================
接球游戏的**规则专家**（不训练，纯 `obs -> action`）。

思路
----
1. 落点已知：rich 观测里直接给了每颗球的「落地剩余步数 t」和「落点 x」，
   不需要专家自己去外推物理（那是游戏的事）。
2. 可达性判定：球若 k 步后落地，挡板中心最多移动 k*s，
   再算上挡板有效半宽（PADDLE_W/2 + BALL_R）：
       |x_land - 板中心| <= k*s + (PADDLE_W/2 + BALL_R)
3. EDF 调度：在赶得上的球里优先接「最早落地」的那颗。
   （单位收益 + 单位处理时间的调度问题，EDF 是最大化完成数的贪心）
4. 精确瞄准落点**中心** —— 中心得 1.0 分，边缘只有 0.2 分。
5. 兜底：全都赶不上时移到屏幕中央，最小化下一颗球的期望移动距离。

与重构前版本的差别
------------------
旧版构造函数收的是 `game` 对象，直接读 game.WIDTH / game.num_balls 等。
现在只收 `obs_spec()`（游戏声明的尺度）和 `action_space`，
因此它对 catcher 的内部实现**零依赖** —— 改游戏常量不会让它静默算错。

用法
----
    from experts import make_expert
    policy = make_expert("catcher", env.observation_space, game)  # game 只用来取 obs_spec
    action, _ = policy.predict(obs, deterministic=True)
"""
from __future__ import annotations

import numpy as np

from experts.base import BasePolicy, ObsSpecMismatch


class ExpertCatcher(BasePolicy):
    """接球专家：EDF 调度 + 精确瞄准落点中心。"""

    # 动作约定：0=左 1=不动 2=右
    LEFT, STAY, RIGHT = 0, 1, 2

    def __init__(self, action_space, spec: dict):
        super().__init__(action_space)

        if spec.get("kind") != "catcher_rich":
            raise ObsSpecMismatch(
                f"ExpertCatcher 需要 kind='catcher_rich' 的观测说明，"
                f"实际拿到 {spec.get('kind')!r}。"
            )

        self.n = int(spec["n_slots"])
        self.per_ball = int(spec["per_ball"])

        # obs * scale = 物理量（由游戏声明，专家不硬编码）
        self.paddle_x_scale = float(spec["paddle_x_scale"])
        self.paddle_speed_scale = float(spec["paddle_speed_scale"])
        self.x_scale = float(spec["x_scale"])
        self.t_scale = float(spec["t_scale"])
        self.land_x_scale = float(spec["land_x_scale"])

        self.paddle_w = float(spec["paddle_w"])
        self.ball_r = float(spec["ball_r"])
        self.width = float(spec["width"])
        self.t_max = float(spec["t_max"])

        # 派生几何量
        self.effective_half = self.paddle_w / 2.0 + self.ball_r
        self.center_x = self.width / 2.0
        # 挡板速度：obs[1] = paddle_v / paddle_speed，而 paddle_v 只有 -s/0/+s
        # 所以「每步最多移动多少像素」就是 paddle_speed_scale
        self.speed = self.paddle_speed_scale
        self.deadzone = self.speed / 2.0

        self._check_dim(spec)

    # ------------------------------------------------------------------
    def _check_dim(self, spec) -> None:
        """自检：动作空间必须是 Discrete(3)，观测维度必须与声明一致。

        这类错误拖到运行时只表现为「得分很低」，排查成本极高，
        所以构造时就把能查的查掉。
        """
        n_actions = getattr(self.action_space, "n", None)
        if n_actions != 3:
            raise ObsSpecMismatch(
                f"ExpertCatcher 需要 Discrete(3) 动作空间（左/不动/右），"
                f"实际是 {self.action_space!r}。"
            )

        self._expected_obs_dim = 2 + self.per_ball * self.n

        declared_dim = spec.get("obs_dim")
        if declared_dim is not None and int(declared_dim) != self._expected_obs_dim:
            raise ObsSpecMismatch(
                f"obs_spec 声明观测共 {declared_dim} 维，"
                f"但 2 + per_ball({self.per_ball}) × n_slots({self.n}) "
                f"= {self._expected_obs_dim}，两者不一致。"
            )

    # ------------------------------------------------------------------
    def _unpack(self, obs):
        """把观测还原成物理量。"""
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if obs.size != self._expected_obs_dim:
            raise ObsSpecMismatch(
                f"观测维度是 {obs.size}，但 obs_spec 声明的是 "
                f"{self._expected_obs_dim}（2 + {self.per_ball}×{self.n}）。"
                f"通常是因为 rich_obs 与 num_balls 变了却没同时更新专家。"
            )

        paddle_x = obs[0] * self.paddle_x_scale
        center = paddle_x + self.paddle_w / 2.0

        blk = obs[2:].reshape(self.n, self.per_ball)
        t = blk[:, 4] * self.t_scale
        x_land = blk[:, 5] * self.land_x_scale
        return center, t, x_land

    # ------------------------------------------------------------------
    def _choose_target(self, center: float, t: np.ndarray,
                       x_land: np.ndarray) -> float:
        """EDF：在赶得上的球里选最早落地的，返回目标 x。"""
        # 还有 k 步可用（至少 1 步，避免 0 步时的除零/边界问题）
        k = np.maximum(np.ceil(t - 1e-9), 1.0)

        reachable = (np.abs(x_land - center) <= k * self.speed + self.effective_half)
        # t 很小（已经贴地）的球当作已处理完，不再作为目标
        reachable &= (t > 1e-9)

        if reachable.any():
            idx = np.where(reachable)[0]
            # 主序：最早落地；次序：同样紧急时选更近的
            order = np.lexsort((np.abs(x_land[idx] - center), k[idx]))
            return float(x_land[idx[order[0]]])

        # 全都赶不上 -> 回中，最小化下一颗球的期望移动距离
        return self.center_x

    def act(self, obs) -> int:
        center, t, x_land = self._unpack(obs)
        target = self._choose_target(center, t, x_land)

        delta = target - center
        if delta > self.deadzone:
            return self.RIGHT
        if delta < -self.deadzone:
            return self.LEFT
        return self.STAY

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """与 SB3 模型签名一致：返回 (action, state)。"""
        return np.array(self.act(obs), dtype=np.int64), state
