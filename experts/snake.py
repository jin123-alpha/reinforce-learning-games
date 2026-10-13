"""
experts/snake.py
================
贪吃蛇的**规则专家**（不训练，纯 `obs -> action`）。

问题形态
--------
snake 与前两个游戏的专家都不一样：

  * 它需要**多步规划**（绕过自己的身体去够食物），而不只是「一步反应」。
    catcher 只需解一个落点方程，dodger 只需前向若干步避碰。
  * 状态里有**自己造成的历史**（蛇身）：走一条现在安全的路，
    可能把蛇盘进死角，几步之后自己把自己困死。
    所以"安全"是相对的，必须区分「这一步不撞」和「走完还有退路」。

思路（先保命，再吃饭，最后留后路）
----------------------------------
1. **安全过滤**：先排除立刻撞墙/撞身体的动作。
   观测里直接给了三个相对方向（直行/右转/左转）的危险度，
   专家读它即可，不必自己重建网格。

2. **能吃到就吃**：某个安全方向的食物标记为 `eat_value`（+1.0）时直接选它。
   食物是唯一的正奖励来源，机会不该放过。

3. **朝食物靠近**：在安全动作里挑「让到食物距离变小」的（标记 +0.5）。
   这一步用观测的离散标记，不需要还原曼哈顿距离。

4. **留退路（专家与"傻瓜贪心"的分水岭）**：
   只做 1~3 的贪心会在蛇变长后绕着障碍打转、把自己盘死。
   所以在候选中再用两个保守项微调：
     * **少转向**：转向本身会把蛇头引向新区域、也可能撞上刚长出来的身体；
       直线前进让蛇身自然展开成一条线，最不容易自困。
     * **保持开阔**：当前三方向危险度之和越小，说明周围越空；
       优先留在开阔处，等价于"别往角落里钻"。
   这两项的权重随**蛇长占比**上升 —— 蛇越长越容易被自己困住。

信息边界（为什么这个专家合法、不是神谕）
----------------------------------------
观测里只有「三方向危险度 + 食物相对档位 + 食物偏移 + 蛇长」，
**没有完整网格**。所以专家无法做真正的 BFS 到食物，只能基于这些分量
做启发式 —— 这正是它能当**合法对照基线**的原因。
真要看"完美规划"的上界，得另写一个能读游戏内部状态的 Oracle，
而那就不叫策略了。

用法
----
    from experts import make_expert
    policy = make_expert("snake", env.action_space, game.obs_spec())
    action, _ = policy.predict(obs, deterministic=True)
"""
from __future__ import annotations

import numpy as np

from experts.base import BasePolicy, ObsSpecMismatch

# 动作语义（相对转向），与 games/snake/snake.py 保持一致
STRAIGHT, TURN_RIGHT, TURN_LEFT, TURN_BACK = 0, 1, 2, 3
# 相对方向下标 k 对应的动作编号：k=0 直行，k=1 右转，k=2 左转
_K_TO_ACTION = (STRAIGHT, TURN_RIGHT, TURN_LEFT)


class ExpertSnake(BasePolicy):
    """贪吃蛇专家：安全过滤 + 进食优先 + 朝食物贪心 + 留退路。"""

    def __init__(self, action_space, spec: dict):
        super().__init__(action_space)

        if spec.get("kind") != "snake":
            raise ObsSpecMismatch(
                f"ExpertSnake 需要 kind='snake' 的观测说明，"
                f"实际拿到 {spec.get('kind')!r}。"
            )

        self.obs_dim = int(spec["obs_dim"])
        self.rel_dirs = tuple(spec["rel_dirs"])       # (0, 1, -1)
        self.relative_actions = bool(spec["relative_actions"])
        if not self.relative_actions:
            raise ObsSpecMismatch(
                "ExpertSnake 的实现基于**相对转向**的动作语义"
                "（直行/右转/左转），当前游戏用的是绝对方向动作"
                "（relative_actions=False），专家需要另写一版。"
            )

        self.grid_w = int(spec["grid_w"])
        self.grid_h = int(spec["grid_h"])
        self.n_cells = int(spec["n_cells"])

        self.length_scale = float(spec["length_scale"])
        self.food_dx_scale = float(spec["food_dx_scale"])
        self.food_dy_scale = float(spec["food_dy_scale"])

        self.eat_value = float(spec.get("eat_value", 1.0))
        self.closer_value = float(spec.get("closer_value", 0.5))

        # 蛇长超过网格的这个比例后，开始显著保守
        self.caution_start = 0.18
        self.caution_full = 0.55

        self._check_dim()

    # ------------------------------------------------------------------
    def _check_dim(self) -> None:
        n_actions = getattr(self.action_space, "n", None)
        if n_actions != 4:
            raise ObsSpecMismatch(
                f"ExpertSnake 需要 Discrete(4) 动作空间（直行/右转/左转/掉头），"
                f"实际是 {self.action_space!r}。"
            )

        expected = 2 + 2 * len(self.rel_dirs) + 3
        if self.obs_dim != expected:
            raise ObsSpecMismatch(
                f"obs_spec 声明观测 {self.obs_dim} 维，"
                f"但 2 + 2×{len(self.rel_dirs)} + 3 = {expected}，两者不一致。"
            )

    # ------------------------------------------------------------------
    def _unpack(self, obs):
        """还原专家需要的分量。

        返回 (danger, food_signal, length_frac, fdx, fdy)：
            danger       每个相对方向的危险度 (3,)，>=0.5 视为会撞
            food_signal  每个相对方向的进食/靠近标记 (3,)
            length_frac  蛇长 / 格子总数
            fdx, fdy     食物相对蛇头的格数偏移（已乘回网格尺度）
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if obs.size != self.obs_dim:
            raise ObsSpecMismatch(
                f"观测维度是 {obs.size}，但 obs_spec 声明的是 {self.obs_dim}。"
                f"通常是因为游戏的观测布局改了却没同步更新专家。"
            )

        n = len(self.rel_dirs)
        block = obs[2:2 + 2 * n].reshape(n, 2)
        danger = block[:, 0]
        food_signal = block[:, 1]
        tail = obs[2 + 2 * n:]
        length_frac = float(tail[0]) * self.length_scale / max(self.n_cells, 1)
        fdx = float(tail[1]) * self.food_dx_scale
        fdy = float(tail[2]) * self.food_dy_scale
        return danger, food_signal, length_frac, fdx, fdy

    # ------------------------------------------------------------------
    def act(self, obs) -> int:
        danger, food_signal, length_frac, fdx, fdy = self._unpack(obs)
        n = len(self.rel_dirs)

        # ---- 1. 安全过滤 ----
        safe = [k for k in range(n) if danger[k] < 0.5]

        if not safe:
            # 三个相对方向全都会撞。此时唯一还有希望的是**掉头** ——
            # 观测里没给掉头的危险度（因为正常策略不该掉头），
            # 但"必死"的时刻值得赌一把：掉头至少不在已知的碰撞方向上。
            # 这是兜底分支，正常情况下不会触发。
            return TURN_BACK

        # ---- 2. 能吃到就吃 ----
        eaters = [k for k in safe if food_signal[k] >= self.eat_value - 1e-6]
        if eaters:
            # 优先直行（少转向 = 少风险）；否则取第一个可吃的方向
            return STRAIGHT if STRAIGHT in eaters else _K_TO_ACTION[eaters[0]]

        # ---- 3+4. 朝食物靠近 + 留退路 ----
        # 保守程度：蛇越长越容易被自己困住，所以要更"稳"
        caution = (length_frac - self.caution_start) / (
            self.caution_full - self.caution_start)
        caution = float(np.clip(caution, 0.0, 1.0))

        # 当前开阔度：三方向危险度之和越小越开阔（0=完全空旷）
        openness = 1.0 - float(danger.sum()) / max(n, 1)

        best_action, best_key = _K_TO_ACTION[safe[0]], None
        for k in safe:
            # 主判据：是否朝食物靠近
            approaching = 1.0 if food_signal[k] >= self.closer_value - 1e-6 else 0.0
            # 距食物更远的方向在必要时也要走（绕行），所以给一个小底分，
            # 而不是直接排除 —— 排除会让蛇在食物被身体挡住时无路可走。
            heading = approaching

            straightness = 1.0 if k == 0 else 0.0

            # 字典序：先最大化 heading，再用保守项做平局裁决。
            # 用一个标量 key 表达两层优先级（heading 权重远大于保守项）。
            key = -(heading * 10.0
                    + caution * (0.9 * straightness + 0.6 * openness))
            if best_key is None or key < best_key:
                best_action, best_key = _K_TO_ACTION[k], key

        return best_action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """与 SB3 模型签名一致：返回 (action, state)。"""
        return np.array(self.act(obs), dtype=np.int64), state
