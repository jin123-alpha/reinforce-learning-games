"""
games/dodger/dodger.py
======================
躲避游戏（Dodger）—— 用来证明「游戏可插拔」。

它和接球游戏**结构完全不同**：
  * 动作：Discrete(3) 但语义不同（左右闪避）
  * 观测：1 个玩家特征 + 最近 4 个障碍物的 3 个特征 = 13 维
    而且障碍物数量是动态的，需要「只保留最近的 N 个，空位用哨兵值填充」
  * 奖励：每存活一步 +1，撞到障碍物回合结束（典型的「生存型」任务）
  * 画面：方块而不是圆/渐变

    python -m games.dodger.dodger            # 直接人类游玩
"""
from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from envs.base import BaseGame, run_human


class DodgerGame(BaseGame):
    name = "dodger"

    WIDTH = 400
    HEIGHT = 600

    PLAYER_W = 40
    PLAYER_H = 18
    PLAYER_Y = HEIGHT - 40
    PLAYER_SPEED = 7.0

    OBST_W = 34
    OBST_H = 18
    OBST_VY_RANGE = (4.0, 7.5)
    SPAWN_EVERY = 22.0          # 平均每多少步生成一个障碍物
    MAX_OBSTACLES = 12

    N_VISIBLE = 4               # 观测里保留最近的几个障碍物
    EMPTY = -1.0                # 空位哨兵值（真实特征都在 [-1, 1] 内的 [0,1] 段）

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 1000,
        player_speed: float | None = None,
        spawn_every: float | None = None,
        n_visible: int | None = None,
    ) -> None:
        assert render_mode in self.render_modes or render_mode is None
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.player_speed = float(player_speed if player_speed is not None else self.PLAYER_SPEED)
        self.spawn_every = float(spawn_every if spawn_every is not None else self.SPAWN_EVERY)
        # 观测里保留几个障碍物槽位。做成参数是为了让专家能在「看得更远」
        # 的配置下被验证；默认值必须与旧版一致（4），否则观测维度变化会
        # 让已训练的权重失效（load 时报维度不匹配）。
        self.n_visible = int(n_visible if n_visible is not None else self.N_VISIBLE)
        assert self.n_visible >= 1

        self.action_space = spaces.Discrete(3)   # 0=左 1=不动 2=右
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1 + 3 * self.n_visible,), dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.font = None
        self._pygame_ready = False
        self.closed = False

        self.player_x = 0.0
        self.obst = np.zeros((0, 3))      # 每行: [x, y, vy]
        self.steps = 0
        self.dodged = 0
        self._rng = np.random.default_rng()

    # ================= 逻辑 =================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.dodged = 0
        self.obst = np.zeros((0, 3))
        self.player_x = (self.WIDTH - self.PLAYER_W) / 2.0
        return self._get_obs(), self._get_info()

    def _spawn(self):
        x = self._rng.uniform(0.0, self.WIDTH - self.OBST_W)
        vy = self._rng.uniform(*self.OBST_VY_RANGE)
        self.obst = np.vstack([self.obst, [x, -self.OBST_H, vy]])

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"非法动作: {action}"
        self.steps += 1

        # 1. 玩家移动
        dx = {0: -self.player_speed, 1: 0.0, 2: self.player_speed}[action]
        self.player_x = float(
            np.clip(self.player_x + dx, 0.0, self.WIDTH - self.PLAYER_W)
        )

        # 2. 障碍物下落
        if len(self.obst):
            self.obst[:, 1] += self.obst[:, 2]

        # 3. 随机生成新障碍物
        if self._rng.random() < 1.0 / self.spawn_every and len(self.obst) < self.MAX_OBSTACLES:
            self._spawn()

        # 4. 碰撞检测（AABB）
        reward = 1.0                 # 存活奖励
        terminated = False
        if len(self.obst):
            px0, px1 = self.player_x, self.player_x + self.PLAYER_W
            py0, py1 = self.PLAYER_Y, self.PLAYER_Y + self.PLAYER_H
            ox = self.obst[:, 0]
            oy = self.obst[:, 1]
            hit = (ox < px1) & (ox + self.OBST_W > px0) & (oy < py1) & (oy + self.OBST_H > py0)
            if hit.any():
                terminated = True
                reward = 0.0         # 撞了就结束，拿不到这一步的存活奖励

        # 5. 移除已经出屏的障碍物
        if len(self.obst):
            before = len(self.obst)
            keep = self.obst[:, 1] < self.HEIGHT
            self.obst = self.obst[keep]
            self.dodged += int(before - len(self.obst))

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ================= 观测 =================
    def _get_obs(self) -> np.ndarray:
        obs = np.full(1 + 3 * self.n_visible, self.EMPTY, dtype=np.float32)
        obs[0] = self.player_x / self.WIDTH
        if len(self.obst):
            # 只保留最靠近玩家（y 最大）的 N 个
            order = np.argsort(-self.obst[:, 1])[: self.n_visible]
            for k, i in enumerate(order):
                o = 1 + 3 * k
                obs[o + 0] = self.obst[i, 0] / self.WIDTH
                obs[o + 1] = self.obst[i, 1] / self.HEIGHT
                obs[o + 2] = self.obst[i, 2] / self.OBST_VY_RANGE[1]
        return obs

    def _get_info(self) -> dict:
        return {
            "score": float(self.steps),
            "steps": self.steps,
            "dodged": self.dodged,
            "obstacles": len(self.obst),
        }

    # ================= 观测说明（给专家策略用） =================
    def obs_spec(self) -> dict:
        """声明观测分量的物理尺度，让纯 obs 专家能还原真实物理量。

        观测布局（1 维玩家 + n_visible 个障碍物 × 3 维）：

            obs[0]           玩家左上角 x  / WIDTH
            第 k 个槽位从 1 + 3*k 开始（k = 0 是**最靠近玩家**的那个）：
                +0  障碍物左上角 x  * WIDTH
                +1  障碍物左上角 y  * HEIGHT
                +2  下落速度 vy      * OBST_VY_RANGE[1]
            空槽位整块等于 EMPTY（-1.0）

        与 catcher 的 obs_spec 一样，这里给的全部是
        「obs * scale = 物理量」的系数，专家据此还原，不硬编码常量。

        注意槽位顺序（按 y 从大到小）本身就是重要信息：
        专家需要知道「哪个障碍物最危险」，而观测里只有这 n_visible 个，
        看不到更远的 —— 所以专家是**合法的**对照基线，不是神谕。
        """
        return {
            "kind": "dodger",
            "n_slots": self.n_visible,
            "per_obstacle": 3,
            "obs_dim": 1 + 3 * self.n_visible,
            # 玩家
            "player_x_scale": float(self.WIDTH),
            "player_speed_scale": float(self.player_speed),
            # 障碍物
            "x_scale": float(self.WIDTH),
            "y_scale": float(self.HEIGHT),
            "vy_scale": float(self.OBST_VY_RANGE[1]),
            # 用于碰撞判定的几何量
            "player_w": float(self.PLAYER_W),
            "player_h": float(self.PLAYER_H),
            "player_y": float(self.PLAYER_Y),
            "obst_w": float(self.OBST_W),
            "obst_h": float(self.OBST_H),
            "width": float(self.WIDTH),
            "height": float(self.HEIGHT),
            "empty": float(self.EMPTY),
        }

    # ================= 得分 =================
    @property
    def score(self) -> float:
        return float(self.steps)

    @property
    def score_text(self) -> str:
        return f"Survived: {self.steps}  Dodged: {self.dodged}"

    # ================= 人类操控 =================
    @property
    def action_hint(self) -> str:
        return "← / A 左移      → / D 右移"

    def keys_to_action(self, keys):
        import pygame
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 2
        return 1

    # ================= 渲染 =================
    def render(self):
        if self.closed:
            return None
        import pygame
        self._init_pygame()
        canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        canvas.fill((16, 18, 28))

        for x in range(0, self.WIDTH + 1, 50):
            pygame.draw.line(canvas, (26, 30, 44), (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT + 1, 50):
            pygame.draw.line(canvas, (26, 30, 44), (0, y), (self.WIDTH, y))

        # 障碍物
        for ox, oy, _ in self.obst:
            r = pygame.Rect(int(ox), int(oy), self.OBST_W, self.OBST_H)
            pygame.draw.rect(canvas, (235, 70, 80), r, border_radius=4)
            pygame.draw.rect(canvas, (255, 190, 190), r, width=2, border_radius=4)

        # 玩家
        pr = pygame.Rect(int(self.player_x), self.PLAYER_Y, self.PLAYER_W, self.PLAYER_H)
        pygame.draw.rect(canvas, (60, 220, 240), pr, border_radius=5)
        pygame.draw.rect(canvas, (235, 255, 255), pr, width=2, border_radius=5)

        line = self.font.render(
            f"Survived: {self.steps}   Dodged: {self.dodged}", True, (255, 255, 255))
        canvas.blit(line, (12, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None

        raw = pygame.image.tobytes(canvas, "RGB") if hasattr(pygame.image, "tobytes") \
            else pygame.image.tostring(canvas, "RGB")
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.HEIGHT, self.WIDTH, 3).copy()

    # ================= 显示初始化 =================
    @property
    def display_caption(self) -> str:
        return "Dodger"

    def _init_font(self):
        """字体由游戏自己创建（各游戏字号不同）。"""
        import pygame
        self.font = pygame.font.Font(None, 26)

    def _init_pygame(self):
        """兼容旧调用点：直接委托给基类的 init_display()。"""
        self.init_display()

    def close(self):
        import pygame
        if self._pygame_ready:
            pygame.display.quit()
            pygame.quit()
            self._pygame_ready = False
        self.window = None
        self.clock = None
        self.closed = True


if __name__ == "__main__":
    run_human(DodgerGame(render_mode="human"))