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
    ) -> None:
        assert render_mode in self.render_modes or render_mode is None
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.player_speed = float(player_speed if player_speed is not None else self.PLAYER_SPEED)
        self.spawn_every = float(spawn_every if spawn_every is not None else self.SPAWN_EVERY)

        self.action_space = spaces.Discrete(3)   # 0=左 1=不动 2=右
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1 + 3 * self.N_VISIBLE,), dtype=np.float32
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
        obs = np.full(1 + 3 * self.N_VISIBLE, self.EMPTY, dtype=np.float32)
        obs[0] = self.player_x / self.WIDTH
        if len(self.obst):
            # 只保留最靠近玩家（y 最大）的 N 个
            order = np.argsort(-self.obst[:, 1])[: self.N_VISIBLE]
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