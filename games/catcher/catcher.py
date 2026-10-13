"""
games/catcher/catcher.py
========================
接球小游戏（Catcher）—— 一个完全独立的 pygame 游戏。

**它不知道强化学习的存在**，只实现 `envs/base.py` 的 `BaseGame` 契约：

    python -m games.catcher.catcher        # 直接人类游玩（方向键 / A D）

契约住在 envs/ 而不是 games/ 是刻意的：envs 定义「什么算兼容游戏」，
games 只负责实现它，所以依赖方向是单向的 games -> envs.base。

玩法
----
* 挡板在屏幕底部左右移动，移动速度受限
* 多颗小球同时下落，碰到左右墙会反弹
* 接住小球得分，且落点越靠挡板中心分越高（挡板红->绿渐变标出得分区）
* 漏接扣分，失误达到 max_misses 回合结束

得分
----
    offset  = |球心x - 挡板中心x| / (PADDLE_W/2 + BALL_R)
    quality = (1 - offset) ** hit_falloff_power
    reward  = hit_reward_min + (hit_reward_max - hit_reward_min) * quality
    漏接    = -miss_penalty
"""
from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium import spaces

# ---- 兼容「作为包导入」与「直接运行本文件」两种情况 ----
if __package__ in (None, ""):
    # python games/catcher/catcher.py -> 把项目根加进 sys.path，
    # 这样才能 import envs.base
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from envs.base import BaseGame, run_human


class CatcherGame(BaseGame):
    name = "catcher"

    # ---------------- 画面 / 物理常量 ----------------
    WIDTH = 400
    HEIGHT = 600

    PADDLE_W = 90
    PADDLE_H = 14
    PADDLE_Y = HEIGHT - 40

    BALL_R = 10
    BALL_VY_RANGE = (5.0, 9.0)      # 下落速度(像素/步)
    BALL_VX_RANGE = (-3.0, 3.0)     # 水平速度(像素/步)
    SPAWN_BAND = 300.0              # 小球在屏幕上方多大的范围内等待出场
    T_MAX = 200.0                   # 落点剩余步数的归一化上限

    def __init__(
        self,
        render_mode: str | None = None,
        num_balls: int = 3,
        paddle_speed: float = 5.0,
        max_misses: int = 10,
        max_steps: int = 1000,
        hit_reward_min: float = 0.2,
        hit_reward_max: float = 1.0,
        hit_falloff_power: float = 2.0,
        miss_penalty: float = 1.0,
        rich_obs: bool = True,
    ) -> None:
        assert render_mode in self.render_modes or render_mode is None
        assert num_balls >= 1

        self.render_mode = render_mode
        self.num_balls = int(num_balls)
        self.paddle_speed = float(paddle_speed)
        self.max_misses = int(max_misses)
        self.max_steps = int(max_steps)
        self.hit_reward_min = float(hit_reward_min)
        self.hit_reward_max = float(hit_reward_max)
        self.hit_falloff_power = float(hit_falloff_power)
        self.miss_penalty = float(miss_penalty)
        self.rich_obs = bool(rich_obs)

        # ---- 给模型的空间定义 ----
        self.action_space = spaces.Discrete(3)          # 0=左 1=不动 2=右
        self.per_ball_dim = 7 if self.rich_obs else 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 + self.per_ball_dim * self.num_balls,),
            dtype=np.float32,
        )

        # ---- 渲染资源（延迟初始化）----
        self.window = None
        self.clock = None
        self.font = None
        self.hint_font = None
        self._pygame_ready = False
        self.closed = False

        # ---- 状态 ----
        n = self.num_balls
        self.paddle_x = 0.0
        self.paddle_v = 0.0
        self.ball_x = np.zeros(n)
        self.ball_y = np.zeros(n)
        self.ball_vx = np.zeros(n)
        self.ball_vy = np.zeros(n)
        self._score = 0.0
        self.hits = 0
        self.misses = 0
        self.steps = 0
        self._rng = np.random.default_rng()

    # ================= 游戏逻辑 =================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self._score = 0.0
        self.hits = 0
        self.misses = 0

        self.paddle_x = (self.WIDTH - self.PADDLE_W) / 2.0
        self.paddle_v = 0.0

        # 分层采样：把出场区间切段，让几颗球错开落下
        seg = self.SPAWN_BAND / self.num_balls
        for i in range(self.num_balls):
            y = -self.BALL_R - (i * seg + self._rng.uniform(0.0, seg * 0.7))
            self._spawn_ball(i, y)

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"非法动作: {action}"

        self.steps += 1
        reward = 0.0

        # 1. 移动挡板（速度受限）
        dx = {0: -self.paddle_speed, 1: 0.0, 2: self.paddle_speed}[action]
        self.paddle_v = dx
        self.paddle_x = float(
            np.clip(self.paddle_x + dx, 0.0, self.WIDTH - self.PADDLE_W)
        )

        # 2. 所有球同时运动
        prev_bottom = self.ball_y + self.BALL_R
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        left = self.ball_x - self.BALL_R < 0.0
        right = self.ball_x + self.BALL_R > self.WIDTH
        self.ball_x[left] = self.BALL_R
        self.ball_vx[left] = np.abs(self.ball_vx[left])
        self.ball_x[right] = self.WIDTH - self.BALL_R
        self.ball_vx[right] = -np.abs(self.ball_vx[right])

        # 3. 判定：本步刚跨过挡板平面
        bottom = self.ball_y + self.BALL_R
        crossed = (prev_bottom < self.PADDLE_Y) & (bottom >= self.PADDLE_Y) & (self.ball_vy > 0)
        in_x = (self.ball_x >= self.paddle_x - self.BALL_R) & (
            self.ball_x <= self.paddle_x + self.PADDLE_W + self.BALL_R
        )
        caught = crossed & in_x
        missed = (self.ball_y - self.BALL_R > self.HEIGHT) & ~caught

        # 4. 得分：越靠中心分越高
        if caught.any():
            center = self.paddle_x + self.PADDLE_W / 2.0
            half = self.PADDLE_W / 2.0 + self.BALL_R
            offset = np.clip(np.abs(self.ball_x[caught] - center) / half, 0.0, 1.0)
            quality = (1.0 - offset) ** self.hit_falloff_power
            gains = self.hit_reward_min + (
                self.hit_reward_max - self.hit_reward_min
            ) * quality
            reward += float(gains.sum())
            self._score += float(gains.sum())
            self.hits += int(caught.sum())

        if missed.any():
            reward -= self.miss_penalty * float(missed.sum())
            self.misses += int(missed.sum())

        for i in np.where(caught | missed)[0]:
            self._spawn_ball(int(i))

        terminated = self.misses >= self.max_misses
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ================= 落点预测（内部物理） =================
    def _landing(self):
        """预测每颗球落到挡板平面所需步数 t 与该时刻的 x（含墙面反弹）。

        这是游戏的**内部**物理，只服务于 _get_obs 构造 rich 观测。
        专家策略不再调用它 —— 专家是纯 obs->action，不接触 game 对象，
        需要的落点信息已经包含在观测里（t_land / x_land 两个分量）。
        这样做的好处：专家不可能「偷看」游戏内部状态，
        因此它是一个合法的对照基线，而不是作弊神谕。
        """
        n = self.num_balls
        t = np.full(n, self.T_MAX)
        falling = self.ball_vy > 0
        if falling.any():
            dist = self.PADDLE_Y - self.BALL_R - self.ball_y
            t[falling] = np.maximum(0.0, dist[falling] / self.ball_vy[falling])
        t = np.minimum(t, self.T_MAX)

        span = self.WIDTH - 2.0 * self.BALL_R
        pos = self.ball_x - self.BALL_R
        wrapped = np.mod(pos + self.ball_vx * t, 2.0 * span)
        x_land = np.where(wrapped > span, 2.0 * span - wrapped, wrapped) + self.BALL_R
        return t, x_land

    # ================= 内部 =================
    def _spawn_ball(self, i: int, y: float | None = None) -> None:
        rng = self._rng
        self.ball_x[i] = rng.uniform(self.BALL_R, self.WIDTH - self.BALL_R)
        self.ball_vx[i] = rng.uniform(*self.BALL_VX_RANGE)
        self.ball_vy[i] = rng.uniform(*self.BALL_VY_RANGE)
        if y is None:
            y = -self.BALL_R - rng.uniform(0.0, self.SPAWN_BAND)
        self.ball_y[i] = float(y)

    def _get_obs(self) -> np.ndarray:
        obs = np.empty(2 + self.per_ball_dim * self.num_balls, dtype=np.float32)
        obs[0] = self.paddle_x / (self.WIDTH - self.PADDLE_W)
        obs[1] = self.paddle_v / self.paddle_speed if self.paddle_speed > 0 else 0.0

        vx_scale = max(abs(self.BALL_VX_RANGE[0]), abs(self.BALL_VX_RANGE[1]))
        vy_scale = max(abs(self.BALL_VY_RANGE[0]), abs(self.BALL_VY_RANGE[1]))

        if self.rich_obs:
            t_land, x_land = self._landing()
            paddle_center = self.paddle_x + self.PADDLE_W / 2.0
            for i in range(self.num_balls):
                o = 2 + 7 * i
                obs[o + 0] = self.ball_x[i] / self.WIDTH
                obs[o + 1] = self.ball_y[i] / self.HEIGHT
                obs[o + 2] = self.ball_vx[i] / vx_scale
                obs[o + 3] = self.ball_vy[i] / vy_scale
                obs[o + 4] = t_land[i] / self.T_MAX
                obs[o + 5] = x_land[i] / self.WIDTH
                obs[o + 6] = (x_land[i] - paddle_center) / self.WIDTH
        else:
            for i in range(self.num_balls):
                o = 2 + 4 * i
                obs[o + 0] = self.ball_x[i] / self.WIDTH
                obs[o + 1] = self.ball_y[i] / self.HEIGHT
                obs[o + 2] = self.ball_vx[i] / vx_scale
                obs[o + 3] = self.ball_vy[i] / vy_scale

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "score": round(self._score, 3),
            "hits": self.hits,
            "misses": self.misses,
            "avg_hit": round(self._score / self.hits, 3) if self.hits else 0.0,
        }

    # ================= 观测说明（给专家策略用） =================
    def obs_spec(self) -> dict:
        """声明观测分量的物理尺度，让纯 obs 专家能还原真实物理量。

        观测布局（rich_obs=True，每球 7 维，前面还有 2 维挡板信息）：

            obs[0]        挡板左上角 x      / (WIDTH - PADDLE_W)
            obs[1]        挡板速度          / paddle_speed
            每个球 i 从 2 + 7*i 开始：
                +0  球心 x        * WIDTH
                +1  球心 y        * HEIGHT
                +2  水平速度      * vx_scale
                +3  下落速度      * vy_scale
                +4  落地剩余步数  * T_MAX
                +5  落点 x        * WIDTH
                +6  (落点x - 挡板中心x) / WIDTH

        专家需要的是「乘回去」的系数，所以这里给的全部是 `obs * scale = 物理量`。
        """
        if not self.rich_obs:
            raise NotImplementedError(
                "catcher 在 rich_obs=False 时观测里不含落点信息（t_land / x_land），"
                "纯 obs 的专家策略无法据此刻画落点。\n"
                "请用 rich_obs=True 训练/演示，或为专家另写一版基于物理外推的实现。"
            )

        vx_scale = max(abs(self.BALL_VX_RANGE[0]), abs(self.BALL_VX_RANGE[1]))
        vy_scale = max(abs(self.BALL_VY_RANGE[0]), abs(self.BALL_VY_RANGE[1]))

        return {
            "kind": "catcher_rich",
            "n_slots": self.num_balls,
            "per_ball": 7,
            "obs_dim": 2 + 7 * self.num_balls,
            # 挡板
            "paddle_x_scale": float(self.WIDTH - self.PADDLE_W),
            "paddle_speed_scale": float(self.paddle_speed),
            # 球
            "x_scale": float(self.WIDTH),
            "y_scale": float(self.HEIGHT),
            "vx_scale": float(vx_scale),
            "vy_scale": float(vy_scale),
            "t_scale": float(self.T_MAX),
            "land_x_scale": float(self.WIDTH),
            "land_dx_scale": float(self.WIDTH),
            # 用于可达性判定与瞄准的几何量
            "paddle_w": float(self.PADDLE_W),
            "ball_r": float(self.BALL_R),
            "paddle_y": float(self.PADDLE_Y),
            "width": float(self.WIDTH),
            "t_max": float(self.T_MAX),
        }

    # ================= 得分 =================
    @property
    def score(self) -> float:
        return self._score

    @property
    def score_text(self) -> str:
        return f"Score: {self._score:.1f}  Hits: {self.hits}"

    # ================= 人类操控 =================
    @property
    def action_hint(self) -> str:
        return "← / A 左移      → / D 右移      不按 = 不动"

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
        self._init_pygame()
        canvas = self._draw_canvas()

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame_display_flip()
            self.clock.tick(self.render_fps)
            return None

        if hasattr(__import__("pygame").image, "tobytes"):
            raw = __import__("pygame").image.tobytes(canvas, "RGB")
        else:
            raw = __import__("pygame").image.tostring(canvas, "RGB")
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.HEIGHT, self.WIDTH, 3
        ).copy()

    # ================= 显示初始化 =================
    @property
    def display_caption(self) -> str:
        """窗口标题（基类 init_display 会用它）。"""
        return f"Catcher - {self.num_balls} balls / speed {self.paddle_speed:g}"

    def _init_font(self):
        """字体由游戏自己创建（各游戏字号不同）。"""
        import pygame
        self.font = pygame.font.Font(None, 28)

    def _init_pygame(self):
        """兼容旧调用点：直接委托给基类的 init_display()。"""
        self.init_display()

    def _draw_canvas(self):
        import pygame
        canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        canvas.fill((20, 24, 40))

        for x in range(0, self.WIDTH + 1, 40):
            for y in range(0, self.HEIGHT + 1, 40):
                pygame.draw.circle(canvas, (34, 40, 62), (x, y), 1)

        # 挡板：红(边缘,低分) -> 绿(中心,高分)
        px = int(self.paddle_x)
        for i in range(self.PADDLE_W):
            t = 1.0 - abs((i + 0.5) / self.PADDLE_W * 2.0 - 1.0)
            color = (int(230 + (90 - 230) * t),
                     int(80 + (230 - 80) * t),
                     int(80 + (120 - 80) * t))
            pygame.draw.line(canvas, color, (px + i, self.PADDLE_Y),
                             (px + i, self.PADDLE_Y + self.PADDLE_H - 1))
        pygame.draw.rect(canvas, (240, 250, 255),
                         pygame.Rect(px, self.PADDLE_Y, self.PADDLE_W, self.PADDLE_H),
                         width=2, border_radius=4)

        # 小球
        for i in range(self.num_balls):
            cx, cy = int(self.ball_x[i]), int(self.ball_y[i])
            pygame.draw.circle(canvas, (255, 165, 40), (cx, cy), self.BALL_R)
            pygame.draw.circle(canvas, (255, 245, 210), (cx - 3, cy - 3), 3)

        # HUD
        line1 = self.font.render(
            f"Score: {self._score:6.1f}   Hits: {self.hits}", True, (255, 255, 255))
        canvas.blit(line1, (12, 10))
        miss_color = (255, 110, 110) if self.misses else (150, 220, 150)
        line2 = self.font.render(
            f"Miss: {self.misses}/{self.max_misses}   Balls: {self.num_balls}",
            True, miss_color)
        canvas.blit(line2, (12, 40))
        return canvas

    def close(self):
        import pygame
        if self._pygame_ready:
            pygame.display.quit()
            pygame.quit()
            self._pygame_ready = False
        self.window = None
        self.clock = None
        self.closed = True


def pygame_display_flip():
    import pygame
    pygame.display.flip()


if __name__ == "__main__":
    run_human(CatcherGame(render_mode="human"))