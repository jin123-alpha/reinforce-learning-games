"""
games/snake/snake.py
====================
贪吃蛇（Snake）—— 第三个内置游戏，用来继续检验框架的通用性。

**它不知道强化学习的存在**，只实现 `envs/base.py` 的 `BaseGame` 契约：

    python -m games.snake.snake          # 直接人类游玩（方向键 / WASD）

它与前两个游戏**结构上又不一样**
------------------------------
| | catcher | dodger | snake |
| --- | --- | --- | --- |
| 状态空间 | 连续（像素/速度）| 连续 | **离散网格** |
| 动作 | Discrete(3) 左右 | Discrete(3) 左右 | **Discrete(4) 四方向**|
| 回合结束 | 失误次数 | 被撞 | 撞墙/撞自己/吃满 |
| 状态是否增长 | 球数固定 | 障碍物动态但无记忆 | **蛇身随吃食物变长** |

第三点最关键：蛇身是**玩家自己行为的累积结果**，所以
「当前状态」里必须编码整条蛇（或者至少是它的可碰撞轮廓）。
这是前两个游戏都没有的性质 —— catcher 的球永远只有 num_balls 颗，
dodger 的障碍物也彼此独立、没有记忆。

观测设计（为什么是 11 维）
--------------------------
动作是「绝对方向」还是「相对转向」？两者都合理，这里选**相对转向**
（0=直行 1=右转 2=左转 3=掉头），因为：

    - 绝对方向下，模型必须额外学会「不要反向」（那会立刻自杀），
      等于把一条硬约束塞进离散动作里，浪费样本；
    - 相对转向下，动作的**语义与朝向无关**（"右转"永远是"朝当前方向的右边转"），
      策略可以在不同朝向上复用同一套判断，样本效率高得多；
    - 人类游玩也更顺手（见 keys_to_action）。

观测向量（全部归一化到 [-1, 1] 或 [0, 1]）：

    obs[0]  朝向的 sin          （用 sin/cos 而不是索引，避免 0/3 的假跳变）
    obs[1]  朝向的 cos
    从 obs[2] 开始，每个方向 2 维，顺序是 [直行, 右, 左]（**不含掉头**）：
        +0  该方向是否立刻有危险       0/1（撞墙 / 撞身体 / 追尾）
        +1  该方向到食物的"接近程度"   -1（远离）~ +1（最近）

    即 2 + 3*2 = 8 维，再补 3 维全局量：
    obs[8]  蛇长 / 网格格子总数         （进度感：吃得越多越危险）
    obs[9]  (食物x - 蛇头x) / 网格宽    （不带朝向的绝对方向，兜底用）
    obs[10] (食物y - 蛇头y) / 网格高

    合计 11 维。

    为什么既给「相对方向的危险/食物信号」又给「绝对坐标差」：
    前者是**策略友好的归纳偏置**（专家直接读它就能做出正确决策，
    而且换个朝向不用重新学），后者保证信息完备（模型总能自己算出
    更复杂的路径规划）。只给前者会让状态变成部分可观测，
    只给后者则样本效率极低 —— 两个都给是刻意的折中。

得分
----
    score = 吃到的食物个数（每颗 +1）
    reward = 吃到食物 +1，其余每步给一个极小的存活惩罚
             （shaping 见 step() 的说明，默认关闭以保持口径干净）

为什么终止条件里有「吃满」
--------------------------
蛇最长就是所有格子（不含蛇头）。一旦吃满，格子用尽不可能再生成食物，
所以那是**真正的胜利**，必须 terminated 而不是 truncated ——
否则 SB3 的 bootstrap 会把它当成「只是被截断」，学出一个
「故意不吃满」的怪异策略。这是一个容易写错、且错了很难看出来的地方。
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
from gymnasium import spaces

# ---- 兼容「作为包导入」与「直接运行本文件」两种情况 ----
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from envs.base import BaseGame, run_human


# 方向常量：0=上 1=右 2=下 3=左（顺时针，便于用 (d+1)%4 表示右转）
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DELTA = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}

# 相对动作（专家的动作语义）
STRAIGHT, TURN_RIGHT, TURN_LEFT, TURN_BACK = 0, 1, 2, 3


class SnakeGame(BaseGame):
    name = "snake"

    # ---------------- 网格常量 ----------------
    GRID_W = 16
    GRID_H = 16
    CELL = 24                       # 每格多少像素 -> 画面 384x384
    HUD_H = 56                      # 顶部留给文字的空白

    WIDTH = GRID_W * CELL
    HEIGHT = GRID_H * CELL + HUD_H

    MAX_STEPS_WITHOUT_FOOD = 120    # 多久没吃到东西就判负（防止绕圈刷步数）

    def __init__(
        self,
        render_mode: str | None = None,
        grid_w: int = GRID_W,
        grid_h: int = GRID_H,
        max_steps: int = 1000,
        initial_len: int = 3,
        step_penalty: float = 0.0,
        food_reward: float = 1.0,
        death_penalty: float = 0.0,
        max_steps_without_food: int = MAX_STEPS_WITHOUT_FOOD,
        relative_actions: bool = True,
    ) -> None:
        assert render_mode in self.render_modes or render_mode is None
        assert grid_w >= 3 and grid_h >= 3, "网格太小，放不下初始蛇身"

        self.render_mode = render_mode
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.max_steps = int(max_steps)
        self.initial_len = int(initial_len)
        self.step_penalty = float(step_penalty)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.max_steps_without_food = int(max_steps_without_food)
        self.relative_actions = bool(relative_actions)

        n_cells = self.grid_w * self.grid_h
        assert self.initial_len + 1 <= n_cells

        # WIDTH/HEIGHT 是**实例**属性（网格尺寸可配），不是类常量 ——
        # 基类 init_display() 会读 self.WIDTH 来建窗口。
        self.WIDTH = self.grid_w * self.CELL
        self.HEIGHT = self.grid_h * self.CELL + self.HUD_H

        # 动作：0=直行 1=右转 2=左转 3=掉头（相对转向，见模块 docstring）
        self.action_space = spaces.Discrete(4)
        # 观测：2（朝向）+ 3 方向 × 2（危险/食物接近度）+ 3（全局）
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # ---- 渲染资源（延迟初始化）----
        self.window = None
        self.clock = None
        self.font = None
        self.hint_font = None
        self._pygame_ready = False
        self.closed = False

        # ---- 状态 ----
        self.snake: deque[tuple[int, int]] = deque()   # [0] 是蛇头
        self.direction = RIGHT
        self.food = (0, 0)
        self.steps = 0
        self.eaten = 0
        self.steps_since_food = 0
        self._rng = np.random.default_rng()
        self._food_consumed_cells = 0

    # ================= 观测维度 =================
    @property
    def obs_dim(self) -> int:
        return 2 + 3 * 2 + 3

    # ================= 游戏逻辑 =================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.eaten = 0
        self.steps_since_food = 0
        self.direction = RIGHT

        # 蛇放在中间，朝右；身体向左延伸
        cx = self.grid_w // 2
        cy = self.grid_h // 2
        self.snake = deque()
        for i in range(self.initial_len):
            self.snake.append((cx - i, cy))

        self._place_food()
        return self._get_obs(), self._get_info()

    def _place_food(self) -> None:
        """在空位上随机放一颗食物。

        用「列出所有空格再抽一个」而不是「随机抽到空为止」：
        后者在蛇很长时会退化成近似无限循环，是这类游戏的经典坑。
        """
        occupied = set(self.snake)
        free = [(x, y)
                for y in range(self.grid_h)
                for x in range(self.grid_w)
                if (x, y) not in occupied]
        if not free:
            # 蛇填满了整个网格：没有地方放食物，保持原位。
            # 这种情况下 step() 会立刻判定「胜利」。
            self.food = (-1, -1)
            return
        idx = int(self._rng.integers(len(free)))
        self.food = free[idx]

    # ------------------------------------------------------------------
    def _resolve_direction(self, action: int) -> int:
        """把动作翻译成新的绝对朝向。

        relative_actions=True （默认）：0=直行 1=右转 2=左转 3=掉头
        relative_actions=False        ：0=上 1=右 2=下 3=左（绝对方向）
        """
        if self.relative_actions:
            turn = {STRAIGHT: 0, TURN_RIGHT: 1, TURN_LEFT: -1, TURN_BACK: 2}[action]
            return (self.direction + turn) % 4
        return int(action)

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"非法动作: {action}"

        self.steps += 1
        self.steps_since_food += 1
        reward = self.step_penalty

        new_dir = self._resolve_direction(action)
        self.direction = new_dir

        hx, hy = self.snake[0]
        dx, dy = _DELTA[new_dir]
        nx, ny = hx + dx, hy + dy

        # ---- 1. 撞墙 ----
        if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
            return (self._get_obs(), reward + self.death_penalty,
                    True, False, self._get_info())

        # ---- 2. 撞自己 ----
        # 注意：尾巴当前这一格**这一步会让出来**（除非蛇正在变长），
        # 所以跟进尾巴是合法的。这是贪吃蛇的经典细节 ——
        # 写成「身体全部算碰撞」会让蛇无法贴尾走位，
        # 表现上就是"偶尔莫名其妙撞死"，很难排查。
        body = set(self.snake)
        tail = self.snake[-1]
        chasing_tail = (nx, ny) == tail
        will_grow = (nx, ny) == self.food
        if chasing_tail and not will_grow:
            body.discard(tail)
        if (nx, ny) in body:
            return (self._get_obs(), reward + self.death_penalty,
                    True, False, self._get_info())

        # ---- 3. 前进 ----
        self.snake.appendleft((nx, ny))

        if will_grow:
            self.eaten += 1
            self.steps_since_food = 0
            reward += self.food_reward
            self._place_food()
            # 吃到食物 -> 不弹尾巴（蛇变长）
            won = len(self.snake) >= self.grid_w * self.grid_h
            if won:
                # 吃满整个网格 = 真正的胜利，必须 terminated（见模块 docstring）
                return (self._get_obs(), reward, True, False, self._get_info())
        else:
            self.snake.pop()

        # ---- 4. 太久没吃到东西：判负 ----
        starved = (self.max_steps_without_food > 0
                   and self.steps_since_food >= self.max_steps_without_food)
        if starved:
            return (self._get_obs(), reward, True, False, self._get_info())

        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated_false(), truncated, self._get_info()

    # ================= 观测 =================
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 朝向：sin/cos 编码（避免 3->0 的数值跳变）
        d = self.direction
        obs[0] = float(np.sin(d * np.pi / 2.0))
        obs[1] = float(np.cos(d * np.pi / 2.0))

        hx, hy = self.snake[0]
        # 「下一步会不会变长」决定尾巴算不算障碍（与 step() 的判定保持一致）
        body = set(self.snake)
        tail = self.snake[-1]

        # 相对三方向：直行 / 右转 / 左转（不含掉头）
        for k, turn in enumerate((0, 1, -1)):
            nd = (d + turn) % 4
            dx, dy = _DELTA[nd]
            tx, ty = hx + dx, hy + dy
            o = 2 + 2 * k

            # --- 危险度 ---
            danger = 0.0
            if not (0 <= tx < self.grid_w and 0 <= ty < self.grid_h):
                danger = 1.0                                   # 撞墙
            else:
                blocked = set(body)
                # 只有「不会同时吃到食物」时，尾巴才会让出来
                if (tx, ty) == tail and (tx, ty) != self.food:
                    blocked.discard(tail)
                if (tx, ty) in blocked:
                    danger = 1.0                               # 撞身体 / 追尾
            obs[o + 0] = danger

            # --- 该方向是否让食物更近 / 更远 / 吃到 ---
            # 编码成 3 档而不是连续的"距离变化率"：
            #   +1.0  这一步**直接吃到**食物
            #   +0.5  朝食物靠近（距离减少）
            #   -0.5  远离食物
            #    0.0  距离不变
            # 离散编码让「朝向食物」这个信号与朝向本身解耦，
            # 专家和模型都能直接读懂，不必先去还原曼哈顿距离。
            if (tx, ty) == self.food:
                obs[o + 1] = 1.0
            else:
                cur_d = abs(hx - self.food[0]) + abs(hy - self.food[1])
                nxt_d = abs(tx - self.food[0]) + abs(ty - self.food[1])
                if nxt_d < cur_d:
                    obs[o + 1] = 0.5
                elif nxt_d > cur_d:
                    obs[o + 1] = -0.5
                else:
                    obs[o + 1] = 0.0

        # --- 全局量 ---
        n_cells = self.grid_w * self.grid_h
        obs[8] = len(self.snake) / n_cells
        obs[9] = (self.food[0] - hx) / self.grid_w
        obs[10] = (self.food[1] - hy) / self.grid_h

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        return {
            "score": float(self.eaten),
            "steps": self.steps,
            "eaten": self.eaten,
            "length": len(self.snake),
            "steps_since_food": self.steps_since_food,
        }

    # ================= 观测说明（给专家策略用） =================
    def obs_spec(self) -> dict:
        """声明观测各分量的物理尺度，供纯 obs 的专家还原真实量。

        与 catcher / dodger 的 obs_spec 同一约定：给的是
        「obs * scale = 物理量」的系数，专家不硬编码任何常量。

        观测布局（11 维）：

            obs[0]   朝向 sin          （已经是物理量，scale = 1）
            obs[1]   朝向 cos
            obs[2+2k]   第 k 个方向（直行/右/左）的危险度 0/1
            obs[3+2k]   第 k 个方向相对食物的取值：
                          +1.0 = 这一步直接吃到
                          +0.5 = 靠近食物
                          -0.5 = 远离食物
                           0.0 = 距离不变
            obs[8]   蛇长 / 格子总数
            obs[9]   (食物x - 蛇头x) / 网格宽   -> * grid_w 还原格数
            obs[10]  (食物y - 蛇头y) / 网格高   -> * grid_h 还原格数

        朝向用 sin/cos 而不是 0..3 的索引，是为了不让「上」和「左」
        在数值上显得最远 —— 对网络来说 0/3 的索引会引入一个假的距离。
        """
        return {
            "kind": "snake",
            "obs_dim": self.obs_dim,
            "n_relative_dirs": 3,
            "rel_dirs": (0, 1, -1),        # 直行 / 右转 / 左转（与 _get_obs 一致）
            "turn_back": 2,                # 掉头对应的转向量（专家通常不用）
            "relative_actions": self.relative_actions,
            "grid_w": int(self.grid_w),
            "grid_h": int(self.grid_h),
            "n_cells": int(self.grid_w * self.grid_h),
            "initial_len": int(self.initial_len),
            "max_steps_without_food": int(self.max_steps_without_food),
            # 还原系数（obs * scale = 物理量）
            "dir_sin_scale": 1.0,
            "dir_cos_scale": 1.0,
            "length_scale": float(self.grid_w * self.grid_h),
            "food_dx_scale": float(self.grid_w),
            "food_dy_scale": float(self.grid_h),
            # 食物接近度的编码档位（专家据此判定"这一步能不能吃到"）
            "eat_value": 1.0,
            "closer_value": 0.5,
        }

    # ================= 得分 =================
    @property
    def score(self) -> float:
        return float(self.eaten)

    @property
    def score_text(self) -> str:
        return f"Eaten: {self.eaten}   Length: {len(self.snake)}"

    # ================= 人类操控 =================
    @property
    def action_hint(self) -> str:
        return "↑ ↓ ← → 或 W A S D 转向（相对当前朝向）"

    def keys_to_action(self, keys):
        """把按键翻译成**相对转向**动作。

        因为我们把动作定义成了相对转向，这里需要把「按了绝对方向键」
        换算成「相对于当前朝向要转多少」。这正是相对动作的人类友好之处：
        玩家按「←」时想要的是「往我左边那个屏幕方向走」，
        至于那对应转多少度，游戏自己算就好。
        """
        import pygame

        want = None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            want = UP
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            want = RIGHT
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            want = DOWN
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            want = LEFT

        if want is None:
            return STRAIGHT          # 没按键 = 继续直行（不能返回"不动"：
                                     # 贪吃蛇没有"停"这个动作，停下来就卡住了）

        if not self.relative_actions:
            return want

        turn = (want - self.direction) % 4
        # turn: 0=直行 1=右转 2=掉头 3=左转  ->  映射到动作编号
        return {0: STRAIGHT, 1: TURN_RIGHT, 2: TURN_BACK, 3: TURN_LEFT}[turn]

    # ================= 渲染 =================
    def render(self):
        if self.closed:
            return None
        import pygame
        self._init_pygame()
        canvas = self._draw_canvas()

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            return None

        raw = pygame.image.tobytes(canvas, "RGB") if hasattr(pygame.image, "tobytes") \
            else pygame.image.tostring(canvas, "RGB")
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.HEIGHT, self.WIDTH, 3).copy()

    def _draw_canvas(self):
        import pygame
        canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        canvas.fill((14, 20, 24))

        top = self.HUD_H
        cell = self.CELL

        # 棋盘格
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if (gx + gy) % 2 == 0:
                    pygame.draw.rect(
                        canvas, (20, 30, 34),
                        pygame.Rect(gx * cell, top + gy * cell, cell, cell))

        # 食物
        if self.food[0] >= 0:
            fx, fy = self.food
            r = pygame.Rect(fx * cell + 3, top + fy * cell + 3,
                            cell - 6, cell - 6)
            pygame.draw.rect(canvas, (240, 90, 90), r, border_radius=5)
            pygame.draw.rect(canvas, (255, 210, 200), r, width=2, border_radius=5)

        # 蛇身：头亮尾暗，方便一眼看出朝向
        n = max(len(self.snake) - 1, 1)
        for i, (sx, sy) in enumerate(self.snake):
            t = i / n
            color = (int(70 + 90 * (1 - t)), int(200 - 60 * t), int(110 - 40 * t))
            r = pygame.Rect(sx * cell + 1, top + sy * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(canvas, color, r, border_radius=4)
            if i == 0:
                pygame.draw.rect(canvas, (235, 255, 255), r, width=2,
                                 border_radius=4)

        # HUD
        line1 = self.font.render(
            f"Eaten: {self.eaten}   Length: {len(self.snake)}", True,
            (255, 255, 255))
        canvas.blit(line1, (10, 8))

        patience = (self.max_steps_without_food - self.steps_since_food
                    if self.max_steps_without_food > 0 else 0)
        color = (255, 170, 90) if patience < 30 else (150, 210, 160)
        line2 = self.hint_font.render(
            f"Grid {self.grid_w}x{self.grid_h}   "
            f"No-food in: {max(patience, 0)}", True, color)
        canvas.blit(line2, (10, 32))
        return canvas

    # ================= 显示初始化 =================
    @property
    def display_caption(self) -> str:
        return f"Snake - {self.grid_w}x{self.grid_h}"

    def _init_font(self):
        import pygame
        self.font = pygame.font.Font(None, 28)
        self.hint_font = pygame.font.Font(None, 22)

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


def terminated_false() -> bool:
    """显式写出来，避免在 return 里堆一串裸 False 让人看错位置。"""
    return False


if __name__ == "__main__":
    run_human(SnakeGame(render_mode="human"))
