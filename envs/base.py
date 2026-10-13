"""
envs/base.py
============
**游戏必须实现的统一接口**，以及人类游玩循环。

为什么这个文件在 envs/ 而不是 games/
------------------------------------
它是一份**契约**：约定「什么算一个兼容的游戏」。

    games/  ->  实现契约（只关心玩法、画面、得分）
    envs/   ->  拥有契约（定义接口、翻译成 Gymnasium、统一 reward/score）

所以依赖方向是单向的：

    games/catcher  ──import──>  envs.base
    envs.game_env  ──import──>  envs.base          （不认识任何具体游戏）

`games/__init__.py` 会把 BaseGame / run_human 重新导出一次，
这样老的 `from games import BaseGame` 依然能用，不破坏已有写法。

游戏只需要实现 4 个方法 + 声明 2 个空间
---------------------------------------
    reset(seed)          重置，返回 (obs, info)
    step(action)         推进一帧，返回 (obs, reward, terminated, truncated, info)
    render()             human 画窗口 / rgb_array 返回 (H,W,3) 数组
    close()              释放 pygame 资源
    action_space         用 gymnasium.spaces 声明
    observation_space    用 gymnasium.spaces 声明

另外两个**可选**成员给人类游玩用：
    keys_to_action(keys) 把按键翻译成动作（没按键要返回「不动」）
    action_hint          按键提示文字
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseGame(ABC):
    """所有游戏必须实现的接口。"""

    # ------------------------------------------------------------------
    # 元信息（子类覆盖）
    # ------------------------------------------------------------------
    name: str = "base"
    render_modes: tuple[str, ...] = ("human", "rgb_array")
    render_fps: int = 60

    # 子类要在 __init__ 里设置这两个（用 gymnasium.spaces 定义）
    action_space = None          # 例: spaces.Discrete(3)
    observation_space = None     # 例: spaces.Box(-1, 1, (23,), np.float32)

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """重置游戏，返回 (观测, info)。"""

    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """推进一帧，返回 (obs, reward, terminated, truncated, info)。

        terminated: 游戏结束（输了/赢了）
        truncated : 被外部截断（比如达到步数上限）
        """

    @abstractmethod
    def render(self):
        """human 模式：画到窗口。rgb_array 模式：返回 (H, W, 3) uint8 数组。"""

    @abstractmethod
    def close(self) -> None:
        """释放 pygame 资源。"""

    # ------------------------------------------------------------------
    # 人类操控（子类按需覆盖）
    # ------------------------------------------------------------------
    def keys_to_action(self, keys):
        """把 pygame 的按键状态翻译成游戏动作。

        `keys` 是 pygame.key.get_pressed() 的返回值。
        没有按键时必须返回一个合法的「什么都不做」动作，否则游戏会卡住。
        """
        raise NotImplementedError

    @property
    def action_hint(self) -> str:
        """给人类玩家看的按键提示。"""
        return ""

    # ------------------------------------------------------------------
    # 得分 —— 与环境层的统一口径
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def score(self) -> float:
        """游戏自己的得分。

        **这是游戏与环境层之间唯一的「成绩」接口。**
        环境层保证在 info 里一定能看到它（连同 episode_return / steps），
        所以任何游戏都不能省略这个属性 —— 故意的，见下。

        注意：score 与 RL 的 reward 不一定是同一个东西。
        catcher 的 score 是「接球累计得分」（0 ~ 30 量级），
        dodger 的 score 是「存活步数」（0 ~ 1000 量级）。
        envs 只统一**访问方式**，不强行统一**数值尺度**；
        需要跨游戏比大小的时候，用 envs/normalize.py 的 Wrapper。
        """

    @property
    def score_text(self) -> str:
        """画在画面左上角的得分文字。"""
        return f"Score: {self.score:.1f}"

    # ------------------------------------------------------------------
    # 观测说明（可选，给纯 obs 专家策略用）
    # ------------------------------------------------------------------
    def obs_spec(self) -> dict:
        """声明观测向量各分量的**物理尺度**，供专家策略还原真实物理量。

        为什么需要这个
        --------------
        观测里的数字都是归一化过的（除以宽度、除以最大速度……），
        纯 obs 的专家策略要把它们**还原**成像素/速度才能做落点计算。
        如果专家自己去硬编码 WIDTH=400、PADDLE_W=90，
        那么游戏改一个常量、专家就静默失效 —— 这就是重构前
        `ExpertCatcher` 直接读 `game.WIDTH` 的隐患。

        改成游戏自己**声明**之后：
          * 专家仍然只拿 obs（不接触 game 实例），保持纯函数性质；
          * 尺度由游戏提供，改常量不会让专家算错；
          * 没实现这个方法的游戏，就用不了需要物理量的专家（明确报错）。

        返回的 dict 至少要能表达「怎么从 obs 还原物理量」。
        约定用 `_scale` 后缀表示「obs * scale = 物理量」。
        没有专家策略的游戏可以完全不实现这个方法。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 没有声明观测尺度（obs_spec），"
            f"因此无法使用需要还原物理量的专家策略。"
        )

    # ------------------------------------------------------------------
    # 显示初始化（人类游玩 / 交互式运行前必须调用）
    # ------------------------------------------------------------------
    def init_display(self) -> None:
        """初始化 pygame 并（human 模式下）创建窗口。

        **为什么需要这个方法**：pygame 的事件系统（`pygame.event.get()`）
        必须在使用前完成初始化，而窗口尺寸只有游戏自己知道。
        重构前这一步隐式地发生在第一次 `render()` 里，于是
        `run_human` 的事件循环在第一帧就会崩：

            pygame.error: video system not initialized

        所以把它显式化：谁要交互，谁先调 `init_display()`。
        默认实现已足够覆盖「pygame.init + set_mode」这一常见情况；
        游戏有额外资源（字体、音效）就覆盖它。
        """
        import pygame

        if getattr(self, "_pygame_ready", False):
            return

        pygame.init()
        pygame.font.init()

        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption(
                getattr(self, "display_caption", None) or self.name
            )
            self.clock = pygame.time.Clock()

        # 字体交给游戏自己按需创建（各游戏字号不同）
        if getattr(self, "font", None) is None and hasattr(self, "_init_font"):
            self._init_font()

        self._pygame_ready = True

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ======================================================================
# 人类游玩循环（只依赖 pygame，不依赖任何 RL 库）
# ======================================================================
def run_human(game: BaseGame, episodes: int = 0) -> None:
    """用键盘玩这个游戏。episodes=0 表示一直玩到关窗口。

    按键:  方向键 / A D 移动      R 重开      P 暂停      Esc 退出

    注意这里必须**先**调 init_display() 建好窗口，再进事件循环：
    游戏的显示初始化原本是第一次 render() 时才发生的，而循环体
    第一件事就是 pygame.event.get() —— 等 render() 来初始化的话，
    事件系统在第一次取事件时还没起来，会直接抛
    `pygame.error: video system not initialized`。
    """
    import pygame

    game.init_display()

    game.reset()
    ep = 0
    running = True
    paused = False

    print(f"=== {game.name} 人类游玩模式 ===")
    if game.action_hint:
        print(game.action_hint)
    print("R 重开 / P 暂停 / 关闭窗口退出")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_p:
                    paused = not paused

        if not paused:
            keys = pygame.key.get_pressed()
            action = game.keys_to_action(keys)
            obs, reward, terminated, truncated, info = game.step(action)
            if terminated or truncated:
                ep += 1
                print(f"[回合 {ep}] {game.score_text}")
                game.reset()

        game.render()
        if getattr(game, "closed", False):
            running = False

    game.close()
