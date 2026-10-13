"""
tests/test_human_play.py
========================
**人类游玩路径的回归测试。**

背景：曾经出现过这个崩溃
------------------------
    Traceback (most recent call last):
      File "human_play.py", line 65, in <module>
        run_human(game)
      File "envs/base.py", line 172, in run_human
        for event in pygame.event.get():
    pygame.error: video system not initialized

根因：pygame 的显示/事件系统原本是**第一次 render() 时才初始化**的
（游戏里的 `_init_pygame()`），而 `run_human` 的循环体第一件事就是
`pygame.event.get()` —— 事件系统还没起来，于是第一帧就崩。

修法：把初始化**显式化**成 `BaseGame.init_display()`，
由 `run_human` 在进循环前主动调用。这样「谁要交互，谁先初始化」，
不再依赖"恰好在 render 里被初始化"这种隐式时序。

本测试用 dummy 视频驱动跑真实的 run_human 循环，并通过投递
QUIT 事件让它自动退出，从而在无显示环境下也能验证这条路径。

运行：
    python tests/test_human_play.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")


# ----------------------------------------------------------------------
def run_snippet(code: str, timeout: int = 60) -> tuple[int, str, str]:
    """在干净子进程里跑一段代码（强制 dummy 视频驱动）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"      # 无显示环境也能跑
    env["SDL_AUDIODRIVER"] = "dummy"

    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ======================================================================
def test_run_human_does_not_crash(c: _Case) -> None:
    """核心回归：run_human 能进入事件循环（不再 video system not initialized）。"""
    for game in ("catcher", "dodger"):
        code = f"""
        import time, threading
        import pygame
        from games import make_game, run_human

        g = make_game("{game}", render_mode="human")

        # 2 秒后投递 QUIT，模拟用户关窗口
        def quit_later():
            time.sleep(2)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        threading.Thread(target=quit_later, daemon=True).start()

        run_human(g)
        print("RUN_HUMAN_OK")
        """
        rc, out, err = run_snippet(code)
        c.check(f"{game}: run_human 正常运行", rc == 0 and "RUN_HUMAN_OK" in out,
                (err.strip().splitlines() or ["?"])[-1])
        c.check(f"{game}: 没有 video system 初始化错误",
                "video system not initialized" not in err,
                err.strip().splitlines()[-1] if err.strip() else "")


def test_init_display_exists_and_idempotent(c: _Case) -> None:
    """init_display 必须存在、幂等，且能建好窗口。"""
    code = """
    import pygame
    from games import make_game

    g = make_game("catcher", render_mode="human")

    assert hasattr(g, "init_display"), "BaseGame 缺少 init_display"

    # 幂等：重复调用不应报错
    g.init_display()
    g.init_display()
    g.init_display()

    assert pygame.display.get_init(), "init_display 之后 display 仍未初始化"
    assert g.window is not None, "human 模式下应创建窗口"
    assert g._pygame_ready, "_pygame_ready 应为 True"

    # 之后 render 应正常
    g.reset()
    g.render()
    g.close()
    print("INIT_DISPLAY_OK")
    """
    rc, out, err = run_snippet(code)
    c.check("init_display 存在、幂等、可用",
            rc == 0 and "INIT_DISPLAY_OK" in out,
            (err.strip().splitlines() or ["?"])[-1])


def test_human_play_script_runs(c: _Case) -> None:
    """直接跑 human_play.py 入口（用 --list-games 避免进入互动循环）。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"

    proc = subprocess.run(
        [sys.executable, str(ROOT / "human_play.py"), "--list-games"],
        capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=60,
    )
    c.check("python human_play.py --list-games 正常",
            proc.returncode == 0 and "catcher" in proc.stdout,
            (proc.stderr.strip().splitlines() or ["?"])[-1])

    # 确认这条路径确实不需要 torch
    c.check("human_play --list-games 不需要 torch",
            "torch" not in proc.stderr.lower())


def test_rgb_array_still_works(c: _Case) -> None:
    """rgb_array 模式仍靠首次 render 自动初始化（不能被本次修改破坏）。"""
    code = """
    from envs import make_env
    env = make_env("catcher", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (600, 400, 3), frame.shape
    assert frame.dtype.name == "uint8", frame.dtype
    env.close()
    print("RGB_OK")
    """
    rc, out, err = run_snippet(code)
    c.check("rgb_array 首帧渲染正常（自动初始化仍在）",
            rc == 0 and "RGB_OK" in out,
            (err.strip().splitlines() or ["?"])[-1])


def test_display_caption_used(c: _Case) -> None:
    """窗口标题应来自游戏的 display_caption（各游戏不同）。"""
    code = """
    from games import make_game
    g1 = make_game("catcher", render_mode="human")
    g2 = make_game("dodger", render_mode="human")
    assert g1.display_caption != g2.display_caption, "两个游戏标题不该相同"
    assert "Catcher" in g1.display_caption
    assert "Dodger" in g2.display_caption
    print("CAPTION_OK")
    """
    rc, out, err = run_snippet(code)
    c.check("各游戏有独立的窗口标题",
            rc == 0 and "CAPTION_OK" in out,
            (err.strip().splitlines() or ["?"])[-1])


def test_run_human_calls_init_display(c: _Case) -> None:
    """防回归：run_human 必须在事件循环**之前**调用 init_display。"""
    import ast

    src = (ROOT / "envs" / "base.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_human":
            func = node
            break

    c.check("envs/base.py 里有 run_human", func is not None)
    if func is None:
        return

    calls = [n.lineno for n in ast.walk(func)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "init_display"]

    c.check("run_human 里调用了 init_display", bool(calls))

    # 事件循环必须晚于 init_display
    event_gets = [n.lineno for n in ast.walk(func)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "get"
                  and isinstance(n.func.value, ast.Attribute)
                  and n.func.value.attr == "event"]

    if calls and event_gets:
        c.check("init_display 在 pygame.event.get() 之前被调用",
                min(calls) < min(event_gets),
                f"init_display@{min(calls)} vs event.get@{min(event_gets)}")


# ======================================================================
def main() -> int:
    c = _Case()
    print("[1/6] run_human 不再崩溃（两个游戏）...")
    test_run_human_does_not_crash(c)
    print("[2/6] init_display 存在且幂等 ...")
    test_init_display_exists_and_idempotent(c)
    print("[3/6] human_play.py 入口可运行 ...")
    test_human_play_script_runs(c)
    print("[4/6] rgb_array 模式未被破坏 ...")
    test_rgb_array_still_works(c)
    print("[5/6] 窗口标题 ...")
    test_display_caption_used(c)
    print("[6/6] 调用顺序断言（防回归）...")
    test_run_human_calls_init_display(c)

    print(f"\n通过 {c.passed} 项", end="")
    if c.failed:
        print(f"，失败 {len(c.failed)} 项：")
        for f in c.failed:
            print("  ✗", f)
        return 1
    print("，全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
