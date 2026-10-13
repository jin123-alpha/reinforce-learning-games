"""
tests/test_compile.py
=====================
**编译与导入测试**：所有 .py 文件语法正确、所有模块可独立导入。

这是最基础的一层保护，但它能抓到重构中最常见的低级错误：
    * 移动文件后忘了改 import
    * 循环导入
    * 语法/缩进错误
    * 模块级代码引用了不存在的名字

「可独立导入」这一条尤其重要：重构把文件搬了位置，
如果只在 `import games` 这种「从上层导入」的路径下测，
很容易漏掉「直接 import 某个子模块」时才会暴露的问题。

运行：
    python tests/test_compile.py
"""
from __future__ import annotations

import ast
import os
import py_compile
import subprocess
import sys
import tempfile
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
def all_python_files() -> list[Path]:
    """项目里所有 .py 文件（跳过缓存和临时目录）。"""
    out = []
    for p in ROOT.rglob("*.py"):
        parts = set(p.parts)
        if "__pycache__" in parts or ".git" in parts:
            continue
        out.append(p)
    return sorted(out)


# ======================================================================
def test_syntax(c: _Case) -> None:
    """每个文件都能通过 py_compile（语法 + 编码）。"""
    files = all_python_files()
    c.check("找到了足够多的源文件", len(files) >= 20, f"只找到 {len(files)} 个")

    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            try:
                py_compile.compile(str(f), cfile=str(Path(tmp) / "x.pyc"),
                                   doraise=True)
            except py_compile.PyCompileError as exc:
                bad.append(f"{f.relative_to(ROOT)}: {exc.msg.strip()}")

    c.check("所有文件语法正确", not bad, "；".join(bad[:5]))


def test_ast_parseable(c: _Case) -> None:
    """用 ast 再确认一遍（py_compile 偶尔会缓存）。"""
    bad = []
    for f in all_python_files():
        try:
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as exc:
            bad.append(f"{f.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
        except UnicodeDecodeError:
            bad.append(f"{f.relative_to(ROOT)}: 不是有效的 UTF-8")
    c.check("所有文件可被 ast 解析", not bad, "；".join(bad[:5]))


def test_each_module_imports(c: _Case) -> None:
    """每个模块单独在干净子进程里导入。"""
    modules = [
        "common.paths", "common.cli", "common.runner",
        "envs", "envs.base", "envs.game_env", "envs.normalize",
        "games", "games.catcher", "games.dodger", "games.snake",
        "experts", "experts.base", "experts.catcher",
        "experts.dodger", "experts.snake",
        "compare_child", "train", "play", "human_play", "compare",
    ]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"

    bad = []
    for mod in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()
            bad.append(f"{mod}: {tail[-1] if tail else '?'}")

    c.check("所有模块可独立导入", not bad, "；".join(bad[:4]))


def test_no_circular_import(c: _Case) -> None:
    """分层不允许循环：envs 不该 import games 的顶层。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"

    # envs.game_env 里的 games import 是**函数内**延迟导入，
    # 所以单独 import envs 不会拉起 games —— 这正是我们想要的解耦。
    code = (
        "import sys; import envs; "
        "print('games' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=str(ROOT), env=env)
    c.check("import envs 不会拉起 games（延迟导入生效）",
            proc.stdout.strip() == "False",
            f"输出 {proc.stdout.strip()!r} / {proc.stderr.strip()[-120:]}")


def test_games_dir_importable_standalone(c: _Case) -> None:
    """游戏文件支持 `python -m games.<game>.<game>` 与直接运行。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"

    # 直接运行游戏文件（不进入互动循环）：靠 __main__ 守卫，
    # 我们用 -c 方式 import 它的模块来间接验证 sys.path 垫片没问题
    for game in ("catcher", "dodger", "snake"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import games.{game}.{game}"],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
        )
        c.check(f"games.{game}.{game} 可导入", proc.returncode == 0,
                proc.stderr.strip().splitlines()[-1] if proc.stderr else "")


# ======================================================================
def main() -> int:
    c = _Case()
    print("[1/5] 语法编译 ...")
    test_syntax(c)
    print("[2/5] AST 解析 ...")
    test_ast_parseable(c)
    print("[3/5] 逐模块独立导入 ...")
    test_each_module_imports(c)
    print("[4/5] 无循环导入 ...")
    test_no_circular_import(c)
    print("[5/5] 游戏文件可独立导入 ...")
    test_games_dir_importable_standalone(c)

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
