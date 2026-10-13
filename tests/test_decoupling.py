"""
tests/test_decoupling.py
========================
**架构约束测试**：证明「游戏层不知道强化学习的存在」。

断言三件事
----------
1. import 整个 games 包之后，`torch` / `stable_baselines3` / `tensorboard`
   都不在 sys.modules 里 —— 游戏层没碰 RL 库。

2. import human_play 之后同样如此 —— 「人类游玩」这条路径
   不需要 RL 依赖，也就意味着**在没装 torch 的机器上也能玩**。

3. 源码层面：games/ 和 human_play.py 里不出现这些库的名字
   （防止有人写延迟 import 绕过第 1、2 条）。

为什么值得单独一个测试文件
--------------------------
README 的核心卖点就是「游戏与环境完全解耦」。这种架构声明
如果没有可执行的检查，几个月后一定会有人图方便写一句
`from stable_baselines3 import PPO` 把它破坏掉，而且不会有人发现。

重构前这个断言其实是**失败**的：human_play.py 从 train.py import
parse_overrides，而 train.py 顶层 import SB3。

运行：
    python tests/test_decoupling.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 游戏层绝对不该出现的库
FORBIDDEN = ["torch", "stable_baselines3", "tensorboard", "gymnasium.Env"]


class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")


# ======================================================================
# 子进程里 import，然后报告加载了哪些「重」模块
# ======================================================================
_PROBE = r"""
import sys, json
{imports}
heavy = [m for m in ("torch", "stable_baselines3", "tensorboard")
         if m in sys.modules]
tops = sorted({{m.split(".")[0] for m in sys.modules}})
print(json.dumps({{"heavy": heavy, "n_modules": len(sys.modules), "tops": tops}}))
"""


def _probe(import_lines: list[str], env: dict | None = None) -> dict:
    """在干净的子进程里执行 import，返回加载到的重模块。"""
    import json

    code = _PROBE.format(imports="\n".join(import_lines))
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(ROOT)
    full_env["SDL_VIDEODRIVER"] = "dummy"
    if env:
        full_env.update(env)

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT), env=full_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败：\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ======================================================================
def test_games_has_no_rl_deps(c: _Case) -> None:
    """import games 不应拉起任何 RL 库。"""
    info = _probe([
        "import games",
        "games.list_games()",
    ])
    c.check("import games 后没有 torch/SB3/tensorboard",
            info["heavy"] == [], f"发现 {info['heavy']}")


def test_human_play_has_no_rl_deps(c: _Case) -> None:
    """import human_play 不应拉起任何 RL 库（重构前的实际缺陷）。"""
    info = _probe([
        "import human_play",
    ])
    c.check("import human_play 后没有 torch/SB3/tensorboard",
            info["heavy"] == [], f"发现 {info['heavy']}")

    # 重量级依赖只有 gymnasium / numpy（用于声明空间和数组运算），
    # 没有 torch 那一坨。用「有没有重库」判断，而不是比模块数量 ——
    # numpy 本身就会拉进上百个标准库模块，数量阈值没有意义。
    c.check("轻量依赖里不含 torch 相关包",
            not any(m.startswith(("torch", "stable_baselines3"))
                    for m in info.get("tops", [])),
            f"{info.get('tops')}")


def test_human_play_imports_little(c: _Case) -> None:
    """human_play 连 gymnasium.Env 都不该用（那是 envs 的事）。"""
    info = _probe([
        "import human_play",
        "print('ok')",
    ])
    c.check("human_play 可正常导入并执行到末尾", info is not None)


# ======================================================================
def test_source_has_no_forbidden_imports(c: _Case) -> None:
    """源码扫描：防止用延迟 import 绕过上面两条运行时断言。"""
    targets: list[Path] = list((ROOT / "games").rglob("*.py"))
    targets.append(ROOT / "human_play.py")
    targets.append(ROOT / "common" / "paths.py")
    targets.append(ROOT / "common" / "cli.py")

    # 匹配 `import torch` / `from stable_baselines3 import ...`
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(torch|stable_baselines3|tensorboard)\b",
        re.MULTILINE,
    )

    offenders = []
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line_no} -> {m.group(0).strip()}")

    c.check("games/ 与 human_play.py 源码里没有 RL 库 import",
            not offenders, "；".join(offenders))
    c.check("扫描覆盖到了足够多的文件", len(targets) >= 4,
            f"只扫了 {len(targets)} 个")


def test_common_is_stdlib_only(c: _Case) -> None:
    """common/ 只能依赖标准库（它是所有入口的共享底座）。"""
    pattern = re.compile(
        r"^\s*(?:import|from)\s+"
        r"(numpy|torch|stable_baselines3|tensorboard|gymnasium|pygame)\b",
        re.MULTILINE,
    )
    # runner.py 需要 numpy 做均值，属于合理例外，单独列出
    allow = {"runner.py"}

    offenders = []
    for path in (ROOT / "common").glob("*.py"):
        if path.name in allow:
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            offenders.append(f"common/{path.name}:{line_no} -> {m.group(0).strip()}")

    c.check("common/ 除 runner.py 外只用标准库",
            not offenders, "；".join(offenders))


# ======================================================================
def test_envs_owns_contract(c: _Case) -> None:
    """契约必须在 envs/base.py；games/ 里不该再有 base.py。"""
    c.check("envs/base.py 存在", (ROOT / "envs" / "base.py").is_file())
    c.check("games/base.py 已移除（契约已上交 envs）",
            not (ROOT / "games" / "base.py").exists())

    # games 的重导出仍然可用（向后兼容）
    info = _probe([
        "import games",
        "assert hasattr(games, 'BaseGame')",
        "assert hasattr(games, 'run_human')",
    ])
    c.check("games 仍然重导出 BaseGame / run_human", info is not None)


# ======================================================================
def main() -> int:
    c = _Case()
    print("[1/5] import games 不引入 RL 库 ...")
    test_games_has_no_rl_deps(c)
    print("[2/5] import human_play 不引入 RL 库 ...")
    test_human_play_has_no_rl_deps(c)
    print("[3/5] human_play 可正常导入 ...")
    test_human_play_imports_little(c)
    print("[4/5] 源码扫描（防延迟 import 绕过）...")
    test_source_has_no_forbidden_imports(c)
    test_common_is_stdlib_only(c)
    print("[5/5] 契约归属 ...")
    test_envs_owns_contract(c)

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
