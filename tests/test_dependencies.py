"""
tests/test_dependencies.py
==========================
**依赖分层测试**：确认每个模块实际拉起的第三方包落在预期范围内。

与 test_decoupling.py 的分工
----------------------------
* test_decoupling.py  只关心「游戏层不许碰 RL 库」这一条硬约束。
* 本文件检查**全部模块**的分层，包括：
    - common/paths.py、common/cli.py 必须**零第三方依赖**（纯标准库）
    - common/runner.py 只允许多 numpy
    - games / envs / experts 只允许多 numpy + gymnasium(+pygame)
    - human_play 不许拉起 torch / SB3 / tensorboard
    - compare_child 允许 SB3（它是入口）

为什么值得测试
--------------
依赖方向是这次重构的核心成果，但它**看不见摸不着** ——
只要有人图方便在 games/ 里写一句 `import torch`，
所有测试都还会通过，只有「在没装 torch 的机器上玩不了游戏」
这种谁也不会立刻发现的问题。

本测试通过**子进程实测** import 后的 sys.modules 来判断，
而不是靠读源码猜。这样连传递依赖（比如 A import B、B import torch）
也能抓到。

运行：
    python tests/test_dependencies.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
# 环境固有噪音：不是我们代码引入的，也不代表分层被破坏
#   _distutils_hack        setuptools 在 sitecustomize 里的自动注入
#   typing_extensions      numpy / gymnasium 的类型标注依赖
#   farama_notifications   gymnasium 启动时打印的提示
NOISE = {
    "_distutils_hack",
    "typing_extensions",
    "farama_notifications",
}

HEAVY = {"torch", "stable_baselines3", "tensorboard"}

# 模块 -> (允许的第三方依赖, 说明)
RULES: dict[str, tuple[set[str], str]] = {
    "common.paths":  (set(), "公共路径工具：必须纯标准库"),
    "common.cli":    (set(), "公共 CLI 工具：必须纯标准库"),
    "common.runner": ({"numpy"}, "运行循环：只允许多 numpy"),

    "envs.base":     ({"numpy", "gymnasium"}, "接口契约"),
    "envs.game_env": ({"numpy", "gymnasium"}, "gym 适配器"),
    "envs.normalize": ({"numpy", "gymnasium"}, "归一化 Wrapper"),

    "games":         ({"numpy", "gymnasium"}, "游戏包"),
    "games.catcher": ({"numpy", "gymnasium"}, "接球游戏"),
    "games.dodger":  ({"numpy", "gymnasium"}, "躲避游戏"),
    "games.snake":   ({"numpy", "gymnasium"}, "贪吃蛇游戏"),

    "experts":         ({"numpy"}, "专家包"),
    "experts.catcher": ({"numpy"}, "接球专家"),
    "experts.dodger":  ({"numpy"}, "躲避专家"),
    "experts.snake":   ({"numpy"}, "贪吃蛇专家"),

    "human_play": ({"numpy", "gymnasium"}, "人类游玩：绝不许碰 RL 库"),
    "play":       ({"numpy", "gymnasium"}, "观看"),
    "train":      ({"numpy", "gymnasium"}, "训练"),
    "compare":    ({"numpy", "gymnasium"}, "对比：父进程只做编排"),
}

# 允许 SB3 的入口（加载权重时必须）
ALLOW_HEAVY = {"compare_child", "play", "train"}


_PROBE = r"""
import sys, json, sysconfig
__import__("{mod}")
site = sysconfig.get_paths()["purelib"]
third = []
for name, m in list(sys.modules.items()):
    top = name.split(".")[0]
    f = getattr(m, "__file__", None) or ""
    if f and ("site-packages" in f or "dist-packages" in f):
        third.append(top)
print(json.dumps(sorted(set(third))))
"""


def probe(mod: str) -> tuple[list[str] | None, str]:
    """在干净子进程里 import 该模块，返回 (第三方包列表, 错误信息)。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["SDL_VIDEODRIVER"] = "dummy"

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(mod=mod)],
        capture_output=True, text=True, cwd=str(ROOT), env=env,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()
        return None, tail[-1] if tail else "未知错误"
    return json.loads(proc.stdout.strip().splitlines()[-1]), ""


# ======================================================================
def test_layering(c: _Case) -> None:
    """逐个模块核对第三方依赖是否越界。"""
    for mod, (allowed, desc) in RULES.items():
        deps, err = probe(mod)
        if deps is None:
            c.check(f"{mod} 可导入", False, err)
            continue

        actual = {d for d in deps if d not in NOISE}
        extra = actual - allowed
        c.check(f"{mod} 依赖不越界（{desc}）", not extra,
                f"多出 {sorted(extra)}；实际 {sorted(actual)}")


def test_common_is_stdlib(c: _Case) -> None:
    """common/paths.py 与 common/cli.py 必须零第三方依赖。"""
    for mod in ("common.paths", "common.cli"):
        deps, err = probe(mod)
        if deps is None:
            c.check(f"{mod} 可导入", False, err)
            continue
        actual = {d for d in deps if d not in NOISE}
        c.check(f"{mod} 是纯标准库", not actual, f"实际 {sorted(actual)}")


def test_no_heavy_in_game_layers(c: _Case) -> None:
    """游戏 / 环境 / 专家三层都不许出现 torch 系列。"""
    for mod in ("games", "games.catcher", "games.dodger", "games.snake",
                "envs", "envs.base", "envs.game_env", "envs.normalize",
                "experts", "experts.catcher", "experts.dodger",
                "experts.snake", "common.paths", "common.cli"):
        deps, err = probe(mod)
        if deps is None:
            c.check(f"{mod} 可导入", False, err)
            continue
        bad = HEAVY & set(deps)
        c.check(f"{mod} 不含 torch/SB3/tensorboard", not bad,
                f"发现 {sorted(bad)}")


def test_human_play_needs_no_rl(c: _Case) -> None:
    """人类游玩这条路径不需要任何 RL 依赖（核心卖点）。"""
    deps, err = probe("human_play")
    if deps is None:
        c.check("human_play 可导入", False, err)
        return

    bad = HEAVY & set(deps)
    c.check("human_play 不需要 RL 依赖", not bad, f"发现 {sorted(bad)}")

    # 反过来确认它确实只用了轻量依赖
    actual = {d for d in deps if d not in NOISE}
    c.check("human_play 依赖只有 numpy/gymnasium",
            actual <= {"numpy", "gymnasium", "pygame"}, f"实际 {sorted(actual)}")


def test_entry_points_importable(c: _Case) -> None:
    """四个入口 + 子进程入口都能导入。"""
    for mod in ("train", "play", "human_play", "compare", "compare_child"):
        deps, err = probe(mod)
        c.check(f"{mod} 可导入", deps is not None, err)


def test_train_defers_torch(c: _Case) -> None:
    """关键设计：import train 时**不**立即拉起 torch。

    训练脚本把 SB3 的导入放进函数里（延迟导入），
    所以 `--list-games`、`--help` 这类命令不需要等 torch。
    如果哪天有人把 import 提到顶层，这个测试会失败。
    """
    deps, err = probe("train")
    if deps is None:
        c.check("train 可导入", False, err)
        return
    c.check("import train 不会立刻加载 torch（延迟导入生效）",
            "torch" not in deps, f"实际拉起了 {sorted(HEAVY & set(deps))}")


# ======================================================================
def main() -> int:
    c = _Case()
    print("[1/6] 逐模块依赖分层 ...")
    test_layering(c)
    print("[2/6] common/ 是纯标准库 ...")
    test_common_is_stdlib(c)
    print("[3/6] 游戏/环境/专家层不含 RL 库 ...")
    test_no_heavy_in_game_layers(c)
    print("[4/6] 人类游玩不需要 RL 依赖 ...")
    test_human_play_needs_no_rl(c)
    print("[5/6] 全部入口可导入 ...")
    test_entry_points_importable(c)
    print("[6/6] 训练脚本延迟导入 torch ...")
    test_train_defers_torch(c)

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
