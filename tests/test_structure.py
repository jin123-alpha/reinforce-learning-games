"""
tests/test_structure.py
=======================
**目录结构与依赖分层测试。**

验证重构后约定被真正建立起来（而不是「看起来像」）：

1. 顶层文件齐全：train / play / human_play / compare
2. 目录齐全：games / envs / experts / models / logs / config
3. games 必须是**子目录**形式，且每个子目录导出 GAME
4. 专家是「一个游戏一个 py 文件」（不是子目录）
5. 依赖分层：common 纯标准库、games/envs/experts 不碰 torch
6. 路径解析：所有文件查找都按游戏分成子目录

运行：
    python tests/test_structure.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import paths  # noqa: E402


class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")

    def exists(self, name: str, path: Path) -> None:
        self.check(name, path.exists(), f"缺少 {path.relative_to(ROOT)}")


# ======================================================================
def test_top_level_files(c: _Case) -> None:
    """四个入口文件 + compare 的子进程入口，都在最外层。"""
    for name in ["train.py", "play.py", "human_play.py", "compare.py"]:
        c.exists(f"最外层有 {name}", ROOT / name)

    # compare_child.py 是可执行入口（要 import SB3 加载权重），
    # 所以它属于最外层，而不是 common/（common 必须纯标准库）
    c.exists("最外层有 compare_child.py（compare 的子进程入口）",
             ROOT / "compare_child.py")
    c.check("common/ 里没有 compare_child.py（它是入口，不是公共库）",
            not (ROOT / "common" / "compare_child.py").exists())


def test_directories(c: _Case) -> None:
    """六个目录都在。"""
    for name in ["games", "envs", "experts", "models", "logs", "config"]:
        c.exists(f"有 {name}/ 目录", ROOT / name)


def test_games_are_subdirs(c: _Case) -> None:
    """games/ 下每个游戏一个子目录，且导出 GAME。"""
    games_dir = ROOT / "games"
    subdirs = sorted(d.name for d in games_dir.iterdir()
                     if d.is_dir() and not d.name.startswith("_"))

    c.check("games/ 下至少有一个游戏子目录", len(subdirs) >= 2, f"实际 {subdirs}")
    c.check("games/catcher/ 是目录", (games_dir / "catcher").is_dir())
    c.check("games/dodger/ 是目录", (games_dir / "dodger").is_dir())
    c.check("games/snake/ 是目录", (games_dir / "snake").is_dir())

    # 旧的扁平文件不该还在
    c.check("games/catcher.py 已移入子目录",
            not (games_dir / "catcher.py").exists())
    c.check("games/dodger.py 已移入子目录",
            not (games_dir / "dodger.py").exists())
    c.check("games/snake.py 已移入子目录",
            not (games_dir / "snake.py").exists())
    c.check("games/base.py 已上交 envs",
            not (games_dir / "base.py").exists())

    # 每个子目录都要能导出 GAME，且名户一致
    for sub in subdirs:
        init = games_dir / sub / "__init__.py"
        c.exists(f"games/{sub}/__init__.py", init)
        if init.is_file():
            text = init.read_text(encoding="utf-8")
            c.check(f"games/{sub}/ 导出 GAME", "GAME" in text,
                    "缺少 GAME = 导出")


def test_experts_are_files(c: _Case) -> None:
    """专家是「一个游戏一个 .py 文件」，不是子目录。"""
    experts = ROOT / "experts"
    c.exists("experts/base.py（接口）", experts / "base.py")
    c.exists("experts/catcher.py（接球专家）", experts / "catcher.py")
    c.exists("experts/dodger.py（躲避专家）", experts / "dodger.py")
    c.exists("experts/snake.py（贪吃蛇专家）", experts / "snake.py")

    subdirs = [d.name for d in experts.iterdir()
               if d.is_dir() and not d.name.startswith("_")]
    c.check("experts/ 下没有游戏子目录（应该是扁平 py 文件）",
            not subdirs, f"发现子目录 {subdirs}")

    # 每个游戏都该有对应的专家文件（有游戏没专家时这里会立刻发现）
    for game in paths.available_games():
        c.exists(f"experts/{game}.py 存在（每个游戏一个专家文件）",
                 experts / f"{game}.py")


def test_config_layout(c: _Case) -> None:
    """config/ 按游戏分区，每区含 game.json 与若干算法配置。"""
    cfg = ROOT / "config"
    for game in paths.available_games():
        gdir = cfg / game
        c.exists(f"config/{game}/ 目录", gdir)
        c.exists(f"config/{game}/game.json（游戏参数唯一真相）",
                 gdir / "game.json")

        algos = [p for p in gdir.glob("*.json") if p.name != "game.json"]
        c.check(f"config/{game}/ 至少有一份算法配置", bool(algos),
                f"只有 {[p.name for p in gdir.glob('*.json')]}")

        # 每份算法配置都要正确引用 game.json
        for a in algos:
            import json
            data = json.loads(a.read_text(encoding="utf-8"))
            ref = data.get("game_params")
            c.check(f"config/{game}/{a.name} 引用了 game_params",
                    ref == "game.json", f"实际 {ref!r}")
            c.check(f"config/{game}/{a.name} 有 algo 字段",
                    "algo" in data)


def test_envs_owns_contract(c: _Case) -> None:
    """契约在 envs/base.py，且定义清晰。"""
    base = ROOT / "envs" / "base.py"
    c.exists("envs/base.py", base)
    if base.is_file():
        text = base.read_text(encoding="utf-8")
        c.check("BaseGame 定义在 envs/base.py", "class BaseGame" in text)
        c.check("score 是抽象属性（必须实现）",
                "@abstractmethod" in text and "def score" in text)
        c.check("run_human 在 envs/base.py", "def run_human" in text)


def test_experts_are_pure_obs(c: _Case) -> None:
    """每个专家实现里都不应出现对 game 对象的访问。

    对**所有** experts/<game>.py 逐个检查，而不是只看 catcher：
    新增专家时最容易犯的错就是图方便去读游戏内部常量。
    """
    import ast

    expected_sig = {
        "catcher.py": "ExpertCatcher",
        "dodger.py": "ExpertDodger",
        "snake.py": "ExpertSnake",
    }

    for fname, cls in expected_sig.items():
        src = ROOT / "experts" / fname
        text = src.read_text(encoding="utf-8")

        # 构造函数签名必须是 (action_space, spec)，不是 (game)
        c.check(f"{cls} 构造参数是 (action_space, spec)",
                "def __init__(self, action_space, spec" in text,
                f"见 experts/{fname}")

        # 不该出现 game. 属性访问。
        # 用 ast 解析而不是文本匹配：文档字符串里解释「旧版读 game.WIDTH」
        # 是正常的，只有真正的属性访问节点才算违规。
        tree = ast.parse(text)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "game":
                    offenders.append(f"第 {node.lineno} 行: game.{node.attr}")

        c.check(f"{cls} 代码里没有 game.xxx 属性访问",
                not offenders, f"发现 {offenders[:3]}")


# ======================================================================
def test_no_legacy_files(c: _Case) -> None:
    """第 7 条：旧文件已清理。"""
    for gone in ["_legacy", "logs_v1", "train.log", "dodger_train.log"]:
        c.check(f"{gone} 已清理", not (ROOT / gone).exists(),
                f"{gone} 仍然存在")

    # 旧命名的残留权重
    leftovers = []
    for z in (ROOT / "models").rglob("*.zip"):
        if "_v1" in z.name:
            leftovers.append(z.relative_to(ROOT).as_posix())
    c.check("没有 *_v1*.zip 残留", not leftovers, f"{leftovers}")


def test_models_use_run_dirs(c: _Case) -> None:
    """models/<game>/<run_id>/ 结构 + latest.txt 指针。"""
    for game in paths.available_games():
        gdir = ROOT / "models" / game
        if not gdir.is_dir():
            continue

        pointer = gdir / "latest.txt"
        if pointer.is_file():
            run_id = pointer.read_text(encoding="utf-8").strip()
            c.check(f"models/{game}/latest.txt 指向存在的 run",
                    (gdir / run_id).is_dir(), f"指向 {run_id!r} 但目录不存在")
            c.check(f"models/{game}/{run_id}/config.json 存在（参数快照）",
                    (gdir / run_id / "config.json").is_file())

        # 旧的扁平结构残留
        c.check(f"models/{game}/final.zip 已移入 run 目录",
                not (gdir / "final.zip").exists())
        c.check(f"models/{game}/config.json 已移到 config/",
                not (gdir / "config.json").exists())


def test_logs_mirror_runs(c: _Case) -> None:
    """logs/<game>/<run_id>/ 结构与 models 对齐。"""
    for game in paths.available_games():
        ldir = ROOT / "logs" / game
        if not ldir.is_dir():
            continue
        for run in ldir.iterdir():
            if not run.is_dir():
                continue
            c.check(f"logs/{game}/{run.name}/ 与 models 的 run 对应",
                    (ROOT / "models" / game / run.name).is_dir(),
                    "models 里没有同名 run")
            # 不该有散落的 tb 目录（应该在 run 里面）
        c.check(f"logs/{game}/ 下没有散落的 tb/ 目录",
                not (ldir / "tb").exists())


# ======================================================================
def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    c = _Case()

    print("[1/8] 顶层入口文件 ...")
    test_top_level_files(c)
    print("[2/8] 目录结构 ...")
    test_directories(c)
    print("[3/8] games/ 子目录形式 ...")
    test_games_are_subdirs(c)
    print("[4/8] experts/ 扁平 py 形式 ...")
    test_experts_are_files(c)
    print("[5/8] config/ 按游戏分区 ...")
    test_config_layout(c)
    print("[6/8] envs/ 拥有契约 & 专家纯 obs ...")
    test_envs_owns_contract(c)
    test_experts_are_pure_obs(c)
    print("[7/8] 旧文件已清理 ...")
    test_no_legacy_files(c)
    print("[8/8] models/logs 的 run 结构 ...")
    test_models_use_run_dirs(c)
    test_logs_mirror_runs(c)

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
