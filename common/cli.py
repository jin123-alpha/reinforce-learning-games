"""
common/cli.py
=============
四个入口脚本（train / play / human_play / compare）共享的 CLI 工具。

**为什么单独一个模块**：`parse_overrides` 原本住在 train.py 里，
而 train.py 顶层 `from stable_baselines3 import DQN, PPO`。
结果 `human_play.py` 为了拿一个 JSON 解析函数，被迫把 torch 也拉了起来 ——
「人类游玩只依赖 games」这句话当场就不成立了。

搬到这里之后依赖链变成：

    common/cli.py -> json / ast（标准库）
    games/        -> numpy / gymnasium.spaces / pygame
    human_play.py -> games + common        ← 不碰 torch

`tests/test_decoupling.py` 会断言这一点。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 兄弟模块导入：既支持 `from common.cli import ...`（包内），
# 也支持把 common/ 加进 sys.path 后 `import cli`（扁平脚本）。
# ----------------------------------------------------------------------
try:
    from . import paths as _paths
except ImportError:                                  # 直接以脚本形式使用 common/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths as _paths

# ----------------------------------------------------------------------
# KEY=VALUE 解析
# ----------------------------------------------------------------------
def parse_value(raw: str):
    """把字符串解析成合适的 Python 值。

    用 ast.literal_eval 而不是 json.loads：
        json.loads("true")  -> True      （json 小写）
        literal_eval("true")-> ValueError （Python 是 True）
    命令行里两种写法都有人用，所以先试 json，再回退 literal_eval，
    最后退化成字符串。这样 `--set rich_obs=true` 和 `=True` 都能用。
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # 退化成字符串时返回 strip 过的版本：
        # `--set  name =  relu ` 里的空格是 shell 噪音，不该进游戏参数。
        return text


def parse_overrides(pairs: list[str] | None) -> dict:
    """把 ["num_balls=5", "rich_obs=true"] 解析成 {'num_balls': 5, ...}

    保持与旧 train.py 里的同名函数行为兼容（旧代码是 json.loads + 字符串回退）。
    """
    out: dict = {}
    for item in pairs or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(
                f"--set 参数 {item!r} 格式错误，应该是 KEY=VALUE，例：--set num_balls=5"
            )
        key = key.strip()
        if not key:
            raise SystemExit(f"--set 参数 {item!r} 缺少键名。")
        out[key] = parse_value(val)
    return out


def merge_overrides(*layers: dict | None) -> dict:
    """按顺序合并多层参数，后面的覆盖前面的。None 会被跳过。"""
    merged: dict = {}
    for layer in layers:
        if layer:
            merged.update(layer)
    return merged


# ----------------------------------------------------------------------
# 共享的参数定义（保证四个入口对同一个参数的说法一致）
# ----------------------------------------------------------------------
def add_game_arg(p, default: str | None = None):
    """--game：所有入口共有的第一个参数。"""
    games = _paths.available_games()
    p.add_argument(
        "--game", type=str, default=default or (games[0] if games else "catcher"),
        choices=games or None,
        help="游戏名（对应 games/<name>/ 子目录），可用：" + "、".join(games or ["?"]),
    )


def add_set_arg(p):
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="临时覆盖游戏参数，可重复。例：--set num_balls=5",
    )


def add_config_arg(p, default: str = "1"):
    p.add_argument(
        "--config", type=str, default=default, metavar="NAME",
        help=f"config/<game>/NAME.json（相对该游戏目录解析，默认 {default}.json）",
    )


def add_run_arg(p):
    p.add_argument(
        "--run", type=str, default=None, metavar="RUN_ID",
        help="models/<game>/RUN_ID；省略则读 latest.txt",
    )


# ----------------------------------------------------------------------
# 配置加载
# ----------------------------------------------------------------------
def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"文件不存在：{path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} 不是合法 JSON：{exc}")


def load_game_params(game: str, name: str | None = None,
                     cli_sets: list[str] | None = None) -> dict:
    """读游戏参数并叠加命令行 --set。

    这是「游戏参数唯一真相」的读取点：config/<game>/game.json。
    算法 config 里的 "game_params" 也最终指到这里（可指向别的文件）。
    """
    try:
        path = _paths.resolve_game_params(game, name)
    except Exception as exc:
        # 没有 game.json 不算致命：直接用 --set，或让游戏用自身默认值
        print(f"[提示] {exc}")
        return parse_overrides(cli_sets)

    params = dict(load_json(path).get("game_kwargs", {}))
    params.update(parse_overrides(cli_sets))
    return params


def load_algo_config(game: str, name: str, cli_sets: list[str] | None = None
                     ) -> tuple[dict, dict]:
    """读 `config/<game>/<name>.json`，返回 (完整配置, 游戏参数)。

    算法 config 的结构：

        {
          "game_params": "game.json",     // 相对**本文件所在目录**解析
          "algo": "dqn",
          "algo_kwargs": {...}
        }

    返回的游戏参数已经叠加了 --set。
    """
    path = _paths.resolve_config(game, name)
    cfg = load_json(path)

    # game_params 的相对路径基准是 config 文件自己所在的目录，不是 CWD。
    # 这样从任何目录运行、或 config 被搬动，行为都一致。
    ref = cfg.get("game_params")
    if ref:
        ref_path = (path.parent / ref).resolve()
        if not ref_path.is_file():
            raise SystemExit(
                f"{path.name} 里引用的游戏参数 {ref!r} 不存在：{ref_path}\n"
                f"（相对路径以 {path.parent} 为基准解析）"
            )
        params = dict(load_json(ref_path).get("game_kwargs", {}))
    else:
        params = {}

    params.update(parse_overrides(cli_sets or []))
    return cfg, params
