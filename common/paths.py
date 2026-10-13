"""
common/paths.py
===============
**全项目唯一的路径解析入口。**

核心规则（整个框架都遵守它）：

    先确定游戏名，再在该游戏自己的子目录里找文件。

也就是说 `--game catcher --config 1.json` 永远解析成
`config/catcher/1.json`，**绝不会**变成当前目录下的 `./1.json`。
这样每个游戏的配置、模型、日志天然隔离，互不干扰。

目录布局：

    catcher/
    ├── config/<game>/
    │   ├── game.json            游戏参数（唯一真相）
    │   └── <name>.json          算法超参，内部用 "game_params" 引用 game.json
    ├── models/<game>/
    │   ├── latest.txt           指向最近一次训练 run
    │   └── <run_id>/{config.json, best/, final.zip}
    └── logs/<game>/
        └── <run_id>/{tb/, evaluations.npz, train.log}

为什么要有这个模块
------------------
重构前 `Path("models") / game` 这类字面量散落在 train.py / play.py /
compare.py 里共 5 处。这种写法**错了不会报错**，只会静默读错目录，
所以必须收敛到一个地方，并且配上单元测试。
"""
from __future__ import annotations

import re
from pathlib import Path

# ----------------------------------------------------------------------
# 项目根目录
# ----------------------------------------------------------------------
# 本文件位于 <root>/common/paths.py，所以 root 是它的上两级。
# 用 __file__ 推导而不是 os.getcwd()，这样从任何目录调用都一样。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_ROOT = PROJECT_ROOT / "config"
MODELS_ROOT = PROJECT_ROOT / "models"
LOGS_ROOT = PROJECT_ROOT / "logs"
GAMES_ROOT = PROJECT_ROOT / "games"
EXPERTS_ROOT = PROJECT_ROOT / "experts"

LATEST_POINTER = "latest.txt"

# 游戏名 / run 名只允许这些字符，防止路径穿越和跨平台非法文件名
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PathResolutionError(Exception):
    """路径解析失败。带上「有哪些可用选项」方便直接照抄命令。"""


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _validate(kind: str, name: str, available: list[str] | None = None) -> str:
    """校验单个名字片段（游戏名、run 名、配置名）。"""
    if not name or not _SAFE_NAME.match(name):
        raise PathResolutionError(
            f"{kind} {name!r} 非法：只能包含字母、数字、点、下划线、连字符，"
            f"且必须以字母或数字开头（防止 ../ 这类路径穿越）。"
        )
    if available is not None and name not in available:
        hint = "、".join(available) if available else "（无）"
        raise PathResolutionError(f"未知{kind} {name!r}。可用：{hint}")
    return name


def _subdirs(path: Path) -> list[str]:
    """列出目录下的子目录名（已排序），目录不存在时返回空表。"""
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def _json_stems(path: Path) -> list[str]:
    """列出目录下的 .json 文件名（不含扩展名，已排序）。"""
    if not path.is_dir():
        return []
    return sorted(p.stem for p in path.glob("*.json"))


# ----------------------------------------------------------------------
# 游戏名枚举（给 argparse 的 choices 和错误提示用）
# ----------------------------------------------------------------------
def available_games() -> list[str]:
    """所有「有 games/<name>/ 子目录」的游戏名。

    注意：这里扫的是**目录**而不是导入 games 包，因此可以在
    `--list-games` 这种只想看列表的场景下避免拉起 pygame/numpy。
    """
    return [d for d in _subdirs(GAMES_ROOT) if not d.startswith("_")]


# ----------------------------------------------------------------------
# 目录解析：每个 root 一个函数
# ----------------------------------------------------------------------
def game_dir(root: Path, game: str) -> Path:
    """任意 root（config/models/logs）下某游戏的目录。"""
    _validate("游戏", game)
    return root / game


def config_dir(game: str) -> Path:
    return game_dir(CONFIG_ROOT, game)


def models_dir(game: str) -> Path:
    return game_dir(MODELS_ROOT, game)


def logs_dir(game: str) -> Path:
    return game_dir(LOGS_ROOT, game)


def run_dir(root: Path, game: str, run_id: str) -> Path:
    """models/logs 下某次训练 run 的目录。"""
    _validate("run", run_id)
    return game_dir(root, game) / run_id


def model_run_dir(game: str, run_id: str) -> Path:
    return run_dir(MODELS_ROOT, game, run_id)


def log_run_dir(game: str, run_id: str) -> Path:
    return run_dir(LOGS_ROOT, game, run_id)


# ----------------------------------------------------------------------
# 文件解析：名字 -> 绝对路径（带「找不到就告诉你有哪些」的错误）
# ----------------------------------------------------------------------
def resolve_config(game: str, name: str | None = None,
                   default: str = "game") -> Path:
    """解析 `config/<game>/<name>.json`。

    name=None 时用 default（`game` -> game.json），
    这样 `python train.py --game catcher` 不传 --config 也能跑。
    """
    stem = default if name is None else name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]

    d = config_dir(game)
    path = d / f"{stem}.json"
    if not path.is_file():
        have = _json_stems(d)
        hint = "、".join(f"{s}.json" for s in have) if have else "（该游戏还没有任何配置文件）"
        raise PathResolutionError(
            f"找不到配置 {path}\nconfig/{game}/ 下现有：{hint}"
        )
    return path


def resolve_game_params(game: str, name: str | None = None) -> Path:
    """解析游戏参数文件，默认 `config/<game>/game.json`。"""
    return resolve_config(game, name, default="game")


def resolve_run(game: str, run_id: str | None = None,
                root: Path | None = None) -> str:
    """解析 run 名：显式给定就用它，否则读 latest.txt。

    返回 run 名（字符串），不是路径 —— 因为 models/ 和 logs/ 下的
    同一个 run_id 要能各自定位。

    root 默认取当前的 MODELS_ROOT（**运行时读取**，不写成默认参数值：
    写成 `root=MODELS_ROOT` 会在定义时就绑定常量，测试里替换根目录会失效）。
    """
    root = root if root is not None else MODELS_ROOT
    base = game_dir(root, game)

    if run_id is not None:
        _validate("run", run_id)
        if not (base / run_id).is_dir():
            have = _subdirs(base)
            hint = "、".join(have) if have else "（还没有任何 run，先训练一次）"
            raise PathResolutionError(
                f"找不到 run {run_id!r}（{base / run_id} 不存在）。\n"
                f"{root.name}/{game}/ 下可用 run：{hint}"
            )
        return run_id

    pointer = base / LATEST_POINTER
    if not pointer.is_file():
        have = _subdirs(base)
        if have:
            raise PathResolutionError(
                f"{pointer} 不存在，无法确定「最近一次训练」。\n"
                f"可用 run：{'、'.join(have)}\n"
                f"用 --run <name> 显式指定，或先跑一次 train.py 生成指针。"
            )
        raise PathResolutionError(
            f"游戏 {game!r} 还没有训练过（{base} 为空）。\n"
            f"先运行：python train.py --game {game}"
        )

    name = pointer.read_text(encoding="utf-8").strip()
    if not name:
        raise PathResolutionError(f"{pointer} 是空的，无法确定最近一次训练。")

    if not (base / name).is_dir():
        have = _subdirs(base)
        hint = "、".join(have) if have else "（目录为空）"
        raise PathResolutionError(
            f"{pointer} 指向 {name!r}，但该目录不存在（可能被删了）。\n"
            f"现有 run：{hint}\n"
            f"用 --run <name> 显式指定，或重跑 train.py 修正指针。"
        )
    return name


def resolve_weights(game: str, run_id: str | None = None,
                    weights: str = "best") -> Path:
    """定位某次 run 的权重文件（不含 .zip 后缀）。

    weights:
        best  -> <run>/best/best_model    （EvalCallback 存的最优）
        final -> <run>/final              （训练结束时的模型）
        best 缺失时自动回退到 final，并**在返回值上标注**，
        避免「报告里写 best、屏幕上放 final」这种不一致。
    """
    run = resolve_run(game, run_id)
    d = model_run_dir(game, run)

    candidates = {
        "best": d / "best" / "best_model",
        "final": d / "final",
    }
    if weights not in candidates:
        raise PathResolutionError(
            f"未知权重类型 {weights!r}，可选：{'、'.join(candidates)}"
        )

    order = [weights] + [k for k in ("best", "final") if k != weights]
    for key in order:
        path = candidates[key]
        if Path(str(path) + ".zip").is_file():
            if key != weights:
                print(f"[提示] {weights} 权重不存在，已回退到 {key}：{path}.zip")
            return path

    raise PathResolutionError(
        f"run {run!r} 里既没有 best_model.zip 也没有 final.zip（{d}）。\n"
        f"这个 run 可能是训练中断留下的。"
    )


def expert_path(game: str) -> Path:
    """专家策略文件 `experts/<game>.py`（可能不存在，调用方自行判断）。"""
    _validate("游戏", game)
    return EXPERTS_ROOT / f"{game}.py"


# ----------------------------------------------------------------------
# 创建目录（训练时用）
# ----------------------------------------------------------------------
def ensure_run_dirs(game: str, run_id: str) -> tuple[Path, Path]:
    """建好某次 run 的 models/ 和 logs/ 目录，返回两者。"""
    m = model_run_dir(game, run_id)
    l = log_run_dir(game, run_id)
    (m / "best").mkdir(parents=True, exist_ok=True)
    (l / "tb").mkdir(parents=True, exist_ok=True)
    return m, l


def write_latest_pointer(game: str, run_id: str) -> Path:
    """把 models/<game>/latest.txt 指向给定 run。"""
    base = models_dir(game)
    base.mkdir(parents=True, exist_ok=True)
    pointer = base / LATEST_POINTER
    pointer.write_text(run_id + "\n", encoding="utf-8")
    return pointer


def new_run_id(algo: str, when=None) -> str:
    """生成 run 名：`20250101-143022_dqn`。

    时间在前 -> 目录列表按名字排序就是时间顺序；
    算法名在后 -> 一眼能看出这个 run 用的什么算法。
    """
    import datetime as _dt

    when = when or _dt.datetime.now()
    _validate("算法", algo)
    return f"{when:%Y%m%d-%H%M%S}_{algo}"
