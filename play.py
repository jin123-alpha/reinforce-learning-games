"""
play.py —— 加载权重并观看它玩
=============================

只做一件事：**加载某次训练的权重，跑给你看**。
（想看随机/专家对比，用 compare.py 的三窗口。）

权重定位规则
------------
    python play.py --game catcher                     # latest run 的 best
    python play.py --game catcher --run 20250101-120000_dqn
    python play.py --game catcher --weights final     # 用训练结束时的模型

--run 省略时读 models/<game>/latest.txt。
找不到时会明确告诉你有哪些 run 可选，或提示先去训练。

为什么 best / final 要显式区分
------------------------------
  best  = EvalCallback 在评估中表现最好的那个 checkpoint
  final = 训练最后一步的模型
两者差别常常很大（尤其 DQN 后期会退化）。README 里引用的分数
（DQN +3.02 / score 10.59）来自 best。如果 play.py 悄悄换成一个，
报告里的数字和屏幕上的表现就会对不上，所以这里默认 best 并把
实际加载的那个打印出来。

用法
----
    python play.py --game catcher
    python play.py --game dodger --run 20250101-093000_ppo --weights final
    python play.py --game catcher --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cli, paths
from common.paths import PathResolutionError
from common.runner import run_policy
from envs import make_env


def load_run_config(game: str, run_id: str) -> dict:
    """读 run 目录里的参数快照，用于重建当时的环境。

    这份快照是 train.py 写的，记录了那次训练实际用的 game_kwargs。
    即使 config/<game>/game.json 后来被改了，这里也能还原当时的环境 ——
    否则「用新参数跑旧权重」会得到无意义的画面。
    """
    import json

    path = paths.model_run_dir(game, run_id) / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_model(path: Path):
    """尝试按 PPO / DQN 顺序加载权重，返回 (model, 算法名)。"""
    from stable_baselines3 import DQN, PPO

    errors = []
    for cls in (PPO, DQN):
        try:
            return cls.load(str(path)), cls.__name__
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{cls.__name__}: {type(exc).__name__}")
    raise SystemExit(
        f"加载模型失败：{path}.zip\n  尝试过 —— {'; '.join(errors)}\n"
        f"该文件可能损坏，或不是 SB3 保存的格式。"
    )


def main(args) -> int:
    # ---- 1. 定位 run 和权重 ----
    try:
        run_id = paths.resolve_run(args.game, args.run)
        weights_path = paths.resolve_weights(args.game, run_id, args.weights)
    except PathResolutionError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    # ---- 2. 重建当时的环境参数 ----
    run_cfg = load_run_config(args.game, run_id)
    game_kwargs = dict(run_cfg.get("game_kwargs", {}))
    if args.set:
        game_kwargs.update(cli.parse_overrides(args.set))

    print(f"游戏      : {args.game}")
    print(f"run       : {run_id}"
          f"{'  （来自 latest.txt）' if args.run is None else ''}")
    print(f"权重      : {weights_path.name}  <{weights_path}.zip>")
    if run_cfg:
        print(f"训练算法  : {run_cfg.get('algo', '?')}"
              f"    步数: {run_cfg.get('train', {}).get('timesteps', '?')}")
    print(f"游戏参数  : {game_kwargs}")

    # ---- 3. 加载模型 ----
    model, algo = load_model(weights_path)
    print(f"实际加载  : {algo}")

    # ---- 4. 跑 ----
    game = make_env(args.game, render_mode="human", **game_kwargs).unwrapped_game

    def report(ep: int, stats: dict) -> None:
        print(f"[回合 {ep:3d}] score {stats['score']:8.2f} "
              f"| return {stats['return']:8.2f} | 步数 {stats['steps']}")

    print("\n关闭窗口或 Ctrl+C 退出\n")
    try:
        summary = run_policy(game, model, episodes=args.episodes,
                             seed_base=args.seed_base, on_episode_end=report)
    except KeyboardInterrupt:
        print("\n已中断")
        summary = {"n": 0, "score_mean": 0.0, "return_mean": 0.0}

    if summary.get("n"):
        print(f"\n共 {summary['n']} 回合 | "
              f"平均 score {summary['score_mean']:.2f} | "
              f"平均 return {summary['return_mean']:.2f}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="加载权重并观看它玩游戏")
    cli.add_game_arg(p, default="catcher")
    cli.add_run_arg(p)
    p.add_argument("--weights", type=str, default="best",
                   choices=["best", "final"],
                   help="best=评估最优 checkpoint（默认），final=训练结束模型")
    p.add_argument("--episodes", type=int, default=0, help="0 表示无限")
    p.add_argument("--seed-base", type=int, default=7000)
    cli.add_set_arg(p)
    p.add_argument("--list-games", action="store_true")
    args = p.parse_args()

    if args.list_games:
        print("可用游戏：" + "、".join(paths.available_games()))
        raise SystemExit(0)

    raise SystemExit(main(args))
