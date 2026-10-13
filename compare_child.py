"""
compare_child.py
================
compare.py 的**子进程入口**：一个进程 = 一个窗口 = 一种策略。

本文件刻意放在最外层（与 train / play / human_play / compare 并列），
而不是塞进 common/ —— 它是一个**可执行入口**，需要 import
stable_baselines3 来加载权重；而 common/ 是所有入口共享的纯工具层，
必须保持只依赖标准库（tests/test_decoupling.py 会断言这件事）。

为什么必须用子进程
------------------
pygame 的 `display.set_mode()` 是**全局单例**：同一个进程里
再调一次只会替换掉当前窗口，不可能同时开两个。
而且 `pygame.display.quit()` / `pygame.quit()` 也是全局的 ——
一个游戏实例 close() 会把其它窗口一起干掉。

所以「三个窗口」在单进程里是架构上做不到的。
用子进程后每个窗口各自持有独立的 pygame 状态，互不干扰。

本文件被 compare.py 用 `sys.executable compare_child.py ...` 拉起，
通过**文件**汇报结果（而不是 stdout）：子进程被关窗口杀掉时
stdout 可能来不及 flush，而结果文件是逐回合追加的，
中途退出也能拿到已完成的部分。

窗口排布
--------
由父进程通过环境变量 SDL_VIDEO_WINDOW_POS 指定，例如 "820,100"。
pygame/SDL 会读它来决定窗口出现在屏幕的哪个位置，
这样三个窗口就自然「从左至右」排开了。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cli, paths
from common.runner import run_policy
from envs import make_env


# ----------------------------------------------------------------------
def build_policy(kind: str, game_name: str, game, run_id: str | None,
                 weights: str):
    """构造策略对象。

    三种策略在这里被**一视同仁**地处理：都只需要有 .predict()，
    这正是「专家是纯 obs->action」这个决定带来的好处 ——
    调用方不需要为专家写任何特殊分支。
    """
    if kind == "random":
        from experts import RandomPolicy
        return RandomPolicy(game.action_space), "随机策略"

    if kind == "expert":
        from experts import has_expert, make_expert
        if not has_expert(game_name):
            raise SystemExit(f"__NO_EXPERT__:{game_name}")
        spec = game.obs_spec()
        return make_expert(game_name, game.action_space, spec), "专家策略（无训练）"

    # model：加载权重
    from stable_baselines3 import DQN, PPO
    path = paths.resolve_weights(game_name, run_id, weights)
    for cls in (PPO, DQN):
        try:
            return cls.load(str(path)), f"{cls.__name__}（{path.name}）"
        except Exception:
            continue
    raise SystemExit(f"加载模型失败：{path}.zip")


# ----------------------------------------------------------------------
def wait_for_turn(sync_dir: Path, episode: int, timeout: float = 120.0) -> None:
    """回合屏障：等所有窗口都到达同一回合再一起开始。

    为什么需要它：三个策略的水平差很多（随机几回合就死、专家能活很久），
    如果各跑各的，第一回合之后窗口之间就完全对不上了 ——
    会看到左边在打第 8 回合、右边还在第 2 回合，根本没法对照。

    实现用「标记文件」而不是 multiprocessing.Barrier：
    进程是被 sys.executable 拉起的独立进程，不是 fork 出来的，
    共享不了内存里的同步原语。文件方式也顺带可调试。
    """
    if sync_dir is None:
        return

    flag = sync_dir / f"ep{episode:04d}_{os.getpid()}"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("ready", encoding="utf-8")

    deadline = time.time() + timeout
    while time.time() < deadline:
        # 每个回合的参与者在拉起时已经确定，用固定数量的标记判断
        ready = list(sync_dir.glob(f"ep{episode:04d}_*"))
        if len(ready) >= _expected_peers():
            return
        time.sleep(0.02)


def _expected_peers() -> int:
    return int(os.environ.get("COMPARE_PEERS", "3"))


# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--game", required=True)
    p.add_argument("--policy", required=True,
                   choices=["random", "expert", "model"])
    p.add_argument("--run", default=None)
    p.add_argument("--weights", default="best", choices=["best", "final"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=7000)
    p.add_argument("--result", required=True, help="结果 jsonl 路径")
    p.add_argument("--sync-dir", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    # 读 run 参数快照，保证三个窗口用**完全一样**的游戏参数
    game_kwargs = {}
    if args.run:
        cfg_path = paths.model_run_dir(args.game, args.run) / "config.json"
        if cfg_path.is_file():
            game_kwargs = json.loads(
                cfg_path.read_text(encoding="utf-8")).get("game_kwargs", {})
    if not game_kwargs:
        try:
            game_kwargs = cli.load_game_params(args.game)
        except Exception:
            game_kwargs = {}

    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        game_env = make_env(args.game, render_mode="human", **game_kwargs)
        game = game_env.unwrapped_game
        policy, label = build_policy(args.policy, args.game, game,
                                     args.run, args.weights)
    except SystemExit as exc:
        # 没有专家策略：写一个明确的标记让父进程显示占位窗口
        if str(exc).startswith("__NO_EXPERT__"):
            result_path.write_text(
                json.dumps({"skipped": True, "reason": "no_expert",
                            "label": args.label}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            return 0
        raise

    # 逐回合把结果追加写盘（中途被关窗口也能保留已完成的部分）
    def on_end(ep: int, stats: dict) -> None:
        with result_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"episode": ep, **stats},
                                ensure_ascii=False) + "\n")

    sync_dir = Path(args.sync_dir) if args.sync_dir else None

    def before_reset(ep: int) -> None:
        wait_for_turn(sync_dir, ep)

    try:
        summary = run_policy(
            game, policy, episodes=args.episodes,
            seed_base=args.seed_base,
            on_episode_end=on_end,
            before_reset=before_reset,
        )
    except KeyboardInterrupt:
        summary = {"n": 0}

    with result_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"done": True, "label": label,
                             "n": summary.get("n", 0),
                             "score_mean": summary.get("score_mean", 0.0),
                             "return_mean": summary.get("return_mean", 0.0)},
                            ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
