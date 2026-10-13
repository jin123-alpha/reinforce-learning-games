"""
train.py —— 训练「任意游戏」
============================

训练代码与具体游戏无关：它只依赖 envs 的 BaseGame 契约
（action_space / observation_space / step / reset）。

目录约定（全部自动按游戏分目录，见 common/paths.py）
----------------------------------------------------
    config/<game>/<n>.json     算法超参（本脚本读它）
    config/<game>/game.json    游戏参数（被上一个文件引用）
    models/<game>/<run_id>/    本次训练的权重
    logs/<game>/<run_id>/      本次训练的日志与 TensorBoard
    models/<game>/latest.txt   指向最近一次 run

每次训练都新建一个 run 目录，**不会覆盖历史**。
这让「对比不同步数的 checkpoint」成为可能 ——
重构前只有一个 best/final，第二次训练就把第一次盖掉了。

用法
----
    python train.py --game catcher                    # 用 config/catcher/1.json
    python train.py --game catcher --config 2.json    # 换一份算法配置
    python train.py --game dodger --timesteps 400000  # 命令行覆盖步数
    python train.py --game catcher --set num_balls=5  # 覆盖游戏参数
    python train.py --game catcher --run-name my_test # 指定 run 名
    python train.py --list-games
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---- 让脚本能从项目根或任意位置运行 ----
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cli, paths
from common.paths import PathResolutionError
from envs import make_env


ALGOS = {}


def _load_algos():
    """延迟导入 SB3（只有真正训练时才需要 torch）。"""
    if not ALGOS:
        from stable_baselines3 import DQN, PPO
        ALGOS.update({"dqn": DQN, "ppo": PPO})
    return ALGOS


# ======================================================================
# 超参展开
# ======================================================================
def build_model(algo: str, env, cfg: dict, args, log_dir: Path):
    """按 config 里的 algo_kwargs 构造模型。

    只有 `net_arch` 要特殊处理：SB3 把它塞在 policy_kwargs 里，
    而其它参数都是构造函数的直接入参。
    """
    algos = _load_algos()
    kwargs = dict(cfg.get("algo_kwargs", {}))

    net_arch = kwargs.pop("net_arch", [256, 256])
    kwargs.setdefault("verbose", 1)
    kwargs["tensorboard_log"] = str(log_dir / "tb")
    kwargs["seed"] = args.seed
    kwargs["device"] = args.device
    kwargs["policy_kwargs"] = {"net_arch": list(net_arch)}

    return algos[algo]("MlpPolicy", env, **kwargs)


# ======================================================================
def write_run_config(run_dir: Path, payload: dict) -> Path:
    """把本次训练**完整解算后**的参数写进 run 目录。

    这份快照是让 checkpoint「自描述」的关键：
    play.py / compare.py 只读它就能重建出一模一样的环境，
    不需要回头去猜当时用了什么参数。
    """
    path = run_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


# ======================================================================
def main(args) -> int:
    # ---- 1. 读配置（算法超参 + 游戏参数）----
    config_name = args.config or "1"
    try:
        cfg, game_kwargs = cli.load_algo_config(args.game, config_name, args.set)
    except PathResolutionError as exc:
        print(f"配置解析失败：\n{exc}", file=sys.stderr)
        return 2

    algo = (args.algo or cfg.get("algo") or "dqn").lower()
    if algo not in ("dqn", "ppo"):
        print(f"未知算法 {algo!r}，可选：dqn、ppo", file=sys.stderr)
        return 2

    train_cfg = dict(cfg.get("train", {}))
    # 命令行优先于配置文件
    timesteps = args.timesteps if args.timesteps is not None \
        else int(train_cfg.get("timesteps", 400_000))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 0))
    eval_freq = args.eval_freq if args.eval_freq is not None \
        else int(train_cfg.get("eval_freq", 20_000))
    n_eval = args.n_eval if args.n_eval is not None \
        else int(train_cfg.get("n_eval", 5))
    args.seed = seed

    # ---- 2. 建 run 目录 ----
    run_id = args.run_name or paths.new_run_id(algo)
    try:
        model_dir, log_dir = paths.ensure_run_dirs(args.game, run_id)
    except PathResolutionError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    # ---- 3. 日志：写进 logs/<game>/<run>/train.log ----
    log_file = log_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("train")

    log.info("=" * 62)
    log.info(f"游戏        : {args.game}")
    log.info(f"游戏参数    : {game_kwargs}")
    log.info(f"算法        : {algo}")
    log.info(f"配置        : config/{args.game}/{config_name}.json")
    log.info(f"run         : {run_id}")
    log.info(f"步数 / 种子 : {timesteps} / {seed}")
    log.info(f"模型目录    : {paths.model_run_dir(args.game, run_id)}")
    log.info(f"日志目录    : {log_dir}")
    log.info("=" * 62)

    # ---- 4. 环境 ----
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    def env_factory():
        return make_env(args.game, **game_kwargs)

    train_env = make_vec_env(env_factory, n_envs=1)
    eval_env = make_vec_env(env_factory, n_envs=1)
    log.info(f"动作空间    : {train_env.action_space}")
    log.info(f"观测空间    : {train_env.observation_space.shape}")

    # 参数快照：play.py / compare.py 靠它重建同样的环境
    write_run_config(model_dir, {
        "game": args.game,
        "run_id": run_id,
        "algo": algo,
        "config_source": f"config/{args.game}/{config_name}.json",
        "game_kwargs": game_kwargs,
        "train": {"timesteps": timesteps, "seed": seed,
                  "eval_freq": eval_freq, "n_eval": n_eval},
        "algo_kwargs": cfg.get("algo_kwargs", {}),
    })

    # ---- 5. 训练 ----
    model = build_model(algo, train_env, cfg, args, log_dir)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        # evaluations.npz 落在这里；因为每个 run 目录独立，
        # 不会像重构前那样被下一次训练覆盖掉
        log_path=str(log_dir),
        eval_freq=max(eval_freq, 1),
        n_eval_episodes=n_eval,
        deterministic=True,
        render=False,
    )

    log.info("开始训练 ...")
    try:
        model.learn(total_timesteps=timesteps, callback=eval_cb)
    except KeyboardInterrupt:
        log.warning("训练被中断，仍会保存当前权重并更新 latest 指针。")

    # ---- 6. 保存 ----
    final_path = model_dir / "final"
    model.save(str(final_path))
    log.info(f"模型已保存: {final_path}.zip")

    pointer = paths.write_latest_pointer(args.game, run_id)
    log.info(f"latest 指针: {pointer} -> {run_id}")

    # ---- 7. 最终评估 ----
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=n_eval,
                                    deterministic=True)
    log.info(f"最终评估    : mean_reward = {mean_r:.2f} +/- {std_r:.2f}")

    # 追加到 eval_summary.csv，便于跨 run 横向比较（README 的表格就是这么做出来的）
    summary = paths.models_dir(args.game) / "eval_summary.csv"
    new_file = not summary.exists()
    with summary.open("a", encoding="utf-8") as fh:
        if new_file:
            fh.write("run_id,algo,timesteps,seed,mean_reward,std_reward\n")
        fh.write(f"{run_id},{algo},{timesteps},{seed},{mean_r:.4f},{std_r:.4f}\n")
    log.info(f"评估汇总    : {summary}")

    train_env.close()
    eval_env.close()
    return 0


# ======================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="训练任意游戏（游戏通过 --game 选择）")
    cli.add_game_arg(p, default="catcher")
    cli.add_config_arg(p, default="1")
    p.add_argument("--algo", type=str, default=None, choices=["dqn", "ppo"],
                   help="覆盖 config 里的 algo")
    p.add_argument("--timesteps", type=int, default=None, help="覆盖训练步数")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run-name", type=str, default=None,
                   help="指定 run 名（默认为自动生成的时间戳_算法）")
    cli.add_set_arg(p)
    p.add_argument("--list-games", action="store_true")
    args = p.parse_args()

    if args.list_games:
        print("可用游戏：" + "、".join(paths.available_games()))
        raise SystemExit(0)

    raise SystemExit(main(args))
