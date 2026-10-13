"""
human_play.py —— 人类自己玩
===========================

**本文件只依赖 games 包和 common 工具，绝不 import envs / stable_baselines3 /
torch / tensorboard。**

这不是洁癖，是可验证的架构保证：它证明了「游戏完全不知道强化学习的存在」。
如果哪天有人往 games/ 里塞了 `import torch`，下面的命令会立刻失败：

    python tests/test_decoupling.py

重构前这一点其实是**不成立**的：human_play.py 从 train.py 里 import
parse_overrides，而 train.py 顶层 `from stable_baselines3 import DQN, PPO`，
于是「让人玩个游戏」也要先把 torch 拉起来（几百 MB、几秒启动时间）。

现在 parse_overrides 搬到了 common/cli.py（纯标准库），依赖链干净了。

游戏参数从哪来
--------------
默认读 `config/<game>/game.json` —— 也就是训练时用的那份参数。
这样才能保证「你手玩的那一局」和「模型训练/评估的那一局」条件一致，
报告里的对比才有意义。用 --set 可以临时改。

用法
----
    python human_play.py --game catcher
    python human_play.py --game dodger --set max_steps=2000
    python human_play.py --list-games
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 只 import 这两个：games（游戏）和 common（纯标准库工具）----
# 注意 run_human 从 games 拿，而不是 envs.base：
# games/__init__.py 已经把它重新导出了，走这条路可以完全绕开
# envs 包（envs 会 import gymnasium）。这样「人类游玩」这条路径
# 连 gymnasium 都不需要，更不用说 torch。
from common import cli, paths
from games import list_games, make_game, run_human


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="人类自己玩游戏")
    cli.add_game_arg(p, default="catcher")
    cli.add_set_arg(p)
    p.add_argument("--list-games", action="store_true")
    args = p.parse_args()

    if args.list_games:
        print("可用游戏：" + "、".join(list_games()))
        raise SystemExit(0)

    # 读 config/<game>/game.json，再用 --set 覆盖
    game_kwargs = cli.load_game_params(args.game, cli_sets=args.set)
    print(f"游戏      : {args.game}")
    print(f"游戏参数  : {game_kwargs}")

    game = make_game(args.game, render_mode="human", **game_kwargs)
    run_human(game)
