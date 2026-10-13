"""
compare.py —— 三窗口横向对比
=============================

指定游戏后同时打开**三个窗口**，从左到右：

    +------------+  +------------+  +------------+
    |   随机     |  |  加载权重  |  |   专家     |
    +------------+  +------------+  +------------+
      左 (x=0)       中 (x=420)     右 (x=840)

三个窗口跑**同一个种子序列**，并且用屏障让每一回合同时开始 ——
所以画面上的差异**只来自策略本身**，不会因为「谁运气好」而失真。

实现要点
--------
1. **子进程隔离**：pygame 的 display 是全局单例，一个进程只可能有一个窗口，
   而且 pygame.quit() 是全局的。所以三窗口只能靠三个进程
   （详见 compare_child.py 的说明，它是本脚本的子进程入口）。
2. **窗口排布**：用环境变量 SDL_VIDEO_WINDOW_POS 给每个子进程指定位置。
3. **回合屏障**：子进程通过标记文件对齐回合起点，避免「左窗口第 8 回合、
   右窗口第 2 回合」这种没法对照的情况。
4. **结果汇总**：每个子进程把逐回合结果写成 jsonl，父进程读出来做对比表。
   用文件而不是 stdout，是为了让「中途关掉某个窗口」也能拿到已完成的数据。

没有专家策略的游戏（如 dodger）会在第三个位置显示明确的占位提示，
而不是静默变成两个窗口 —— 那样看起来像工具坏了。

用法
----
    python compare.py --game catcher
    python compare.py --game catcher --episodes 20
    python compare.py --game dodger --episodes 10
    python compare.py --game catcher --no-expert-window   # 只开两个窗口
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cli, paths
from common.paths import PathResolutionError


# ----------------------------------------------------------------------
# 窗口几何
# ----------------------------------------------------------------------
WINDOW_W, WINDOW_H = 400, 600
GAP = 20


def window_positions(n: int) -> list[tuple[int, int]]:
    """从左到右的窗口坐标。"""
    return [(i * (WINDOW_W + GAP), 100) for i in range(n)]


# ----------------------------------------------------------------------
def read_results(path: Path) -> tuple[list[dict], dict | None]:
    """读一个子进程的结果文件，返回 (逐回合记录, 汇总记录)。"""
    episodes, summary = [], None
    if not path.is_file():
        return episodes, summary

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("done"):
            summary = rec
        elif rec.get("skipped"):
            return [], rec
        elif "episode" in rec:
            episodes.append(rec)
    return episodes, summary


def print_table(rows: list[tuple[str, list[dict], dict | None]],
                seeds: range) -> None:
    """打印三个策略的对比表。"""
    print("\n" + "=" * 74)
    print(f"{'策略':<20} {'平均 score':>12} {'平均 return':>14} {'平均步数':>12}")
    print("-" * 74)

    for label, episodes, summary in rows:
        if summary and summary.get("skipped"):
            print(f"{label:<20} {'—':>12} {'该游戏暂无专家策略':>14}")
            continue
        if not episodes:
            print(f"{label:<20} {'（无数据）':>12}")
            continue
        n = len(episodes)
        s = sum(e["score"] for e in episodes) / n
        r = sum(e["return"] for e in episodes) / n
        st = sum(e["steps"] for e in episodes) / n
        print(f"{label:<20} {s:>12.2f} {r:>14.2f} {st:>12.1f}")

    print("=" * 74)

    # 逐回合明细：因为种字同步，同一回合的横向比较才有意义
    detailed = [(lb, eps) for lb, eps, _ in rows if eps]
    if detailed and all(len(eps) == len(detailed[0][1]) for _, eps in detailed):
        print("\n逐回合 score（同种子，可横向对比）：")
        header = f"{'回合':>4} {'seed':>6} " + " ".join(
            f"{lb[:12]:>13}" for lb, _ in detailed)
        print(header)
        print("-" * len(header))
        for k in range(len(detailed[0][1])):
            seed = list(seeds)[k] if k < len(list(seeds)) else "-"
            cells = " ".join(f"{eps[k]['score']:>13.2f}" for _, eps in detailed)
            print(f"{k + 1:>4} {seed:>6} {cells}")


# ----------------------------------------------------------------------
def main(args) -> int:
    # ---- 解析 run（三个窗口共用同一份游戏参数）----
    try:
        run_id = paths.resolve_run(args.game, args.run)
    except PathResolutionError as exc:
        print(f"{exc}\n\n提示：随机/专家窗口不需要权重，"
              f"但为了三个窗口参数一致仍需要一次训练的 config。", file=sys.stderr)
        return 2

    cfg_path = paths.model_run_dir(args.game, run_id) / "config.json"
    game_kwargs = {}
    if cfg_path.is_file():
        game_kwargs = json.loads(
            cfg_path.read_text(encoding="utf-8")).get("game_kwargs", {})

    from experts import has_expert
    expert_ok = has_expert(args.game)

    print(f"游戏      : {args.game}")
    print(f"run       : {run_id}")
    print(f"游戏参数  : {game_kwargs}")
    print(f"回合数    : {args.episodes}   种子: {args.seed_base} 起"
          f"（三窗口同步）")

    # ---- 组装三个子进程 ----
    specs = [
        ("随机策略", "random"),
        ("加载权重", "model"),
    ]
    if expert_ok:
        specs.append(("专家策略", "expert"))
    elif args.no_expert_window:
        print("专家策略  : 该游戏暂无专家策略（已按 --no-expert-window 跳过）")
    else:
        print(f"专家策略  : 该游戏暂无专家策略，第三个窗口将显示占位提示")
        specs.append(("专家策略", "expert"))

    n_windows = len(specs)
    positions = window_positions(n_windows)

    tmp = Path(tempfile.mkdtemp(prefix="compare_"))
    sync_dir = tmp / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)

    child = Path(__file__).resolve().parent / "compare_child.py"
    procs: list[tuple[str, subprocess.Popen, Path]] = []

    for (label, kind), (x, y) in zip(specs, positions):
        result_file = tmp / f"{kind}.jsonl"
        env = dict(os.environ)
        env["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        env["COMPARE_PEERS"] = str(n_windows)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent)

        cmd = [
            sys.executable, str(child),
            "--game", args.game,
            "--policy", kind,
            "--episodes", str(args.episodes),
            "--seed-base", str(args.seed_base),
            "--result", str(result_file),
            "--sync-dir", str(sync_dir),
            "--label", label,
        ]
        if kind == "model":
            cmd += ["--run", run_id, "--weights", args.weights]

        print(f"  窗口 {x:>4},{y:<4} -> {label}")
        procs.append((label, subprocess.Popen(cmd, env=env), result_file))

    print(f"\n已启动 {n_windows} 个窗口。关闭所有窗口即可结束，"
          f"或按 Ctrl+C 中断。\n")

    # ---- 等子进程结束 ----
    try:
        for label, proc, _ in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n中断，正在关闭窗口 ...")
        for _, proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()

    # ---- 汇总 ----
    rows = []
    for label, _, result_file in procs:
        episodes, summary = read_results(result_file)
        rows.append((label, episodes, summary))
    print_table(rows, range(args.seed_base, args.seed_base + args.episodes))

    # 清理临时目录（结果已经读进内存）
    try:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass
    return 0


# ======================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="三窗口横向对比：随机 / 加载权重 / 专家")
    cli.add_game_arg(p, default="catcher")
    cli.add_run_arg(p)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=7000,
                   help="三个窗口共用的起始种子（同步对比的关键）")
    p.add_argument("--weights", type=str, default="best",
                   choices=["best", "final"])
    p.add_argument("--no-expert-window", action="store_true",
                   help="游戏没有专家策略时只开两个窗口")
    args = p.parse_args()

    raise SystemExit(main(args))
