"""
tools/migrate_models.py
=======================
把**旧目录结构**的 models/ / logs/ 迁移到新结构（一次性脚本）。

旧结构（重构前）
----------------
    models/<game>/config.json          游戏参数（play.py 读它）
    models/<game>/best/best_model.zip  评估最优
    models/<game>/final.zip            训练结束
    models/<game>/tb/                  TensorBoard
    models/<game>/evaluations.npz      评估曲线
    models/<game>/best_v1_model.zip    手动 _v1 残留

新结构
------
    models/<game>/<run_id>/config.json         完整参数快照
    models/<game>/<run_id>/best/best_model.zip
    models/<game>/<run_id>/final.zip
    models/<game>/latest.txt                   -> run_id
    logs/<game>/<run_id>/tb/
    logs/<game>/<run_id>/evaluations.npz

run_id 从权重文件的修改时间生成，保证多个旧 run 之间的先后顺序正确。

用法
----
    python tools/migrate_models.py --dry-run     # 先看会做什么
    python tools/migrate_models.py               # 真正执行
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import paths  # noqa: E402


# ----------------------------------------------------------------------
def guess_algo(zip_path: Path, cfg: dict | None = None) -> str:
    """猜测旧权重用的算法。优先看 config，再看 TB 目录名，最后默认 dqn。"""
    if cfg and cfg.get("algo"):
        return str(cfg["algo"]).lower()

    # TB 目录名形如 DQN_1 / PPO_2
    for tb in zip_path.parent.parent.glob("tb/*"):
        name = tb.name.split("_")[0].lower()
        if name in ("dqn", "ppo"):
            return name
    return "dqn"


def mtime_run_id(path: Path, algo: str) -> str:
    """用文件修改时间生成 run_id，保证旧 run 之间的时间顺序。"""
    when = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return f"{when:%Y%m%d-%H%M%S}_{algo}"


# ----------------------------------------------------------------------
def collect_legacy_runs(game_dir: Path) -> list[dict]:
    """找出旧结构里所有「看起来是一次训练」的权重文件。

    判定方式：直接躺在 models/<game>/ 或 models/<game>/best/ 下的 .zip。
    已经在新结构里（<run_id>/*.zip）的会被跳过。
    """
    if not game_dir.is_dir():
        return []

    runs: list[dict] = []
    cfg = None
    cfg_path = game_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = None

    # models/<game>/best/best_model.zip  -> best
    best_zip = game_dir / "best" / "best_model.zip"
    # models/<game>/final.zip            -> final
    final_zip = game_dir / "final.zip"

    if best_zip.is_file() or final_zip.is_file():
        anchor = best_zip if best_zip.is_file() else final_zip
        runs.append({
            "kind": "current",
            "algo": guess_algo(anchor, cfg),
            "run_id": mtime_run_id(anchor, guess_algo(anchor, cfg)),
            "best_zip": best_zip if best_zip.is_file() else None,
            "final_zip": final_zip if final_zip.is_file() else None,
            "tb": game_dir / "tb",
            "evals": game_dir / "evaluations.npz",
            "cfg": cfg,
        })

    return runs


def collect_v1_leftovers(game_dir: Path) -> list[Path]:
    """旧的 *_v1_model.zip 之类的手动版本残留（第 7 条：清理）。"""
    if not game_dir.is_dir():
        return []
    out = []
    for z in game_dir.glob("*.zip"):
        if "_v1" in z.name or z.name.startswith(("best_", "final_")):
            out.append(z)
    return out


# ----------------------------------------------------------------------
def migrate_game(game: str, dry_run: bool) -> None:
    gdir = paths.models_dir(game)
    if not gdir.is_dir():
        print(f"  跳过：{gdir} 不存在")
        return

    runs = collect_legacy_runs(gdir)
    leftovers = collect_v1_leftovers(gdir)

    if not runs:
        print(f"  没有发现旧结构的权重（可能已经是新结构）")
    for r in runs:
        run_id = r["run_id"]
        mdir = paths.model_run_dir(game, run_id)
        ldir = paths.log_run_dir(game, run_id)

        print(f"  run {run_id}")
        print(f"    模型 -> {mdir.relative_to(ROOT)}")
        if r["best_zip"]:
            print(f"      best  <- {r['best_zip'].name}")
        if r["final_zip"]:
            print(f"      final <- {r['final_zip'].name}")
        if r["tb"].is_dir():
            print(f"    TB   -> {ldir / 'tb'}")
        if r["evals"].is_file():
            print(f"    曲线 -> {ldir / 'evaluations.npz'}")

        if dry_run:
            continue

        (mdir / "best").mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)

        if r["best_zip"]:
            shutil.move(str(r["best_zip"]), str(mdir / "best" / "best_model.zip"))
        if r["final_zip"]:
            shutil.move(str(r["final_zip"]), str(mdir / "final.zip"))

        if r["tb"].is_dir():
            for child in r["tb"].iterdir():
                shutil.move(str(child), str(ldir / "tb" / child.name))
            r["tb"].rmdir()
        if r["evals"].is_file():
            shutil.move(str(r["evals"]), str(ldir / "evaluations.npz"))

        # 写完整参数快照，让 play.py 能重建当时的环境
        algo = r["algo"]
        cfg = r["cfg"] or {}
        snapshot = {
            "game": game,
            "run_id": run_id,
            "algo": algo,
            "config_source": "(迁移自旧结构，无原始算法配置)",
            "game_kwargs": cfg.get("game_kwargs", {}),
            "train": {"timesteps": None, "note": "旧结构未记录步数"},
            "algo_kwargs": {},
        }
        (mdir / "config.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

        paths.write_latest_pointer(game, run_id)
        print(f"    latest.txt -> {run_id}")

    for z in leftovers:
        print(f"  清理残留权重: {z.name}")
        if not dry_run:
            z.unlink()

    # 旧的 config.json 已经在 config/<game>/game.json 里了
    old_cfg = gdir / "config.json"
    if old_cfg.is_file() and runs:
        print(f"  移除旧 config.json（参数已移到 config/{game}/game.json）")
        if not dry_run:
            old_cfg.unlink()

    # 清掉空目录
    if not dry_run:
        for d in (gdir / "best", gdir / "tb"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()


# ======================================================================
def main() -> int:
    p = argparse.ArgumentParser(description="把旧 models/ 迁移到 run 目录结构")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不动文件")
    args = p.parse_args()

    print(f"项目根：{ROOT}")
    print(f"模式  ：{'预演（不动文件）' if args.dry_run else '实际执行'}\n")

    games = paths.available_games()
    if not games:
        print("没有发现任何游戏目录。")
        return 1

    for game in games:
        print(f"[{game}]")
        migrate_game(game, args.dry_run)
        print()

    if args.dry_run:
        print("以上是预演结果。去掉 --dry-run 才会真正移动文件。")
    else:
        print("迁移完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
