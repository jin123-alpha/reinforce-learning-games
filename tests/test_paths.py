"""
tests/test_paths.py
===================
`common/paths.py` 的单元测试。

为什么这个模块值得单独测：重构前 `Path("models") / game` 这类字面量
散落在 5 处，写错了**不会报错**，只会静默读错目录 —— 这类 bug
到了「三个窗口比出来的数不对劲」的时候才发现，排查成本极高。
所以路径规则必须有测试兜住。

运行：
    python -m pytest tests/ -v
    python tests/test_paths.py          # 不装 pytest 也能跑
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import paths  # noqa: E402


# ----------------------------------------------------------------------
# 简单断言框架：不依赖 pytest 也能跑
# ----------------------------------------------------------------------
class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")

    def raises(self, name: str, fn, exc=Exception) -> None:
        try:
            fn()
        except exc:
            self.passed += 1
        except Exception as other:                       # 类型不对也算失败
            self.failed.append(f"{name} — 抛了 {type(other).__name__} 而不是 {exc.__name__}")
        else:
            self.failed.append(f"{name} — 本应抛异常但没有")


# ======================================================================
# 1. 路径拼接规则
# ======================================================================
def test_layout(c: _Case) -> None:
    c.check("config 目录按游戏分区",
            paths.config_dir("catcher").name == "catcher"
            and paths.config_dir("catcher").parent.name == "config")

    c.check("models 目录按游戏分区",
            paths.models_dir("dodger").parent.name == "models")

    c.check("logs 目录按游戏分区",
            paths.logs_dir("catcher").parent.name == "logs")

    c.check("run 目录嵌在游戏目录下",
            paths.model_run_dir("catcher", "20250101-120000_dqn")
            == paths.models_dir("catcher") / "20250101-120000_dqn")

    c.check("expert 文件路径按游戏命名",
            paths.expert_path("catcher").name == "catcher.py")


# ======================================================================
# 2. 非法名字必须被拦住（路径穿越）
# ======================================================================
def test_reject_unsafe(c: _Case) -> None:
    for bad in ["../etc", "a/b", "..", "", "cat cher", ".hidden", "a\\b"]:
        c.raises(f"拒绝非法游戏名 {bad!r}", lambda b=bad: paths.models_dir(b),
                 paths.PathResolutionError)

    for bad in ["../x", "a/b", ".."]:
        c.raises(f"拒绝非法 run 名 {bad!r}",
                 lambda b=bad: paths.model_run_dir("catcher", b),
                 paths.PathResolutionError)


# ======================================================================
# 3. config 解析：相对游戏目录，而不是 CWD
# ======================================================================
def test_resolve_config(c: _Case, root: Path) -> None:
    d = root / "config" / "catcher"
    d.mkdir(parents=True, exist_ok=True)
    (d / "game.json").write_text('{"game_kwargs": {}}', encoding="utf-8")
    (d / "1.json").write_text('{"algo": "dqn"}', encoding="utf-8")

    saved = paths.CONFIG_ROOT
    paths.CONFIG_ROOT = root / "config"
    try:
        c.check("默认解析到 game.json",
                paths.resolve_config("catcher").name == "game.json")

        c.check("--config 1 解析到 1.json",
                paths.resolve_config("catcher", "1").name == "1.json")

        c.check("带 .json 后缀也能解析",
                paths.resolve_config("catcher", "1.json").name == "1.json")

        c.check("解析结果是绝对路径，与 CWD 无关",
                paths.resolve_config("catcher", "1").is_absolute())

        c.raises("不存在的 config 要报错",
                 lambda: paths.resolve_config("catcher", "nope"),
                 paths.PathResolutionError)

        # 关键回归：绝不能解析成 CWD 下的同名文件
        c.check("不会退化成 CWD 相对路径",
                paths.resolve_config("catcher", "1")
                != Path("1.json").resolve())
    finally:
        paths.CONFIG_ROOT = saved


# ======================================================================
# 4. run 解析 + latest.txt 指针
# ======================================================================
def test_resolve_run(c: _Case, root: Path) -> None:
    g = root / "models" / "catcher"
    run_a = g / "20250101-120000_dqn"
    (run_a / "best").mkdir(parents=True, exist_ok=True)
    (run_a / "best" / "best_model.zip").write_bytes(b"PK")
    (run_a / "final.zip").write_bytes(b"PK")

    saved = paths.MODELS_ROOT
    paths.MODELS_ROOT = root / "models"
    try:
        c.check("显式 run 名可用",
                paths.resolve_run("catcher", "20250101-120000_dqn")
                == "20250101-120000_dqn")

        c.raises("不存在的 run 要报错并列出可选项",
                 lambda: paths.resolve_run("catcher", "nope"),
                 paths.PathResolutionError)

        # 没有 latest.txt 时，报错信息要提到「还没训练过」
        try:
            paths.resolve_run("catcher")
            c.failed.append("缺少 latest.txt 本应报错")
        except paths.PathResolutionError as exc:
            c.check("缺 latest.txt 的提示提到可用 run",
                    "20250101-120000_dqn" in str(exc),
                    str(exc))

        # 写指针后应指向该 run
        paths.write_latest_pointer("catcher", "20250101-120000_dqn")
        c.check("latest.txt 生效",
                paths.resolve_run("catcher") == "20250101-120000_dqn")

        # 指针指向已删除的 run -> 必须报错，而不是静默失败
        (run_a / "best" / "best_model.zip").unlink()
        (run_a / "final.zip").unlink()
        import shutil
        shutil.rmtree(run_a)

        try:
            paths.resolve_run("catcher")
            c.failed.append("悬空 latest.txt 本应报错")
        except paths.PathResolutionError as exc:
            c.check("悬空指针的提示包含指向的名字",
                    "20250101-120000_dqn" in str(exc), str(exc))
    finally:
        paths.MODELS_ROOT = saved


# ======================================================================
# 5. 权重解析：best / final / 回退
# ======================================================================
def test_resolve_weights(c: _Case, root: Path) -> None:
    run_id = "20250101-120000_dqn"
    g = root / "models" / "catcher" / run_id
    (g / "best").mkdir(parents=True, exist_ok=True)

    saved = paths.MODELS_ROOT
    paths.MODELS_ROOT = root / "models"
    try:
        # 两个都有 -> 各自命中，且不发生回退
        (g / "best" / "best_model.zip").write_bytes(b"PK")
        (g / "final.zip").write_bytes(b"PK")

        c.check("best 命中 best_model",
                paths.resolve_weights("catcher", run_id, "best").name == "best_model")
        c.check("final 命中 final",
                paths.resolve_weights("catcher", run_id, "final").name == "final")

        # 只有 final -> 请求 best 时应回退，而不是报错
        (g / "best" / "best_model.zip").unlink()
        c.check("best 缺失时回退到 final",
                paths.resolve_weights("catcher", run_id, "best").name == "final")

        # 两个都没有 -> 报错
        (g / "final.zip").unlink()
        c.raises("两个权重都没有要报错",
                 lambda: paths.resolve_weights("catcher", run_id, "best"),
                 paths.PathResolutionError)

        c.raises("未知权重类型要报错",
                 lambda: paths.resolve_weights("catcher", run_id, "middle"),
                 paths.PathResolutionError)
    finally:
        paths.MODELS_ROOT = saved


# ======================================================================
# 6. run 名生成
# ======================================================================
def test_new_run_id(c: _Case) -> None:
    import datetime as dt

    rid = paths.new_run_id("dqn", dt.datetime(2025, 1, 2, 14, 30, 22))
    c.check("run 名含时间与算法", rid == "20250102-143022_dqn", rid)

    c.check("run 名按字典序 == 按时间序",
            paths.new_run_id("dqn", dt.datetime(2025, 1, 1))
            < paths.new_run_id("dqn", dt.datetime(2025, 1, 2)))

    c.raises("非法算法名被拒绝",
             lambda: paths.new_run_id("../evil"), paths.PathResolutionError)


# ======================================================================
def main() -> int:
    c = _Case()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        test_layout(c)
        test_reject_unsafe(c)
        test_resolve_config(c, root / "cfgcase")
        test_resolve_run(c, root / "runcase")
        test_resolve_weights(c, root / "wcase")
        test_new_run_id(c)

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
