"""
tests/test_cli.py
=================
`common/cli.py` 的单元测试，重点是 parse_overrides 的**向后兼容性**。

为什么测这个：旧 train.py 的 parse_overrides 用 json.loads，
所以 `--set rich_obs=true` 能work、而 `--set rich_obs=True` 会退化成
字符串 "True"（然后被 game 当成真值，凑巧能用）。
搬进 common/cli.py 后我加了 ast.literal_eval 回退，
必须确认**旧写法一个都没坏**，否则就是静默的行为变更。

运行：
    python tests/test_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import cli  # noqa: E402


class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")

    def eq(self, name: str, got, want) -> None:
        self.check(name, got == want, f"得到 {got!r}，期望 {want!r}")

    def raises(self, name: str, fn) -> None:
        try:
            fn()
        except SystemExit:
            self.passed += 1
        except Exception as other:
            self.failed.append(f"{name} — 抛了 {type(other).__name__}")
        else:
            self.failed.append(f"{name} — 本应 SystemExit")


# ======================================================================
def test_parse_value(c: _Case) -> None:
    # 整数 / 浮点 / 布尔（json 小写）—— 旧代码就支持
    c.eq("int", cli.parse_value("5"), 5)
    c.eq("float", cli.parse_value("5.0"), 5.0)
    c.eq("json 小写 true", cli.parse_value("true"), True)
    c.eq("json 小写 false", cli.parse_value("false"), False)
    c.eq("null", cli.parse_value("null"), None)
    c.eq("负数", cli.parse_value("-3"), -3)
    c.eq("科学计数", cli.parse_value("1e-4"), 1e-4)

    # Python 风格大写 —— 新增的回退能力
    c.eq("Python 风格 True", cli.parse_value("True"), True)
    c.eq("Python 风格 False", cli.parse_value("False"), False)
    c.eq("Python 风格 None", cli.parse_value("None"), None)

    # 容器类型
    c.eq("列表", cli.parse_value("[256, 256]"), [256, 256])
    c.eq("tuple 语法退化为字符串或 tuple", cli.parse_value("(1,2)"), (1, 2))

    # 纯字符串退化
    c.eq("普通字符串", cli.parse_value("relu"), "relu")
    c.eq("带空格的字符串", cli.parse_value("  hello  "), "hello")


# ======================================================================
def test_parse_overrides(c: _Case) -> None:
    # 旧 train.py 的用法，必须逐字兼容
    got = cli.parse_overrides(["num_balls=5", "paddle_speed=4", "rich_obs=false"])
    c.eq("多参数解析", got, {"num_balls": 5, "paddle_speed": 4, "rich_obs": False})

    c.eq("空输入返回空 dict", cli.parse_overrides([]), {})
    c.eq("None 输入返回空 dict", cli.parse_overrides(None), {})

    # 值里含等号：只按第一个 = 分割
    c.eq("值里含等号", cli.parse_overrides(["a=b=c"]), {"a": "b=c"})

    # 浮点 key 场景（旧代码同样处理）
    c.eq("浮点值", cli.parse_overrides(["lr=5e-4"]), {"lr": 5e-4})

    # 格式错误要给出可操作的报错，而不是静默丢参数
    c.raises("缺少 = 要报错", lambda: cli.parse_overrides(["num_balls"]))
    c.raises("空 key 要报错", lambda: cli.parse_overrides(["=5"]))


# ======================================================================
def test_merge(c: _Case) -> None:
    c.eq("后面的层覆盖前面",
         cli.merge_overrides({"a": 1, "b": 2}, {"b": 3}), {"a": 1, "b": 3})
    c.eq("跳过 None 层",
         cli.merge_overrides(None, {"a": 1}, None), {"a": 1})
    c.eq("全 None 得空 dict", cli.merge_overrides(None, None), {})


# ======================================================================
def test_load_algo_config(c: _Case, root: Path) -> None:
    """算法 config 里的 game_params 必须相对**配置文件自己**解析。"""
    from common import paths

    saved = paths.CONFIG_ROOT
    paths.CONFIG_ROOT = root / "config"
    try:
        d = paths.CONFIG_ROOT / "catcher"
        d.mkdir(parents=True, exist_ok=True)
        (d / "game.json").write_text(
            '{"game_kwargs": {"num_balls": 3, "paddle_speed": 5.0}}', encoding="utf-8")
        (d / "1.json").write_text(
            '{"game_params": "game.json", "algo": "dqn",'
            ' "algo_kwargs": {"net_arch": [256, 256]}}', encoding="utf-8")

        cfg, params = cli.load_algo_config("catcher", "1")
        c.eq("读到 algo", cfg["algo"], "dqn")
        c.eq("读到 algo_kwargs", cfg["algo_kwargs"], {"net_arch": [256, 256]})
        c.eq("引用到 game.json 的参数", params, {"num_balls": 3, "paddle_speed": 5.0})

        # --set 要覆盖文件里的值
        _, params2 = cli.load_algo_config("catcher", "1", ["num_balls=7"])
        c.eq("--set 覆盖文件值", params2["num_balls"], 7)
        c.eq("--set 不影响其他参数", params2["paddle_speed"], 5.0)

        # 引用不存在的文件要报错
        (d / "bad.json").write_text('{"game_params": "nope.json"}', encoding="utf-8")
        c.raises("悬空 game_params 引用要报错",
                 lambda: cli.load_algo_config("catcher", "bad"))

        # 不写 game_params 也能读（游戏参数走 CLI 或默认）
        (d / "noref.json").write_text('{"algo": "ppo"}', encoding="utf-8")
        _, params3 = cli.load_algo_config("catcher", "noref", ["max_steps=500"])
        c.eq("无引用时只用 --set", params3, {"max_steps": 500})
    finally:
        paths.CONFIG_ROOT = saved


# ======================================================================
def test_load_game_params(c: _Case, root: Path) -> None:
    from common import paths

    saved = paths.CONFIG_ROOT
    paths.CONFIG_ROOT = root / "gpc"
    try:
        d = paths.CONFIG_ROOT / "catcher"
        d.mkdir(parents=True, exist_ok=True)
        (d / "game.json").write_text(
            '{"game_kwargs": {"num_balls": 3, "max_misses": 10}}', encoding="utf-8")

        params = cli.load_game_params("catcher")
        c.eq("读游戏参数", params, {"num_balls": 3, "max_misses": 10})

        params2 = cli.load_game_params("catcher", cli_sets=["num_balls=5"])
        c.eq("--set 覆盖", params2["num_balls"], 5)

        # 游戏没有 game.json 时不该崩溃，应该用 --set 继续
        params3 = cli.load_game_params("dodger", cli_sets=["max_steps=2000"])
        c.eq("缺 game.json 时退化为 --set", params3, {"max_steps": 2000})
    finally:
        paths.CONFIG_ROOT = saved


# ======================================================================
def main() -> int:
    import tempfile

    c = _Case()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        test_parse_value(c)
        test_parse_overrides(c)
        test_merge(c)
        test_load_algo_config(c, root / "ac")
        test_load_game_params(c, root / "gp")

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
