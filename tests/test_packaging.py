"""
tests/test_packaging.py
=======================
**打包与分组安装测试**：pyproject.toml 的 extras 定义是否正确。

背景
----
依赖按「能跑什么」分成三组：

    [1]  最基本的人类游玩          numpy + gymnasium + pygame-ce
    [2]  agent 游玩（已有权重）    + stable-baselines3（连带 torch）
    [3]  训练                      + tensorboard

踩过的坑（本测试就是为此而写）
------------------------------
1. **不能把全部依赖放进 `dependencies`**。
   pip 装 `.[N]` 时，装的是 `dependencies ∪ extra N`。
   如果 dependencies 里已经有 torch，那么 `pip install -e .[1]`
   也会装 torch，extras 就完全失去意义了。
   → 所以 dependencies 只放最小的组 1，其余靠 extras 追加。

2. **extras 不支持嵌套引用**。
   不能写 `"3": [".[2]"]`，必须把 stable-baselines3 重复列在组 3 里。
   → 所以本测试检查「递进包含」关系是否真的成立。

3. **extras 名可以是数字，也可以有语义别名**。
   `.[1]` 和 `.[play]` 应该等价。

本测试读 pyproject.toml 做静态校验，并真实调用 pip 的 dry-run
确认解析结果 —— 后者能抓到「元数据写对了但 pip 解析不出来」的情况。

运行：
    python tests/test_packaging.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Case:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}{(' — ' + detail) if detail else ''}")


# ----------------------------------------------------------------------
def load_pyproject() -> dict:
    """读 pyproject.toml（用标准库 tomllib，Python 3.11+）。"""
    import tomllib
    path = ROOT / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def dep_names(deps: list[str]) -> set[str]:
    """从依赖字符串里抽出包名（去掉版本约束和 extras）。

    注意要归一化：PyPI 上 `stable-baselines3` 的元数据名是
    `stable_baselines3`（下划线），所以比较时统一成小写 + 连字符。
    """
    out = set()
    for d in deps:
        name = d.split(";")[0]              # 去掉环境标记
        name = name.split("[")[0]           # 去掉 pkg[extra]
        for sep in (">=", "==", "<=", "~=", "!=", ">", "<", "="):
            if sep in name:
                name = name.split(sep)[0]
                break
        name = name.strip().lower().replace("_", "-")
        if name:
            out.add(name)
    return out


# ======================================================================
def test_last_configured_extra() -> str | None:
    """挑一个本轮要浅装验证的 extra。默认组 1（最轻）。"""
    return os.environ.get("TEST_EXTRA")


# ======================================================================
def test_structure(c: _Case) -> None:
    """静态检查：extras 定义齐全、递进关系成立。"""
    data = load_pyproject()
    proj = data.get("project", {})
    extras = proj.get("optional-dependencies", {})

    c.check("pyproject.toml 有 project 段", bool(proj))
    c.check("有 optional-dependencies 段", bool(extras))

    # 三组数字 extras 都要存在
    for key in ("1", "2", "3"):
        c.check(f"定义了 extra [{key}]", key in extras,
                f"实际有 {sorted(extras)}")

    if not all(k in extras for k in ("1", "2", "3")):
        return

    n1 = dep_names(extras["1"])
    n2 = dep_names(extras["2"])
    n3 = dep_names(extras["3"])

    # ---- 各组的核心内容 ----
    base = {"numpy", "gymnasium", "pygame-ce"}
    c.check("组 1 = 游戏/环境最小集", n1 == base, f"实际 {sorted(n1)}")

    c.check("组 2 含 stable-baselines3", "stable-baselines3" in n2,
            f"实际 {sorted(n2)}")
    c.check("组 2 不含 tensorboard（那是训练的事）",
            "tensorboard" not in n2, f"实际 {sorted(n2)}")

    c.check("组 3 含 stable-baselines3", "stable-baselines3" in n3)
    c.check("组 3 含 tensorboard", "tensorboard" in n3)

    # ---- 递进包含（因为 extras 不能嵌套，必须手工保证）----
    c.check("组 1 ⊆ 组 2", n1 <= n2, f"缺 {sorted(n1 - n2)}")
    c.check("组 2 ⊆ 组 3", n2 <= n3, f"缺 {sorted(n2 - n3)}")

    # ---- 组 1 绝不能含重依赖（这是分组的全部意义）----
    for heavy in ("torch", "stable-baselines3", "tensorboard"):
        c.check(f"组 1 不含 {heavy}", heavy not in n1, f"实际 {sorted(n1)}")


def test_default_deps_are_minimal(c: _Case) -> None:
    """关键：dependencies 必须是最小集，否则 extras 失效。

    这是实测踩到的坑 —— pip 装 `.[N]` 是 `dependencies ∪ extra N`，
    如果 dependencies 里放了全部，那 `.[1]` 也会装 torch。
    """
    proj = load_pyproject()["project"]
    default = dep_names(proj.get("dependencies", []))
    group1 = dep_names(proj.get("optional-dependencies", {}).get("1", []))

    c.check("dependencies 等于组 1（最小集）", default == group1,
            f"dependencies={sorted(default)} 组1={sorted(group1)}")

    for heavy in ("torch", "stable-baselines3", "tensorboard"):
        c.check(f"dependencies 不含 {heavy}", heavy not in default,
                f"实际 {sorted(default)}")


def test_aliases(c: _Case) -> None:
    """语义化别名应与数字 extras 等价。"""
    extras = load_pyproject()["project"].get("optional-dependencies", {})

    pairs = [("1", "play"), ("2", "agent"), ("3", "train")]
    for num, alias in pairs:
        if num not in extras or alias not in extras:
            c.check(f"别名 [{alias}] 存在", alias in extras)
            continue
        a = dep_names(extras[num])
        b = dep_names(extras[alias])
        c.check(f"[{num}] 与 [{alias}] 等价", a == b,
                f"{sorted(a)} vs {sorted(b)}")

    if "all" in extras:
        allset = dep_names(extras["all"])
        g3 = dep_names(extras.get("3", []))
        c.check("[all] 覆盖组 3", g3 <= allset)


def test_packaging_config(c: _Case) -> None:
    """打包范围：包含游戏代码，排除测试/数据目录。"""
    data = load_pyproject()
    tool = data.get("tool", {}).get("setuptools", {})

    c.check("配置了 py-modules（四个入口脚本）", "py-modules" in tool)
    if "py-modules" in tool:
        mods = set(tool["py-modules"])
        for m in ("train", "play", "human_play", "compare"):
            c.check(f"py-modules 含 {m}", m in mods, f"实际 {sorted(mods)}")

    find = tool.get("packages", {}).get("find", {})
    c.check("配置了 packages.find", bool(find))
    if find:
        include = find.get("include", [])
        exclude = find.get("exclude", [])
        for pkg in ("games*", "envs*", "experts*", "common*"):
            c.check(f"include 含 {pkg}", pkg in include)
        for ex in ("tests*", "tools*"):
            c.check(f"exclude 含 {ex}", ex in exclude)


# ======================================================================
def test_pip_resolves_extras(c: _Case) -> None:
    """真实调用 pip 的 dry-run，确认 extras 能被解析出预期内容。

    只做解析（--dry-run --report），不实际下载安装。
    这一步能抓到「元数据写对了但 pip 解析不出」的问题。
    """
    if os.environ.get("SKIP_PIP_DRYRUN"):
        c.check("（跳过 pip dry-run）", True)
        return

    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        made = subprocess.run([sys.executable, "-m", "venv", str(venv)],
                              capture_output=True, text=True)
        if made.returncode != 0:
            c.check("（无法创建 venv，跳过 pip dry-run）", True)
            return

        vpy = venv / "bin" / "python"
        if not vpy.exists():
            vpy = venv / "Scripts" / "python.exe"
        if not vpy.exists():
            c.check("（找不到 venv 的 python）", False)
            return

        expect = {
            "1": {"numpy", "gymnasium", "pygame-ce"},
            "2": {"numpy", "gymnasium", "pygame-ce", "stable-baselines3"},
            "3": {"numpy", "gymnasium", "pygame-ce",
                  "stable-baselines3", "tensorboard"},
        }
        forbid = {
            "1": {"stable-baselines3", "tensorboard", "torch"},
            "2": {"tensorboard"},
            "3": set(),
        }

        for extra, wanted in expect.items():
            report = Path(tmp) / f"r{extra}.json"
            proc = subprocess.run(
                [str(vpy), "-m", "pip", "install", "--dry-run", "--quiet",
                 "--report", str(report), "-e", f".[{extra}]"],
                capture_output=True, text=True, cwd=str(ROOT), timeout=600,
            )
            if proc.returncode != 0 or not report.exists():
                c.check(f"[{extra}] pip 能解析", False,
                        (proc.stderr.strip().splitlines() or ["?"])[-1])
                continue

            data = json.loads(report.read_text(encoding="utf-8"))
            names = {i["metadata"]["name"].lower().replace("_", "-")
                     for i in data.get("install", [])}

            missing = wanted - names
            c.check(f"[{extra}] 解析出预期依赖", not missing,
                    f"缺 {sorted(missing)}")

            bad = forbid[extra] & names
            c.check(f"[{extra}] 没有多余的重量级依赖", not bad,
                    f"多出 {sorted(bad)}")


# ======================================================================
def test_requirements_txt_documented(c: _Case) -> None:
    """requirements.txt 应存在，并说明 extras 的正确用法。"""
    req = ROOT / "requirements.txt"
    c.check("requirements.txt 存在", req.is_file())
    if not req.is_file():
        return

    text = req.read_text(encoding="utf-8")
    c.check("说明了 pip install -e .[N] 用法",
            "pip install -e .[1]" in text or "pip install -e .[" in text)
    c.check("解释了为什么不能用 -r file[1]",
            "extras" in text and ("包名" in text or "不能用在" in text))
    c.check("给出了各组的内容说明",
            all(k in text for k in ("组 1", "组 2", "组 3")))


# ======================================================================
def main() -> int:
    c = _Case()
    print("[1/7] extras 结构与递进关系 ...")
    test_structure(c)
    print("[2/7] 默认依赖必须是最小集 ...")
    test_default_deps_are_minimal(c)
    print("[3/7] 语义化别名 ...")
    test_aliases(c)
    print("[4/7] 打包范围配置 ...")
    test_packaging_config(c)
    print("[5/7] requirements.txt 文档 ...")
    test_requirements_txt_documented(c)
    print("[6/7] pip 真实解析 extras（dry-run）...")
    test_pip_resolves_extras(c)
    print("[7/7] 完成")

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