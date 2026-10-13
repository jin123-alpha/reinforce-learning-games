"""
tests/run_all.py
================
一次跑完所有测试。

    python tests/run_all.py

包含的套件
----------
    test_paths.py       路径解析规则（按游戏分目录、run 指针、权重定位）
    test_cli.py         CLI 解析与配置加载（含对旧写法的兼容性）
    test_experts.py     专家策略（含新旧实现的**等价性**验证）
    test_decoupling.py  架构约束（游戏层不依赖 RL 库）
    test_structure.py   重构后的目录与依赖分层

为什么专家等价性测试特别重要
----------------------------
重构把 ExpertCatcher 从「读 game 对象」改成「纯 obs」。
这是个行为等价的改动，但「等价」不能靠肉眼保证 ——
test_experts.py 内联了一份旧实现，让新旧两版在 4000+ 步真实
轨迹上逐动作比对。有这个测试，重构才敢说没改坏东西。
"""
from __future__ import annotations

import importlib
import io
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUITES = [
    ("编译检查", "tests.test_compile"),
    ("路径解析", "tests.test_paths"),
    ("CLI / 配置", "tests.test_cli"),
    ("专家策略", "tests.test_experts"),
    ("人类游玩", "tests.test_human_play"),
    ("依赖分层", "tests.test_dependencies"),
    ("打包分组", "tests.test_packaging"),
    ("架构解耦", "tests.test_decoupling"),
    ("目录结构", "tests.test_structure"),
]

# 默认跳过的套件：需要联网或较慢。
# test_packaging 会建 venv 并跑 pip dry-run 来验证 extras 解析，
# 默认跳过；用 RUN_SLOW=1 打开，或单独跑该文件。
SLOW_SUITES = {"tests.test_packaging"}


def main() -> int:
    # 无头运行，避免测试时弹窗
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    run_slow = os.environ.get("RUN_SLOW") == "1"
    suites = [s for s in SUITES
              if run_slow or s[1] not in SLOW_SUITES]
    skipped = [s[0] for s in SUITES if s not in suites]

    if skipped:
        print(f"（跳过较慢的套件：{'、'.join(skipped)}；"
              f"用 RUN_SLOW=1 一起跑）")

    results = []
    for title, modname in suites:
        print(f"\n{'=' * 70}")
        print(f"  {title}   ({modname})")
        print("=" * 70)

        try:
            mod = importlib.import_module(modname)
            # 捕获被测套件的输出，只在失败时回放，保持总览清爽
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    rc = mod.main()
            except SystemExit as exc:
                rc = int(exc.code or 0)
            except Exception:
                print(buf.getvalue())
                traceback.print_exc()
                rc = 1
            else:
                if rc != 0:
                    print(buf.getvalue())
        except Exception:
            traceback.print_exc()
            rc = 1

        results.append((title, rc))

    print(f"\n{'=' * 70}")
    print("  总结")
    print("=" * 70)
    failed = 0
    for title, rc in results:
        mark = "通过 ✓" if rc == 0 else "失败 ✗"
        print(f"  {title:<16} {mark}")
        failed += rc != 0

    print()
    if failed:
        print(f"{failed} 个套件失败。")
        return 1
    print(f"全部 {len(results)} 个套件通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
