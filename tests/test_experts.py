"""
tests/test_experts.py
=====================
专家策略的正确性与**等价性**测试。

核心问题：把专家从「读 game 对象」改成「纯 obs」之后，
它的**行为**有没有变？

做法：在这里内联一份**重构前的实现**（OldExpert，逐字照抄
原来的 ExpertCatcher 逻辑），然后让新旧两版跑同一批
预置观测状态，逐个断言动作完全相同。

这样重构就有一个可执行的等价性证据，而不是「看起来没问题」。

运行：
    python tests/test_experts.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs import make_env  # noqa: E402
from experts import ExpertCatcher, RandomPolicy, has_expert, make_expert  # noqa: E402
from experts.base import ObsSpecMismatch  # noqa: E402


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


# ======================================================================
# 重构前的实现（逐字照抄旧 ExpertCatcher 的决策逻辑，用于对照）
# ======================================================================
class OldExpertCatcher:
    """旧版：从 game 对象读常量、调用 game.landing() 之外的部分。"""

    def __init__(self, game) -> None:
        self.n = int(game.num_balls)
        self.rich = bool(game.rich_obs)
        self.width = float(game.WIDTH)
        self.height = float(game.HEIGHT)
        self.paddle_w = float(game.PADDLE_W)
        self.paddle_y = float(game.PADDLE_Y)
        self.ball_r = float(game.BALL_R)
        self.speed = float(game.paddle_speed)
        self.t_max = float(game.T_MAX)
        self.span = self.width - 2.0 * self.ball_r
        self.half = self.paddle_w / 2.0 + self.ball_r
        self.deadzone = self.speed / 2.0
        self.center_x = self.width / 2.0
        self.vx_scale = max(abs(game.BALL_VX_RANGE[0]), abs(game.BALL_VX_RANGE[1]))
        self.vy_scale = max(abs(game.BALL_VY_RANGE[0]), abs(game.BALL_VY_RANGE[1]))

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float64).ravel()
        paddle_x = obs[0] * (self.width - self.paddle_w)
        center = paddle_x + self.paddle_w / 2.0

        blk = obs[2:].reshape(self.n, 7)
        t = blk[:, 4] * self.t_max
        x_land = blk[:, 5] * self.width

        k = np.maximum(np.ceil(t - 1e-9), 1.0)
        can = (np.abs(x_land - center) <= k * self.speed + self.half) & (t > 1e-9)
        if can.any():
            idx = np.where(can)[0]
            order = np.lexsort((np.abs(x_land[idx] - center), k[idx]))
            target = float(x_land[idx[order[0]]])
        else:
            target = self.center_x

        d = target - center
        if d > self.deadzone:
            return 2
        if d < -self.deadzone:
            return 0
        return 1


# ======================================================================
def test_equivalence(c: _Case) -> None:
    """新旧专家在真实游戏轨迹上必须给出完全相同的动作。"""
    env = make_env("catcher", num_balls=3, paddle_speed=5.0)
    game = env.unwrapped_game

    old = OldExpertCatcher(game)
    new = ExpertCatcher(env.action_space, game.obs_spec())

    # 用固定种子跑若干回合，覆盖各种球位/球速组合
    total, mismatch = 0, 0
    first_bad = None
    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        for _ in range(300):
            a_old = old.act(obs)
            a_new = int(new.act(obs))
            total += 1
            if a_old != a_new:
                mismatch += 1
                if first_bad is None:
                    first_bad = (seed, np.array(obs), a_old, a_new)
            obs, _, term, trunc, _ = env.step(a_old)
            if term or trunc:
                break

    c.check("新旧专家动作完全一致", mismatch == 0,
            f"{total} 步里 {mismatch} 步不一致；首个差异 {first_bad}")
    c.check("对照样本量足够", total > 2000, f"只比了 {total} 步")
    env.close()


# ======================================================================
def test_different_params(c: _Case) -> None:
    """换游戏参数后专家仍要正确（旧版靠读 game 才能自适应，
    新版靠 obs_spec 声明，两者都应成立）。"""
    for n_balls, speed in [(1, 5.0), (3, 5.0), (5, 4.0), (3, 8.0)]:
        env = make_env("catcher", num_balls=n_balls, paddle_speed=speed)
        game = env.unwrapped_game
        old = OldExpertCatcher(game)
        new = ExpertCatcher(env.action_space, game.obs_spec())

        ok = True
        detail = ""
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            for _ in range(200):
                if old.act(obs) != int(new.act(obs)):
                    ok = False
                    detail = f"seed={seed}"
                    break
                obs, _, term, trunc, _ = env.step(old.act(obs))
                if term or trunc:
                    break
            if not ok:
                break
        c.check(f"参数 num_balls={n_balls} speed={speed} 下等价", ok, detail)
        env.close()


# ======================================================================
def test_expert_beats_random(c: _Case) -> None:
    """专家必须显著强于随机 —— 否则说明「专家」根本没在工作。"""
    def run(policy_factory, episodes=15):
        env = make_env("catcher", num_balls=3, paddle_speed=5.0)
        game = env.unwrapped_game
        policy = policy_factory(env, game)
        scores = []
        for seed in range(7000, 7000 + episodes):
            obs, _ = env.reset(seed=seed)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(action))
                if term or trunc:
                    scores.append(info["score"])
                    break
        env.close()
        return float(np.mean(scores))

    rand = run(lambda env, game: RandomPolicy(env.action_space))
    exp = run(lambda env, game: make_expert("catcher", env.action_space,
                                            game.obs_spec()))

    c.check("专家得分远高于随机", exp > rand * 3.0,
            f"随机 {rand:.2f} vs 专家 {exp:.2f}")
    # README 里记录的专家分数约 17.5，给出宽松下界防止实现退化
    c.check("专家得分达到量级预期（>10）", exp > 10.0, f"专家仅 {exp:.2f}")
    print(f"      [参考] 随机 {rand:.2f} / 专家 {exp:.2f}")


# ======================================================================
def test_obs_spec_contract(c: _Case) -> None:
    """obs_spec 的自洽性与错误处理。"""
    env = make_env("catcher", num_balls=4)
    game = env.unwrapped_game
    spec = game.obs_spec()

    c.eq("obs_dim 与观测空间一致", spec["obs_dim"],
         int(np.prod(env.observation_space.shape)))
    c.eq("n_slots 与 num_balls 一致", spec["n_slots"], 4)

    expert = ExpertCatcher(env.action_space, spec)
    c.eq("专家推导的维度正确", expert._expected_obs_dim, spec["obs_dim"])
    env.close()

    # 维度不符必须报错，而不是静默跑出低分
    env2 = make_env("catcher", num_balls=3)
    bad_spec = dict(env2.unwrapped_game.obs_spec())
    bad_spec["obs_dim"] = 999
    try:
        ExpertCatcher(env2.action_space, bad_spec)
        c.failed.append("obs_dim 不符本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env2.close()

    # kind 不对也要报错
    try:
        ExpertCatcher(make_env("dodger").action_space, {"kind": "dodger"})
        c.failed.append("kind 不符本应报错")
    except ObsSpecMismatch:
        c.passed += 1


def test_rich_obs_required(c: _Case) -> None:
    """rich_obs=False 时专家应明确拒绝，而不是算错。"""
    env = make_env("catcher", rich_obs=False)
    try:
        env.unwrapped_game.obs_spec()
        c.failed.append("rich_obs=False 时 obs_spec 本应抛 NotImplementedError")
    except NotImplementedError:
        c.passed += 1
    env.close()


def test_dodger_has_no_expert(c: _Case) -> None:
    """没有专家的游戏要能被显式查询出来（compare.py 靠它显示占位）。"""
    c.check("catcher 有专家", has_expert("catcher"))
    c.check("dodger 暂无专家", not has_expert("dodger"))

    try:
        make_expert("dodger", make_env("dodger").action_space, {})
        c.failed.append("dodger 没有专家，make_expert 本应报错")
    except ValueError:
        c.passed += 1


# ======================================================================
def main() -> int:
    # 无头运行（WSL 无 X 显示时也能测）
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    c = _Case()
    print("[1/5] 新旧专家等价性 ...")
    test_equivalence(c)
    print("[2/5] 不同游戏参数下的等价性 ...")
    test_different_params(c)
    print("[3/5] 专家 vs 随机 ...")
    test_expert_beats_random(c)
    print("[4/5] obs_spec 契约 ...")
    test_obs_spec_contract(c)
    test_rich_obs_required(c)
    print("[5/5] 无专家游戏的降级 ...")
    test_dodger_has_no_expert(c)

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
