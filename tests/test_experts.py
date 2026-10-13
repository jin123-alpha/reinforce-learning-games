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
from experts import (  # noqa: E402
    ExpertCatcher, ExpertDodger, ExpertSnake,
    RandomPolicy, has_expert, make_expert,
)
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


def test_expert_registry(c: _Case) -> None:
    """所有内置游戏都该有专家；未实现的游戏要能被查询出来并明确报错。

    注意：dodger 和 snake 现在**都**有专家了，所以降级路径改用
    「完全不存在的游戏名」来验证 —— 否则这个测试会随着专家越写越多
    而变成永远失败。
    """
    c.check("catcher 有专家", has_expert("catcher"))
    c.check("dodger 有专家", has_expert("dodger"))
    c.check("snake 有专家", has_expert("snake"))

    # 未实现专家的游戏必须能被查询出来，且 make_expert 明确报错
    c.check("不存在的游戏没有专家", not has_expert("no_such_game"))

    env = make_env("dodger")
    try:
        make_expert("no_such_game", env.action_space, {})
        c.failed.append("不存在的游戏，make_expert 本应报错")
    except ValueError:
        c.passed += 1
    env.close()


# ======================================================================
# 躲避专家
# ======================================================================
def test_dodger_obs_spec_contract(c: _Case) -> None:
    """dodger 的 obs_spec 必须自洽，且能被专家构造。"""
    env = make_env("dodger")
    game = env.unwrapped_game
    spec = game.obs_spec()

    c.eq("dodger obs_dim 与观测空间一致", spec["obs_dim"],
         int(np.prod(env.observation_space.shape)))
    c.eq("n_slots 与 n_visible 一致", spec["n_slots"], game.n_visible)
    c.eq("dodger kind 正确", spec["kind"], "dodger")

    expert = ExpertDodger(env.action_space, spec)
    c.eq("躲避专家推导的维度正确", expert._expected_obs_dim, spec["obs_dim"])
    env.close()

    # 维度不符必须报错，而不是静默跑出低分
    env2 = make_env("dodger")
    bad = dict(env2.unwrapped_game.obs_spec())
    bad["obs_dim"] = 999
    try:
        ExpertDodger(env2.action_space, bad)
        c.failed.append("dodger obs_dim 不符本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env2.close()

    # kind 不对也要报错（拿 catcher 的 spec 去建躲避专家）
    env3 = make_env("catcher")
    try:
        ExpertDodger(env3.action_space, env3.unwrapped_game.obs_spec())
        c.failed.append("用 catcher 的 spec 建 ExpertDodger 本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env3.close()


def test_dodger_expert_beats_random(c: _Case) -> None:
    """躲避专家必须显著强于随机。"""
    def run(policy_factory, episodes=20):
        env = make_env("dodger")
        game = env.unwrapped_game
        policy = policy_factory(env, game)
        steps = []
        for seed in range(7000, 7000 + episodes):
            obs, _ = env.reset(seed=seed)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(action))
                if term or trunc:
                    steps.append(info["steps"])
                    break
        env.close()
        return float(np.mean(steps))

    rand = run(lambda env, game: RandomPolicy(env.action_space))
    exp = run(lambda env, game: make_expert("dodger", env.action_space,
                                            game.obs_spec()))

    c.check("躲避专家强于随机", exp > rand * 1.2,
            f"随机 {rand:.1f} vs 专家 {exp:.1f}")
    # 实测专家约 350 步、随机约 200 步，给一个宽松下界防退化
    c.check("躲避专家达到量级预期（>250）", exp > 250.0,
            f"专家仅 {exp:.1f}")
    print(f"      [参考] dodger 随机 {rand:.1f} / 专家 {exp:.1f}")


def test_dodger_expert_never_walks_into_obstacle(c: _Case) -> None:
    """只要存在安全动作，专家就绝不能主动撞上去。

    这是躲避专家的**最低正确性标准**：它不该死于「明明有路可走」。
    同时统计多少比例的死亡是"三个动作全不安全"（结构性死亡）。
    """
    env = make_env("dodger")
    game = env.unwrapped_game
    policy = ExpertDodger(env.action_space, game.obs_spec())

    unavoidable = 0
    avoidable_deaths = []
    for seed in range(7000, 7000 + 15):
        obs, _ = env.reset(seed=seed)
        while True:
            px, obst = policy._unpack(obs)
            # 用与专家相同的"步末位置"判定，统计当前有几个安全动作
            obst_next = policy._advance(obst)
            safe = []
            for a, dx in ((0, -game.player_speed), (1, 0.0),
                          (2, game.player_speed)):
                nx = float(np.clip(px + dx, 0.0,
                                   game.WIDTH - game.PLAYER_W))
                if not policy._collides(nx, obst_next):
                    safe.append(a)

            action, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            if term or trunc:
                if term:
                    if not safe:
                        unavoidable += 1
                    else:
                        avoidable_deaths.append((seed, info["steps"], safe))
                break
    env.close()

    c.check("躲避专家没有一次死于「有路可走」",
            not avoidable_deaths,
            f"{len(avoidable_deaths)} 次可避免死亡：{avoidable_deaths[:3]}")
    print(f"      [参考] 15 回合中 {unavoidable} 次属结构性死亡（无路可走）")


# ======================================================================
# 贪吃蛇专家
# ======================================================================
def test_snake_obs_spec_contract(c: _Case) -> None:
    """snake 的 obs_spec 必须自洽，且能被专家构造。"""
    env = make_env("snake")
    game = env.unwrapped_game
    spec = game.obs_spec()

    c.eq("snake obs_dim 与观测空间一致", spec["obs_dim"],
         int(np.prod(env.observation_space.shape)))
    c.eq("snake kind 正确", spec["kind"], "snake")
    c.eq("网格尺寸与游戏一致", (spec["grid_w"], spec["grid_h"]),
         (game.grid_w, game.grid_h))

    expert = ExpertSnake(env.action_space, spec)
    c.eq("贪吃蛇专家推导的维度正确", expert.obs_dim, spec["obs_dim"])
    env.close()

    # 维度不符必须报错
    env2 = make_env("snake")
    bad = dict(env2.unwrapped_game.obs_spec())
    bad["obs_dim"] = 999
    try:
        ExpertSnake(env2.action_space, bad)
        c.failed.append("snake obs_dim 不符本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env2.close()

    # kind 不对也要报错
    env3 = make_env("dodger")
    try:
        ExpertSnake(env3.action_space, env3.unwrapped_game.obs_spec())
        c.failed.append("用 dodger 的 spec 建 ExpertSnake 本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env3.close()

    # 绝对方向动作下应明确拒绝，而不是算错
    env4 = make_env("snake", relative_actions=False)
    try:
        ExpertSnake(env4.action_space, env4.unwrapped_game.obs_spec())
        c.failed.append("relative_actions=False 时 ExpertSnake 本应报错")
    except ObsSpecMismatch:
        c.passed += 1
    env4.close()


def test_snake_expert_beats_random(c: _Case) -> None:
    """贪吃蛇专家必须显著强于随机。"""
    def run(policy_factory, episodes=20):
        env = make_env("snake")
        game = env.unwrapped_game
        policy = policy_factory(env, game)
        eaten = []
        for seed in range(7000, 7000 + episodes):
            obs, _ = env.reset(seed=seed)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(action))
                if term or trunc:
                    eaten.append(info["eaten"])
                    break
        env.close()
        return float(np.mean(eaten))

    rand = run(lambda env, game: RandomPolicy(env.action_space))
    exp = run(lambda env, game: make_expert("snake", env.action_space,
                                            game.obs_spec()))

    c.check("贪吃蛇专家远强于随机", exp > max(rand * 3.0, 5.0),
            f"随机 {rand:.2f} vs 专家 {exp:.2f}")
    # 实测专家约 24，给一个宽松下界防退化
    c.check("贪吃蛇专家达到量级预期（>8）", exp > 8.0,
            f"专家仅 {exp:.2f}")
    print(f"      [参考] snake 随机 {rand:.2f} / 专家 {exp:.2f}")


def test_snake_expert_never_dies_when_safe_move_exists(c: _Case) -> None:
    """只要观测里存在安全方向，专家就不该撞上去。"""
    env = make_env("snake")
    game = env.unwrapped_game
    policy = ExpertSnake(env.action_space, game.obs_spec())

    avoidable = []
    for seed in range(7000, 7000 + 10):
        obs, _ = env.reset(seed=seed)
        while True:
            danger, _, _, _, _ = policy._unpack(obs)
            has_safe = bool((danger < 0.5).any())

            action, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            if term or trunc:
                # 饿死 / 步数截断不算撞死
                starved = (game.max_steps_without_food > 0
                           and info["steps_since_food"]
                           >= game.max_steps_without_food)
                if term and has_safe and not starved:
                    avoidable.append((seed, info["steps"]))
                break
    env.close()

    c.check("贪吃蛇专家没有一次死于「有安全方向」",
            not avoidable, f"{len(avoidable)} 次可避免死亡：{avoidable[:3]}")


# ======================================================================
def main() -> int:
    # 无头运行（WSL 无 X 显示时也能测）
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    c = _Case()
    print("[1/7] 新旧接球专家等价性 ...")
    test_equivalence(c)
    print("[2/7] 不同游戏参数下的等价性 ...")
    test_different_params(c)
    print("[3/7] 接球专家 vs 随机 ...")
    test_expert_beats_random(c)
    print("[4/7] obs_spec 契约（catcher）...")
    test_obs_spec_contract(c)
    test_rich_obs_required(c)
    print("[5/7] 躲避专家 ...")
    test_dodger_obs_spec_contract(c)
    test_dodger_expert_beats_random(c)
    test_dodger_expert_never_walks_into_obstacle(c)
    print("[6/7] 贪吃蛇专家 ...")
    test_snake_obs_spec_contract(c)
    test_snake_expert_beats_random(c)
    test_snake_expert_never_dies_when_safe_move_exists(c)
    print("[7/7] 专家注册表与降级 ...")
    test_expert_registry(c)

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
