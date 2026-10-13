# RL 游戏框架 —— 游戏与环境完全解耦

一个把「**游戏**」和「**强化学习训练**」彻底拆开的小框架。

**核心思路**：游戏只管玩法、得分、画面、以及"给模型的观测"，
完全不知道强化学习的存在；环境层既是**接口契约的拥有者**
（定义"什么算一个兼容的游戏"），也是一个几十行的**薄适配器**；
训练代码则与具体游戏无关。

```
┌─────────────────────────────────────────────────────────┐
│  入口层   train.py  play.py  human_play.py  compare.py   │
│          （与游戏无关，只认统一定义的接口）                  │
└──────────────────────────┬──────────────────────────────┘
                           │  envs/game_env.py  ← 薄适配器
                           │  并保证统一 reward / score 口径
                           ▼
┌─────────────────────────────────────────────────────────┐
│  环境层   envs/base.py     ★ 接口契约（BaseGame）          │
│          envs/game_env.py  BaseGame -> gymnasium.Env      │
│          envs/normalize.py 可选：跨游戏得分归一化           │
└──────────────────────────┬──────────────────────────────┘
                           │  games/* -> envs.base（单向依赖）
                           ▼
┌─────────────────────────────────────────────────────────┐
│  游戏层   games/catcher/   games/dodger/                  │
│          纯 pygame，可独立人类游玩，不依赖任何 RL 库         │
└─────────────────────────────────────────────────────────┘

  横切关注点：
    experts/   规则专家（纯 obs -> action，不接触 game 对象）
    common/    路径解析与 CLI 工具（纯标准库）
    config/    每个游戏一套可版本化的参数
```

## 目录结构

```
catcher/
├── games/                        ★ 游戏层（每个游戏一个子目录）
│   ├── catcher/
│   │   ├── __init__.py           GAME = CatcherGame
│   │   └── catcher.py            接球游戏（多球 + 限速挡板 + 中心得分）
│   ├── dodger/
│   │   ├── __init__.py           GAME = DodgerGame
│   │   └── dodger.py             躲避游戏（生存型任务，结构完全不同）
│   └── __init__.py               自动扫描子目录注册（无需手改）
│
├── envs/                         ★ 环境层
│   ├── base.py                   ★ BaseGame 接口契约 + 人类游玩循环
│   ├── game_env.py               BaseGame -> gymnasium.Env，统一 reward/score
│   ├── normalize.py              可选：跨游戏得分归一化 Wrapper
│   └── __init__.py
│
├── experts/                      ★ 专家策略（一个游戏一个 py 文件）
│   ├── base.py                   BasePolicy / RandomPolicy 接口
│   ├── catcher.py                ExpertCatcher（规则 + EDF 调度）
│   └── __init__.py               注册表
│
├── common/                       ★ 公共工具（纯标准库）
│   ├── paths.py                  全项目唯一的路径解析入口
│   ├── cli.py                    KEY=VALUE 解析、配置加载
│   └── runner.py                 单策略运行循环（play / compare 共用）
│
├── config/                       ★ 配置（按游戏分区）
│   ├── catcher/
│   │   ├── game.json             游戏参数（唯一真相）
│   │   ├── 1.json                DQN 算法超参（引用 game.json）
│   │   └── 2.json                PPO 算法超参
│   └── dodger/
│       ├── game.json
│       └── 1.json
│
├── models/<game>/<run_id>/       ★ 权重（每次训练一个 run，不覆盖）
│   ├── config.json               本次训练完整参数快照
│   ├── best/best_model.zip       评估最优 checkpoint
│   └── final.zip                 训练结束的模型
├── models/<game>/latest.txt      指向最近一次 run
│
├── logs/<game>/<run_id>/         ★ 日志（与 models 的 run 一一对应）
│   ├── tb/                       TensorBoard 事件
│   ├── evaluations.npz           评估曲线
│   └── train.log                 文本日志
│
├── train.py                      训练任意游戏
├── play.py                       加载权重运行
├── human_play.py                 人类游玩（只依赖 games）
├── compare.py                    三窗口对比：随机 / 权重 / 专家
├── compare_child.py              compare 的子进程入口（一个进程一个窗口）
│
├── tests/                        测试（含架构约束与等价性验证）
└── tools/migrate_models.py       旧结构迁移脚本（一次性）
```

### 一条贯穿全项目的路径规则

**先确定游戏名，再在该游戏的子目录里找文件。**

```bash
python train.py --game catcher --config 1.json
#                              ↑ 解析成 config/catcher/1.json
#                                绝不会变成当前目录下的 ./1.json
```

这条规则由 `common/paths.py` 统一实现，四个入口共用。
它保证了每个游戏的配置、权重、日志天然隔离，互不干扰 ——
重构前 `Path("models") / game` 这种字面量散落在 5 处，
**写错了不会报错**，只会静默读错目录，很难排查。

---

## 一、游戏层 `games/`

### `BaseGame` 统一接口

游戏只需要实现 4 个方法 + 2 个属性：

| 成员 | 作用 |
| --- | --- |
| `reset(seed)` | 重置，返回 `(obs, info)` |
| `step(action)` | 推进一帧，返回 `(obs, reward, terminated, truncated, info)` |
| `render()` | `human` 模式画窗口 / `rgb_array` 模式返回 `(H,W,3)` 数组 |
| `close()` | 释放资源 |
| `action_space` | 用 `gymnasium.spaces` 声明 |
| `observation_space` | 用 `gymnasium.spaces` 声明 |

以及两个**可选**的人类操控成员：

| 成员 | 作用 |
| --- | --- |
| `keys_to_action(keys)` | 把 `pygame.key.get_pressed()` 翻译成动作（没按键要返回"不动"）|
| `action_hint` | 按键提示文字 |
| `score` | 游戏自己的得分（**抽象属性，必须实现**）|
| `score_text` | 画面上的得分文字（有默认实现）|

以及一个**可选**成员，只给需要物理量的专家策略用：

| 成员 | 作用 |
| --- | --- |
| `obs_spec()` | 声明观测各分量的物理尺度，让纯 obs 的专家能还原真实物理量 |

> 契约住在 `envs/base.py` 而不是 `games/`，是刻意的：
> **envs 定义"什么算兼容游戏"，games 只负责实现它**。
> 依赖方向因此严格单向：`games/* → envs.base`。
> `games/__init__.py` 仍然重导出 `BaseGame` / `run_human`，
> 所以 `from games import BaseGame` 这类老写法继续可用。
>
> 游戏层只依赖 `pygame`、`numpy` 和 `gymnasium.spaces`
> （后者只是**声明空间**的轻量库，不含任何 RL 算法）。

### 内置游戏

| 游戏 | 动作 | 观测 | 得分 | 玩法 |
| --- | --- | --- | --- | --- |
| **catcher** 接球 | `Discrete(3)` | `Box(23,)` | 落点越靠挡板中心越高 | 限速挡板接住多颗下落小球 |
| **dodger** 躲避 | `Discrete(3)` | `Box(13,)` | 存活步数 | 左右闪避从天而降的障碍物 |

两个游戏**结构故意做得不同**，用来证明框架的通用性：

- catcher：几何物理 + 连续状态 + 稠密奖励 + 多目标调度
- dodger：动态数量的障碍物（观测要"只留最近 N 个，空位填哨兵值"）+ 生存型奖励

### 人类游玩

```bash
python -m games.catcher.catcher   # 直接玩接球（游戏文件可独立运行）
python -m games.dodger.dodger     # 直接玩躲避
# 或者统一走入口脚本（会读取 config/<game>/game.json 的参数）
python human_play.py --game catcher
```

按键：**← / →（或 A / D）**移动，**R** 重开，**P** 暂停，**Esc** 退出。

---

## 二、环境层 `envs/`

### 统一 reward / score 口径

`envs/game_env.py` 是**薄适配器**，不含任何游戏逻辑。但它负责一件
契约级的事：**不管游戏自己怎么写 step()，环境都保证 info 里一定有**

| 字段 | 含义 |
| --- | --- |
| `score` | 游戏自己的得分（= `game.score`） |
| `episode_return` | 本回合累计 reward（**由环境累加**，游戏不用管） |
| `steps` | 本回合已走步数 |

由环境累加而不是让每个游戏自己算，是因为重构前 `play.py` 和
`compare.py` **各自手写**了这段累加逻辑，迟早会漂移。收敛到环境里
之后只有一份真相。

> **注意**：envs 只统一得分的**访问方式**，不统一**数值尺度**。
> catcher 的 score 是接球得分（0~30 量级），dodger 是存活步数
> （0~1000 量级），两者本来就不可比。同一个游戏内比较三种策略
> 是公平的（尺度相同），所以 `compare.py` 不需要归一化。
> 真要跨游戏比大小，用 `envs/normalize.py` 的可选 Wrapper。

```python
from envs import make_env

env = make_env("catcher")            # 或 "dodger"
env = make_env("catcher", num_balls=5, paddle_speed=4)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
# info["score"], info["episode_return"], info["steps"] 一定有
```

**游戏里的任何参数都能透传**：

```python
make_env("catcher", num_balls=5, paddle_speed=4, rich_obs=False)
```

---

## 三、安装（按需分组）

依赖分成三组，用 `pyproject.toml` 的 **extras** 定义。
分组依据是「实际能跑什么」，不是随便切的：

| 组 | 内容 | 能做什么 | 包数 |
| --- | --- | --- | --- |
| **1** | numpy + gymnasium + pygame-ce | 自己玩游戏、跑游戏/环境接口、用专家策略 | ~7 |
| **2** | 组 1 + stable-baselines3（连带 torch） | `play.py` / `compare.py` **加载已有权重**并观看 | ~35 |
| **3** | 组 2 + tensorboard | `train.py` 训练、看曲线、跑对比实验 | ~44 |

```bash
pip install -e .[1]        # 最基本的人类游玩
pip install -e .[2]        # agent 游玩（已有权重）
pip install -e .[3]        # 训练
pip install -e .[1,2,3]    # 显式写全部
pip install -e .[all]      # 全部（别名）
pip install -e .           # 只装组 1（最小可玩）
```

也可以用语义化别名：`.[play]` = `.[1]`，`.[agent]` = `.[2]`，`.[train]` = `.[3]`。

> **为什么不用 `pip install -r requirements.txt[1,2,3]`**
>
> pip 的 extras 语法 `pkg[extra]` 只能用在**包名**上，不能用在
> `-r <文件>` 上。`requirements.txt[1]` 会被 pip 当成一个文件名去找，
> 直接报 `Could not open requirements file`。所以分组安装必须走
> pyproject.toml 的 extras。
>
> `requirements.txt` 仍然保留，作为「不想用 editable 安装、只想装
> 依赖」的兜底 —— 它装的是**组 1**（最小可玩集），不是全部。
> 否则只想玩游戏的人也会被迫下载 2GB 的 CUDA。

### 三个容易踩的坑（已由 `tests/test_packaging.py` 断言）

1. **`dependencies` 必须是最小集。** pip 装 `.[N]` 时实际安装的是
   `dependencies ∪ extra N`。如果把全部依赖写进 `dependencies`，
   那么 `pip install -e .[1]` 也会装 torch，**extras 就完全失效了**。
   所以 `dependencies` 里只放组 1，其余靠 extras 追加。

2. **extras 不支持嵌套引用。** 不能写 `"3": [".[2]"]`，
   Python 的打包规范没这个语法，必须把 `stable-baselines3`
   重复列在组 3 里。测试通过「组 1 ⊆ 组 2 ⊆ 组 3」的集合关系来兜住这一点。

3. **组 1 真的不需要 torch。** 这不是估计 —— `human_play.py` 的
   依赖链实测只有 `numpy` + `gymnasium`（见
   `tests/test_dependencies.py`）。所以在一台没装 torch 的机器上，
   `pip install -e .[1]` 之后就能直接玩游戏。

---

## 四、入口脚本

所有脚本都用 `--game` 选择游戏；**先确定游戏名，再在该游戏的子目录里找文件**。
游戏参数可用 `--set KEY=VALUE` 临时覆盖。

```bash
cd /mnt/d/大三下课程/强化学习/game

# ---- 人类自己玩（只依赖 games，不需要 torch）----
python human_play.py --game catcher
python human_play.py --game dodger --set max_steps=2000

# ---- 训练（配置写在 config/<game>/<n>.json）----
python train.py --game catcher                     # 用 config/catcher/1.json
python train.py --game catcher --config 2.json     # 换一份算法配置（PPO）
python train.py --game dodger  --timesteps 250000  # 覆盖步数
python train.py --game catcher --set num_balls=5   # 覆盖游戏参数
python train.py --game catcher --run-name my_test  # 指定 run 名

# ---- 加载权重运行 ----
python play.py --game catcher                      # latest run 的 best
python play.py --game catcher --weights final      # 用训练结束的模型
python play.py --game catcher --run 20250101-120000_dqn
python play.py --game catcher --episodes 5

# ---- 三窗口对比：随机 | 加载权重 | 专家 ----
python compare.py --game catcher --episodes 10
python compare.py --game dodger  --episodes 10

# ---- TensorBoard ----
tensorboard --logdir logs/catcher          # 单游戏的所有 run
tensorboard --logdir logs                  # 全部游戏
```

### 产物按 run 分开存放，重训不覆盖

```
config/<game>/game.json              游戏参数的唯一真相
config/<game>/<n>.json               算法超参（内部引用 game.json）

models/<game>/<run_id>/config.json   本次训练完整参数快照
models/<game>/<run_id>/best/best_model.zip
models/<game>/<run_id>/final.zip
models/<game>/latest.txt             指向最近一次 run

logs/<game>/<run_id>/tb/             TensorBoard
logs/<game>/<run_id>/evaluations.npz 评估曲线
logs/<game>/<run_id>/train.log       文本日志
```

每次训练新建一个 run 目录（名字形如 `20250101-143022_dqn`），
所以**历史 checkpoint 不会被覆盖** —— 这是能对比"训练 30 万步 vs
60 万步"的前提。重构前只有一个 `best/` + `final.zip`，
第二次训练就把第一次盖掉了。

### `play.py` 的 best / final 为什么要显式区分

| | 含义 |
| --- | --- |
| `best` | `EvalCallback` 在评估中表现最好的 checkpoint（默认）|
| `final` | 训练最后一步的模型 |

两者差别常常很大（DQN 后期会退化）。如果 `play.py` 悄悄换一个，
报告里引用的数字和屏幕上的表现就会对不上 —— 所以它默认 `best`、
把实际加载的那个打印出来，并且在 `best` 缺失时**提示**已回退到 `final`。

---

## 五、新增一个游戏（只需 2 步）

**第 1 步**：建 `games/my_game/` 目录，写一个继承 `BaseGame` 的类。

```python
# games/my_game/my_game.py
from gymnasium import spaces
import numpy as np
from envs.base import BaseGame, run_human

class MyGame(BaseGame):
    name = "my_game"          # 必须与目录名一致（导入时会断言校验）

    def __init__(self, render_mode=None, difficulty=1.0):
        assert render_mode in self.render_modes or render_mode is None
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-1, 1, (8,), np.float32)
        self.closed = False
        # ... 你的状态

    def reset(self, *, seed=None, options=None):
        # ... 返回 (obs, info)
        ...

    def step(self, action):
        # ... 返回 (obs, reward, terminated, truncated, info)
        ...

    def render(self): ...
    def close(self): ...

    @property
    def score(self) -> float:
        return self._score          # 必须实现（抽象属性）

    def keys_to_action(self, keys): ...   # 可选，人类游玩用
    @property
    def action_hint(self): return "..."

if __name__ == "__main__":
    run_human(MyGame(render_mode="human"))
```

**第 2 步**：在 `games/my_game/__init__.py` 里导出 `GAME`。

```python
from .my_game import MyGame
GAME = MyGame
```

**完成。** 训练/演示代码一行都不用改，也不用回头编辑任何注册表 ——
`games/__init__.py` 会自动扫描子目录。

```bash
mkdir -p config/my_game && echo '{"game_kwargs": {}}' > config/my_game/game.json
python train.py --game my_game --config 1.json
python play.py  --game my_game
```

> 目录名、`name` 属性、`GAME` 导出三者必须对得上。
> 不一致时 `games/__init__.py` 会**直接报错**并说明原因 ——
> 因为 `models/<game>/` 和 `logs/<game>/` 都按目录名组织，
> 名字不一致会导致"报告里写 A、实际加载 B 的权重"。
>
> 如果游戏是**连续动作**，把 `action_space` 声明成 `spaces.Box(...)` 即可，
> PPO 直接支持；`play.py` / `train.py` 无需改动。

> 如果游戏是**连续动作**，把 `action_space` 声明成 `spaces.Box(...)` 即可，
> PPO 直接支持；`play.py` / `train.py` 无需改动。

---

## 六、专家策略（不训练的硬处理）

`experts/catcher.py` 里实现了**接球游戏的规则专家**，接口与 SB3 模型完全一致：

```python
from experts import make_expert
policy = make_expert("catcher", env.action_space, game.obs_spec())
action, _ = policy.predict(obs, deterministic=True)
```

### 专家是**纯 `obs → action`** 的，不接触 game 对象

第二、三个参数刻意设计成 `(action_space, spec)` 而不是 `game` 实例：

| | 旧做法 | 现在 |
| --- | --- | --- |
| 拿什么 | `game` 对象，直接读 `game.WIDTH`、`game.num_balls` | 只拿动作空间 + 观测尺度声明 |
| 风险 | 游戏改内部常量 → 专家**静默**算错落点 | 尺度由游戏 `obs_spec()` 声明，改常量不会算错 |
| 性质 | 能读到观测里没有的状态（近似作弊） | 严格只吃观测，是**合法的对照基线** |

因为签名与 SB3 一致，所以

```python
RandomPolicy  /  ExpertCatcher  /  SB3 model
```

三者在 `play.py` / `compare.py` 里是**同一种东西**（都只有 `.predict`），
调用方不需要任何 `if kind == ...` 分支 —— 这正是三窗口对比能写得
很干净的原因。

**思路**

| 步骤 | 做法 |
| --- | --- |
| ① 落点已知 | rich 观测里直接给了每球"落地剩余步数 t"和"落点 x"，不必自己外推 |
| ② 可达性判定 | `\|x_land − 中心\| ≤ k·s + (PADDLE_W/2 + BALL_R)` |
| ③ EDF 调度 | 在赶得上的球里优先接**最早落地**的（单位收益任务调度的最优贪心）|
| ④ 精确瞄准 | 对准落点**中心**（正中 1.0 分，边缘 0.2 分）|
| ⑤ 兜底 | 全都赶不上时移到屏幕中央 |

> rich 观测是专家的前提：`rich_obs=False` 时观测里没有落点信息，
> 此时调用 `obs_spec()` 会**明确报错**，而不是让专家算出一个错的落点
> 然后表现为"得分莫名其妙很低"。

### 实测对比（catcher，3 球 / 板速 5）

| 策略 | return | score | 接住/回合 | 平均每球 | 失误 |
| --- | --- | --- | --- | --- | --- |
| 随机策略 | −7.96 ± 1.30 | 2.04 | 3.9 | 0.516 | 10.0 |
| DQN（训练 60 万步）| +3.02 ± 3.47 | 10.59 | 18.5 | 0.575 | 7.6 |
| **专家策略（0 训练）** | **+12.19 ± 3.24** | **17.56** | **21.4** | **0.821** | **5.4** |
| 瞬移神谕（理论上界） | +24.50 | 24.56 | 27.1 | 0.905 | 0.1 |

> 专家策略是 DQN 的 **4 倍**，且不需要训练。
> 诊断 630 次接球：**73% 的偏差 ≤ 2.5px**（移动粒度的理论极限，得分≈0.93），
> 剩下 22% 是"勉强够到"的边缘接球 —— 这是**速度受限下的结构性上限**。

数字可用 `python compare.py --game catcher --episodes 10` 复现
（三窗口跑同步种子，逐回合明细可直接横向对比）。

---

## 七、`compare.py` 的三窗口

```
+------------+  +------------+  +------------+
|   随机     |  |  加载权重  |  |   专家     |
+------------+  +------------+  +------------+
  左 (x=0)       中 (x=420)     右 (x=840)
```

### 为什么必须用子进程

pygame 的 `display.set_mode()` 是**全局单例** —— 同一个进程里再调一次
只会替换掉当前窗口，不可能同时开两个。而且 `pygame.quit()` 也是全局的，
一个游戏实例 `close()` 会把其它窗口一起干掉。

所以三个窗口只能是**三个进程**（`compare_child.py`）。
父进程用 `SDL_VIDEO_WINDOW_POS` 环境变量给每个子进程指定位置，
于是它们自然"从左至右"排开。

### 种子同步 + 回合屏障

三个窗口跑**同一个种子序列**（默认 7000 起），并且通过标记文件
在每回合开始前**对齐**：

- 同种子 → 三个策略面对完全相同的球序，差异**只来自策略本身**
- 屏障 → 避免"左窗口打到第 8 回合、右窗口还在第 2 回合"这种没法对照的情况
  （随机策略几回合就死，专家能活很久，不屏障的话画面会迅速错位）

### 结果汇总

每个子进程把逐回合结果写成 jsonl，父进程读出来打印对比表：

```
策略                       平均 score      平均 return         平均步数
--------------------------------------------------------------------------
随机策略                         3.50          -6.50        679.0
加载权重                        11.17           3.50       1000.0
专家策略                        17.03          11.03       1000.0

逐回合 score（同种子，可横向对比）：
  回合   seed          随机策略          加载权重          专家策略
   1   7000          5.66          9.95         18.33
   2   7001          0.52         12.10         16.37
```

用文件而不是 stdout 传递结果，是为了让"中途关掉某个窗口"
也能拿到已完成的数据。

---

## 八、测试

```bash
python tests/run_all.py            # 跑全部快速套件（约 40 秒）
RUN_SLOW=1 python tests/run_all.py # 连打包测试一起跑（需联网建 venv）
python tests/test_experts.py       # 也可以单个跑
```

| 套件 | 检查什么 |
| --- | --- |
| `test_compile` | 全部 .py 语法正确、每个模块可独立导入、无循环导入 |
| `test_paths` | 路径解析规则（按游戏分目录、run 指针、权重回退、拒绝路径穿越）|
| `test_cli` | KEY=VALUE 解析（**含对旧写法的兼容性**）、配置引用解析 |
| `test_experts` | 专家**新旧实现等价性**（4000+ 步逐动作比对）+ 强于随机 |
| `test_human_play` | `run_human` 不崩溃；`init_display` 的**调用顺序**（AST 断言）|
| `test_dependencies` | 依赖分层：common 纯标准库、游戏层不含 torch、human_play 不需 RL |
| `test_packaging` | extras 分组：递进关系、默认依赖是最小集、**真实 pip dry-run 解析** |
| `test_decoupling` | 架构约束：games / human_play 不引入 RL 库（含源码扫描防绕过）|
| `test_structure` | 目录结构、games 子目录形式、config 布局、旧文件已清理 |

> **`test_experts` 的等价性验证值得一提**：把专家从"读 game 对象"
> 改成"纯 obs"是个行为等价的重构，但"等价"不能靠肉眼保证。
> 测试里内联了一份**重构前的实现**，让新旧两版在 4000+ 步真实轨迹上
> 逐动作比对，全部一致 —— 这样重构才敢说没改坏东西。
>
> **`test_packaging` 会真跑 pip**：它建一个临时 venv，对
> `.[1]` / `.[2]` / `.[3]` 各跑一次 `pip install --dry-run --report`，
> 断言组 1 里确实没有 torch、组 2 里确实没有 tensorboard。
> 因为要联网解析元数据，默认跳过。

---

## 九、环境信息

- Python 3.11.17 / gymnasium 1.4.0 / stable-baselines3 2.9.0
- torch 2.14.1（CPU）/ pygame-ce 2.5.8
- WSL2 + WSLg（`x11` 驱动，窗口 400×600）验证通过
- 环境位置：`/home/jinfy/.conda/envs/rl-sb3/bin/python`

依赖分层（由 `tests/test_dependencies.py` 实测断言）：

| 层 | 允许的第三方依赖 |
| --- | --- |
| `common/paths.py`、`common/cli.py` | **无**（纯标准库）|
| `common/runner.py` | numpy |
| `games/`、`envs/`、`experts/` | numpy + gymnasium（**不含 torch**）|
| `human_play.py` | numpy + gymnasium（**不需要 torch 就能玩**）|
| `train.py` / `play.py` | 延迟导入 SB3（`--help` / `--list-games` 不用等 torch）|
| `compare_child.py` | SB3（它是需要加载权重的入口）|