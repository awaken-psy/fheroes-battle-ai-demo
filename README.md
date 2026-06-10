# fheroes-battle-ai-demo

从 [fheroes2](https://github.com/ihhub/fheroes2) 项目中提取核心战斗 AI 算法的独立演示项目。
无美术资源，纯几何形状 + 颜色。专注**战术层 AI**——只做战场内的决策，不做冒险地图/战略层。

> 本项目复刻了 fheroes2 的战斗 AI 算法，基于 GPL-2.0 许可证发布。
> 原作版权 © ihhub 及 fheroes2 贡献者。详见 [LICENSE](LICENSE)。

## 项目进度

里程碑 M1–M7e（规则层）+ A1–A4（经典 AI 决策层）+ R1–R7（深度学习训练管线）**全部完成**。

- **561 个测试**全通过，CI 守护
- **63 种兵种**，**38 种法术**，规则保真度 ~99%
- 经典 AI 决策覆盖率 ~97%（126 条审计逐条对齐）
- 3600 局实战验证
- 完整 PPO 自我博弈训练管线（PyTorch + Gymnasium）

详见 [docs/MILESTONES.md](docs/MILESTONES.md)。

### 三大里程碑层

| 层 | 里程碑 | 说明 |
|----|--------|------|
| **规则层** | M1–M7e | 战斗引擎完整实现：63 兵种、38 法术、攻城、士气/运气、特殊能力 |
| **经典 AI** | A1–A4 | 忠实复刻 fheroes2 C++ 源码决策链，~97% 覆盖率 |
| **深度学习** | R1–R7 | CNN+PPO 自我博弈训练管线，从观测编码到训练脚本 |

### 深度学习 R 系列路线

```
R1(可插拔骨架) → R2(观测编码) ──┐
                               ├→ R4(环境封装) → R5(神经网络) ──┐
              R3(动作空间) ────┘                                ├→ R6(PPO训练器) → R7(训练管线)
```

| 里程碑 | 文件 | 说明 | 测试 |
|--------|------|------|------|
| R2 | `ai/observation.py` | 33 通道 hex grid + 20 维全局向量，player-relative | 38 |
| R3 | `ai/action_space.py` | 13566 维扁平离散 + 合法性 mask | 53 |
| R4 | `ai/env.py`, `ai/self_play.py` | Gymnasium BattleEnv + 自博弈 runner | 32 |
| R5 | `ai/deep/model.py` | 4×64 ResBlock CNN + Policy/Value 双头，~4.15M 参数 | 21 |
| R6 | `ai/deep/trainer.py` | TrajectoryBuffer + GAE + PPOTrainer (CleanRL 风格) | 30 |
| R7 | `ai/deep/player.py`, `ai/deep/pipeline.py`, `scripts/train.py` | DeepAI + 训练管线 CLI | 31 |

## 快速开始

```bash
uv sync                                    # 安装依赖（含 PyTorch）
uv run main.py                             # GUI 模式
uv run main.py configs/example.json        # CLI 无头模式
uv run pytest                              # 跑测试（561 个）
uv run python scripts/arena.py --preset Balanced --games 500 --mirror   # 经典 AI 批量自对弈
```

### 训练深度学习 AI

```bash
# 从零训练
uv run python scripts/train.py --total-steps 100000 --eval-interval 5000

# 从 checkpoint 恢复
uv run python scripts/train.py --resume checkpoints/checkpoint_5000.pt

# 自定义阵容
uv run python scripts/train.py --config configs/even_clash.json --eval-games 50

# 训练输出 JSON lines 日志，可用 jq 过滤
uv run python scripts/train.py --total-steps 50000 | jq 'select(.type=="eval")'
```

训练参数全部可通过 CLI 控制（`--lr`, `--gamma`, `--clip-eps`, `--phase1-steps` 等）。

## 使用方式

### GUI 模式（无参数）

```bash
uv run main.py
```

#### 摆兵阶段

| 操作 | 说明 |
|------|------|
| 点击左侧兵种 | 选择要放置的兵种 |
| 点击左侧 **Team** 按钮 | 切换蓝方/红方 |
| 点击战场格子 | 放置选中兵种（只能在己方半场） |
| 右键点击战场格子 | 移除该格的单位 |
| 点击 **Start Battle** | 开始战斗 |
| 点击 **Preset** | 加载预设阵型 |

#### 战斗阶段

| 按键 | 说明 |
|------|------|
| `Space` | 暂停/继续 |
| `1` / `2` / `3` | 慢速 / 正常 / 快速 |
| `F` | 快进：瞬间算完所有回合，写日志后回到摆兵界面 |
| `R` | 中止：立即结束本局（写入日志），回到摆兵界面 |
| `D` | 开关 AI 决策调试信息 |
| `F12` | 截图到 `/tmp/demo-screenshot.png` |
| `+` / `-` | 调整窗口大小 |
| `F11` | 全屏切换 |

#### 战场上的图形含义

| 形状 | 含义 |
|------|------|
| ● 圆形 | 近战步兵 |
| △ 三角形 | 射手（弓箭手） |
| ◇ 菱形 | 飞行单位 |

黄色圆圈 = 当前行动单位
红色虚线 = 射击/攻击目标
高亮格子 = 移动路径
兵种旁数字 = `数量/总HP`（如 `12/120`）

### CLI 无头模式（传配置文件）

```bash
# 单场战斗
uv run main.py configs/example.json

# 指定输出路径
uv run main.py configs/example.json -o result.log

# 批量
uv run main.py configs/*.json
```

不启动任何 GUI，读取配置 → 跑完战斗 → 写日志。

#### 配置文件格式

最简形式是 JSON 数组，每项指定阵营、兵种、位置（`type` 对应兵种表中的名称）：

```json
[
  {"team": 0, "type": "Archer", "col": 2, "row": 3},
  {"team": 0, "type": "Swordsman", "col": 1, "row": 5},
  {"team": 1, "type": "Griffin", "col": 8, "row": 4},
  {"team": 1, "type": "Cavalry", "col": 9, "row": 6}
]
```

也支持对象形式，附带英雄（法术）、难度、士气/运气：

```json
{
  "units": [ {"team": 0, "type": "Pikeman", "col": 1, "row": 2}, ... ],
  "heroes": {
    "0": {"name": "Solmyr", "power": 4, "spell_points": 18,
          "spells": ["Lightning Bolt", "Haste", "Bless"]},
    "1": {"name": "Crag Hack", "power": 2, "spell_points": 12}
  },
  "difficulty": "Normal",
  "morale": {"0": 1, "1": 0},
  "luck": {"0": 0, "1": 2}
}
```

示例：`configs/example.json`（最简）、`configs/mage_duel.json`（带英雄/法术）。

## 打包

```bash
uv pip install pyinstaller
uv run pyinstaller --onefile --name fheroes-battle-ai-demo \
  --hidden-import=config --hidden-import=config.colors \
  --hidden-import=config.units --hidden-import=config.presets \
  --hidden-import=config.timing \
  --hidden-import=engine --hidden-import=engine.hex_grid \
  --hidden-import=engine.unit --hidden-import=engine.battle_state \
  --hidden-import=engine.battle_logger --hidden-import=engine.actions \
  --hidden-import=engine.hero --hidden-import=engine.spells \
  --hidden-import=engine.castle \
  --hidden-import=ai --hidden-import=ai.base --hidden-import=ai.factory \
  --hidden-import=ai.classic --hidden-import=ai.classic.planner \
  --hidden-import=ai.classic.evaluation --hidden-import=ai.classic.scoring \
  --hidden-import=ai.classic.strategy --hidden-import=ai.classic.spells \
  --hidden-import=ai.classic.retreat \
  --hidden-import=ui --hidden-import=ui.fonts \
  --hidden-import=ui.renderer --hidden-import=ui.hex_renderer \
  --hidden-import=ui.game \
  --hidden-import=ui.screens --hidden-import=ui.screens.setup \
  --hidden-import=ui.screens.battle \
  main.py
```

产出单个可执行文件 `dist/fheroes-battle-ai-demo`，可分发：

```bash
./dist/fheroes-battle-ai-demo                        # GUI
./dist/fheroes-battle-ai-demo configs/example.json   # CLI
```

> 注：打包仅包含经典 AI（ClassicAI），不包含深度学习组件（PyTorch）。

## 战斗日志

每场战斗结束后自动保存到 `log/` 目录，文件名带时间戳：

```
log/2026-05-30_04-15-23.log
```

日志内容包含：双方阵容、逐回合 AI 决策链、战斗结果、胜负判定。
按 `F` 快进或 `R` 中止也会写入日志。


## 项目结构

```
fheroes-battle-ai-demo/
├── main.py                  统一入口（GUI / CLI）
├── headless.py              无头战斗引擎（被 main.py 调用）
├── pyproject.toml           项目配置 + 依赖
│
├── configs/                 战斗配置文件（CLI / 训练脚本输入）
│   ├── example.json         最简阵容
│   ├── mage_duel.json       带英雄/法术
│   └── even_clash.json      多兵种对抗（训练常用）
│
├── scripts/
│   ├── arena.py             批量 AI-vs-AI 自对弈（镜像/置信区间/撤退率）
│   ├── train.py             PPO 自我博弈训练脚本（R7）
│   ├── ai_validation.py     AI 决策审计工具
│   └── fingerprint.py       代码指纹生成
│
├── docs/                    文档
│   ├── MILESTONES.md        里程碑（含退出标准）
│   ├── VERIFICATION.md      验证清单
│   ├── rules-audit.md       规则对照（318 项）
│   ├── ai-audit.md          AI 行为审计（126 条）
│   └── 战斗AI学习指南.md
│
├── config/                  纯数据常量（无逻辑）
│   ├── colors.py            调色板
│   ├── units.py             兵种定义（63 种，含特殊能力）
│   ├── presets.py           预设阵型
│   └── timing.py            动画/延迟常量
│
├── engine/                  核心引擎（零 pygame，可纯逻辑单测）
│   ├── hex_grid.py          六角格几何 + 寻路
│   ├── unit.py              单位类（属性/效果/能力/战力公式）
│   ├── battle_state.py      战斗状态机 + 伤害/施法/士气运气/胜负
│   ├── hero.py              英雄（法术威力/法力/法术书）
│   ├── spells.py            法术定义（38 种）+ 有时限状态效果
│   ├── castle.py            攻城（城墙/护城河/箭塔）
│   ├── battle_logger.py     战斗日志记录
│   └── actions.py           行动类型（Move/Attack/Skip/Cast/Retreat）
│
├── ai/                      AI 决策系统
│   ├── base.py              AIPlayer 抽象基类
│   ├── factory.py           工厂 + 注册表（create_ai）
│   ├── observation.py       观测编码（R2: 33 通道 grid + 20 维全局）
│   ├── action_space.py      动作空间（R3: 13566 维 + 合法性 mask）
│   ├── env.py               Gymnasium BattleEnv（R4）
│   ├── self_play.py         自博弈 runner + eval_vs_classic
│   ├── classic/             经典 AI（忠实复刻 fheroes2 C++ 源码）
│   │   ├── planner.py       顶层决策调度
│   │   ├── evaluation.py    局面分析
│   │   ├── scoring.py       威胁评分 + 位置评估
│   │   ├── spells.py        施法 AI
│   │   ├── retreat.py       撤退决策
│   │   └── strategy.py      策略枚举
│   └── deep/                深度学习 AI
│       ├── model.py         BattleNet CNN（R5: ResBlock + Policy/Value 双头）
│       ├── trainer.py       PPOTrainer（R6: GAE + PPO-Clip）
│       ├── player.py        DeepAI(AIPlayer) + make_agent_fn（R7）
│       └── pipeline.py      训练工具（配置/课程/checkpoint）（R7）
│
├── ui/                      渲染层（依赖 engine + config）
│   ├── game.py              Game 类：窗口、缩放、主循环
│   ├── fonts.py             字体系统 + team helpers
│   ├── renderer.py          共享绘制工具
│   ├── hex_renderer.py      六角格像素层（pygame 集中于此）
│   └── screens/
│       ├── setup.py         布阵界面
│       └── battle.py        战斗界面 + 动画引擎
│
├── tests/                   自动化测试（561 个，无需显示器）
└── log/                     战斗日志（自动生成，gitignore）
```

**依赖方向**：`config → engine → ai → ui`。`engine` 与 `ai` 完全脱离 pygame——所有像素/绘图集中在 `ui/`，核心逻辑可在无显示器环境单测（CI 即如此跑 pytest + arena）。

**AI 架构**：`ai.base.AIPlayer` 抽象接口 → `ai.classic.ClassicAI`（规则驱动）+ `ai.deep.player.DeepAI`（神经网络驱动），通过 `ai.factory.create_ai()` 统一创建。

## 技术栈

- **Python 3.11+**
- **PyTorch** — 深度学习（CNN 骨干 + PPO 训练）
- **Gymnasium** — RL 环境标准接口
- **NumPy** — 数值计算
- **pygame** — 渲染、输入、窗口管理
- **uv** — 包管理与依赖解析
- **PyInstaller** — 打包为单文件可执行

## 安全机制

- **反僵局**：进攻方连续 50 回合无任何死亡 → 撤退判负（对齐原版 `MAX_TURNS_WITHOUT_DEATHS`）
- **200 回合绝对兜底**：触顶按剩余 army strength 判定胜方，防止无限循环
- **`_next_unit()` 无递归**：避免栈溢出

## 后续发展方向

核心复刻 + 深度学习管线已完成。以下为留待选做（未排期）：

- [ ] **实际训练与调参**：运行 `train.py`，观察 vs ClassicAI 胜率曲线，调优超参数
- [ ] **宽体单位 + 双格攻击**：单位占两格 → 改占位/寻路/邻接
- [ ] **更多阵容训练**：扩展到多兵种/带英雄/攻城场景
- [ ] **arena 权重调优**：经典 AI threat / 姿态阈值参数搜索
- [ ] **与原版决策对照**：把 fheroes2 C++ AI 当 oracle 跑同一快照比一致率

### 关键参考文件

| 本项目模块 | fheroes2 源码 | 功能 |
|-----------|--------------|------|
| `ai/classic/planner.py` | [ai_battle.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp) | 战斗 AI 主逻辑 |
| `ai/classic/spells.py` | [ai_battle_spell.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp) | 法术 AI |
| `ai/classic/retreat.py` | [ai_battle.cpp:703-870](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L703-L870) | 撤退决策 |
| `ai/classic/evaluation.py` | [ai_battle.cpp:949](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L949) | 局面分析 |
| `engine/hex_grid.py` | [battle_board.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_board.cpp) | 六角格引擎 |
| `engine/battle_state.py` | [battle_arena.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_arena.cpp) | 战斗机制 |
| `engine/unit.py` | [battle_troop.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_troop.cpp) | 单位逻辑 |
