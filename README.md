# fheroes-battle-ai-demo

从 [fheroes2](https://github.com/ihhub/fheroes2) 项目中提取核心战斗 AI 算法的独立演示项目。
无美术资源，纯几何形状 + 颜色。专注**战术层 AI**——只做战场内的决策，不做冒险地图/战略层。

> 本项目复刻了 fheroes2 的战斗 AI 算法，基于 GPL-2.0 许可证发布。
> 原作版权 © ihhub 及 fheroes2 贡献者。详见 [LICENSE](LICENSE)。

## 复刻进度 `v1.0`

里程碑 M1–M5 全部完成，覆盖原版战斗 AI 约 **~90%**（82 个 pytest + CI 守护）。详见 [docs/MILESTONES.md](docs/MILESTONES.md)。

已实现并对齐原版的能力：

- **战术决策**：`planUnitTurn` 分发、射手护弓/逃跑、近战攻防、`analyzeBattleState` 姿态判断
- **保真量纲**：战力公式 `(1+0.1atk+0.05def)×getMonsterBaseStrength×count`；威胁评分 = 期望伤害 / 距离衰减
- **法术系统**：英雄（法术威力/法力/法术书）+ 6 法术（Magic Arrow / Lightning Bolt / Haste / Slow / Bless / Curse）+ 有时限状态效果 + 施法 AI（阈值 + 法力折价 + 各法术 ratio）
- **撤退**：劣势（`myStr×难度系数 < enemyStr`）时英雄撤退 + 告别伤害法术
- **特殊能力**：无限反击 / 吸血 / 死亡凝视 / 自愈
- **引擎机制**：交替出手回合序、士气/运气（军队级，AI 不评估）、狂暴、50 回合反僵局
- **验证闭环**：`scripts/arena.py` 批量 AI-vs-AI 自对弈（镜像 + 置信区间 + 撤退率）

> 留待选做（未排期）：宽体单位、攻城/城墙、更多兵种(~60)、arena 权重调参、与原版决策对照。

## 快速开始

```bash
uv run main.py                          # GUI 模式
uv run main.py configs/example.json     # CLI 无头模式
uv run pytest                           # 跑测试（需先 uv sync --group dev）
uv run python scripts/arena.py --preset Balanced --games 500 --mirror   # 批量自对弈
```

首次运行会自动创建 `.venv` 并安装 pygame。

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
  --hidden-import=ai --hidden-import=ai.planner \
  --hidden-import=ai.evaluation --hidden-import=ai.scoring \
  --hidden-import=ai.strategy --hidden-import=ai.spells --hidden-import=ai.retreat \
  --hidden-import=ui --hidden-import=ui.fonts \
  --hidden-import=ui.renderer --hidden-import=ui.hex_renderer --hidden-import=ui.game \
  --hidden-import=ui.screens --hidden-import=ui.screens.setup \
  --hidden-import=ui.screens.battle \
  main.py
```

产出单个可执行文件 `dist/fheroes-battle-ai-demo`（约 22M），可分发：

```bash
./dist/fheroes-battle-ai-demo                        # GUI
./dist/fheroes-battle-ai-demo configs/example.json   # CLI
```

## 战斗日志

每场战斗结束后自动保存到 `log/` 目录，文件名带时间戳：

```
log/2026-05-30_04-15-23.log
```

日志内容包含：双方阵容、逐回合 AI 决策链、战斗结果、胜负判定。
按 `F` 快进或 `R` 中止也会写入日志。

## 兵种属性

| 兵种 | 攻击 | 防御 | 生命 | 速度 | 伤害 | 数量 | 类型 | 特殊能力 |
|------|------|------|------|------|------|------|------|------|
| Swordsman | 5 | 5 | 15 | 4 | 3 | 20 | 步兵 | — |
| Archer | 4 | 3 | 10 | 3 | 2 | 15 | 射手 | — |
| Griffin | 6 | 4 | 12 | 7 | 3 | 8 | 飞行 | 无限反击 |
| Pikeman | 4 | 7 | 20 | 3 | 2 | 25 | 步兵 | — |
| Cavalry | 7 | 4 | 12 | 6 | 4 | 10 | 步兵 | — |
| Vampire | 6 | 6 | 20 | 6 | 4 | 8 | 飞行 | 吸血 |
| Troll | 10 | 5 | 40 | 4 | 7 | 4 | 步兵 | 自愈 |
| Medusa | 8 | 9 | 25 | 5 | 6 | 4 | 步兵 | 死亡凝视 |

## AI 行为观察指南

对应 [docs/战斗AI学习指南.md](docs/战斗AI学习指南.md) 中的算法：

1. **射手逃跑**：用 "Flyer Threat" 预设，观察弓箭手面对狮鹫时的逃跑决策（狮鹫是飞行单位 → 弓箭手不会逃跑，因为飞兵追得上）
2. **射手射击优先级**：观察弓箭手射击哪个目标（基于 threat 评分）
3. **近战追击"逃不掉"的目标**：慢速步兵 vs 快速飞行兽，AI 会优先追速度慢的目标
4. **防御战术**：用 "Archer Defense" 预设，蓝方弓箭手多 → 近战步兵会保护射手（走绿色路线到射手旁边）
5. **进攻战术**：红方全骑兵 → 不保护射手，直接冲锋

## 项目结构

```
fheroes-battle-ai-demo/
├── main.py                统一入口（GUI / CLI）
├── headless.py            无头战斗引擎（被 main.py 调用）
├── pyproject.toml         项目配置
│
├── configs/               战斗配置文件（CLI 模式输入）
│   ├── example.json       最简阵容
│   └── mage_duel.json     带英雄/法术
│
├── scripts/
│   └── arena.py           批量 AI-vs-AI 自对弈（镜像/置信区间/撤退率）
│
├── docs/                  文档
│   ├── MILESTONES.md      里程碑 M1–M5（含退出标准）
│   ├── TODO.md            复刻路线图
│   └── 战斗AI学习指南.md
│
├── config/                纯数据常量（无逻辑）
│   ├── colors.py          调色板
│   ├── units.py           兵种定义（含特殊能力）
│   ├── presets.py         预设阵型
│   └── timing.py          动画/延迟常量
│
├── engine/                核心引擎（零 pygame，可纯逻辑单测）
│   ├── hex_grid.py        六角格几何 + 寻路（纯几何）
│   ├── unit.py            单位类（属性/效果/能力/战力公式）
│   ├── battle_state.py    战斗状态机 + 伤害/施法/士气运气/胜负
│   ├── hero.py            英雄（法术威力/法力/法术书）
│   ├── spells.py          法术定义 + 有时限状态效果
│   ├── battle_logger.py   战斗日志记录
│   └── actions.py         行动类型（Move/Attack/Skip/Cast/Retreat）
│
├── ai/                    AI 决策系统
│   ├── planner.py         顶层决策调度（含狂暴/谨慎走位）
│   ├── evaluation.py      局面分析（兵力对比、战术标志）
│   ├── scoring.py         威胁评分 + 位置评估
│   ├── spells.py          施法 AI（select_best_spell）
│   ├── retreat.py         撤退决策
│   └── strategy.py        策略枚举（预留，符合原版无独立策略层）
│
├── ui/                    渲染层（依赖 engine + config）
│   ├── game.py            Game 类：窗口、缩放、主循环
│   ├── fonts.py           字体系统 + team helpers
│   ├── renderer.py        共享绘制工具（Popup、按钮、单位）
│   ├── hex_renderer.py    六角格像素层 + 绘图（pygame 集中于此）
│   └── screens/
│       ├── setup.py       布阵界面
│       └── battle.py      战斗界面 + 动画引擎
│
├── tests/                 自动化测试（82 个，无需显示器）
├── log/                   战斗日志（自动生成，gitignore）
└── assets/                资源目录（预留贴图/音效）
```

**依赖方向：`config → engine → ai → ui`**。`engine` 与 `ai` 完全脱离 pygame——所有像素/绘图集中在 `ui/hex_renderer.py` 与 `ui/`，因此核心逻辑可在无显示器环境单测（CI 即如此跑 pytest + arena）。

## 后续发展方向

本项目专注**战术层 AI**——只做战场内决策（移动、攻击、法术、士气等），不做冒险地图/城镇等战略层。核心复刻 M1–M5 已完成（见 [docs/MILESTONES.md](docs/MILESTONES.md)）；以下为留待选做（未排期，多为独立大块或研究级）：

- [ ] **宽体单位 + 双格攻击**：单位占两格 → 改占位/寻路/邻接（`optimalAttackVector` / `doubleCellAttackValue`）
- [ ] **攻城 / 城墙 / 箭塔**：城墙格、守方箭塔、隔墙远程惩罚、桥、地震法术——基本是第二套战斗模式
- [ ] **更多兵种(~60)**：逼近原版完整数值表（接口已就绪，主要是数据录入）
- [ ] **arena 权重调优**：在批量对战闭环上对 threat / 姿态阈值做参数搜索
- [ ] **与原版决策对照**：把 fheroes2 C++ AI 当 oracle 跑同一快照比一致率
- [ ] **阵型编队 / MCTS 陪练**（可选）

### 关键参考文件

| 本项目模块 | fheroes2 源码 | 功能 |
|-----------|--------------|------|
| `ai/planner.py` | [ai_battle.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp) | 战斗 AI 主逻辑 |
| `ai/spells.py` | [ai_battle_spell.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp) | 法术 AI（selectBestSpell） |
| `ai/retreat.py` | [ai_battle.cpp:703-870](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L703-L870) | 撤退决策 |
| `ai/evaluation.py` | [ai_battle.cpp:949](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L949) | 局面分析 |
| `engine/hex_grid.py` | [battle_board.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_board.cpp) | 六角格引擎 |
| `engine/battle_state.py` | [battle_arena.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_arena.cpp) | 战斗机制 |
| `engine/unit.py` | [battle_troop.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_troop.cpp) | 单位逻辑 |

## 技术栈

- **Python 3.11+**
- **pygame** — 渲染、输入、窗口管理
- **uv** — 包管理与依赖解析
- **PyInstaller** — 打包为单文件可执行
- 无第三方 AI/ML 框架，所有决策逻辑手工实现以忠实复刻原版算法

## 安全机制

- **反僵局**：进攻方连续 50 回合无任何死亡 → 撤退判负（对齐原版 `MAX_TURNS_WITHOUT_DEATHS`）
- **200 回合绝对兜底**：触顶按剩余 army strength 判定胜方，防止无限循环
- **`_next_unit()` 无递归**：避免栈溢出
