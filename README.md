# HoMM2 Battle AI Demo

从 [fheroes2](https://github.com/ihhub/fheroes2) 项目中提取核心战斗 AI 算法的独立演示项目。
无美术资源，纯几何形状 + 颜色。专注**战术层 AI**——只做战场内的决策，不做冒险地图/战略层。

## 快速开始

```bash
cd learn/battle-ai-demo
uv run main.py                # GUI 模式
uv run main.py configs/example.json   # CLI 无头模式
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

JSON 数组，每项指定阵营、兵种、位置（`type` 对应兵种表中的名称）：

```json
[
  {"team": 0, "type": "Archer", "col": 2, "row": 3},
  {"team": 0, "type": "Swordsman", "col": 1, "row": 5},
  {"team": 1, "type": "Griffin", "col": 8, "row": 4},
  {"team": 1, "type": "Cavalry", "col": 9, "row": 6}
]
```

示例配置见 `configs/example.json`。

## 打包

```bash
uv pip install pyinstaller
uv run pyinstaller --onefile --name battle-ai-demo \
  --hidden-import=config --hidden-import=config.colors \
  --hidden-import=config.units --hidden-import=config.presets \
  --hidden-import=config.timing \
  --hidden-import=engine --hidden-import=engine.hex_grid \
  --hidden-import=engine.unit --hidden-import=engine.battle_state \
  --hidden-import=engine.battle_logger --hidden-import=engine.actions \
  --hidden-import=ai --hidden-import=ai.planner \
  --hidden-import=ai.evaluation --hidden-import=ai.scoring \
  --hidden-import=ai.strategy \
  --hidden-import=ui --hidden-import=ui.fonts \
  --hidden-import=ui.renderer --hidden-import=ui.game \
  --hidden-import=ui.screens --hidden-import=ui.screens.setup \
  --hidden-import=ui.screens.battle \
  main.py
```

产出单个可执行文件 `dist/battle-ai-demo`（约 22M），可分发：

```bash
./dist/battle-ai-demo                        # GUI
./dist/battle-ai-demo configs/example.json   # CLI
```

## 战斗日志

每场战斗结束后自动保存到 `log/` 目录，文件名带时间戳：

```
log/2026-05-30_04-15-23.log
```

日志内容包含：双方阵容、逐回合 AI 决策链、战斗结果、胜负判定。
按 `F` 快进或 `R` 中止也会写入日志。

## 兵种属性

| 兵种 | 攻击 | 防御 | 生命 | 速度 | 伤害 | 数量 | 类型 |
|------|------|------|------|------|------|------|------|
| Swordsman | 5 | 5 | 15 | 4 | 3 | 20 | 步兵 |
| Archer | 4 | 3 | 10 | 3 | 2 | 15 | 射手 |
| Griffin | 6 | 4 | 12 | 7 | 3 | 8 | 飞行 |
| Pikeman | 4 | 7 | 20 | 3 | 2 | 25 | 步兵 |
| Cavalry | 7 | 4 | 12 | 6 | 4 | 10 | 步兵 |

## AI 行为观察指南

对应学习指南 `learn/ai决策/战斗AI学习指南.md` 中的算法：

1. **射手逃跑**：用 "Flyer Threat" 预设，观察弓箭手面对狮鹫时的逃跑决策（狮鹫是飞行单位 → 弓箭手不会逃跑，因为飞兵追得上）
2. **射手射击优先级**：观察弓箭手射击哪个目标（基于 threat 评分）
3. **近战追击"逃不掉"的目标**：慢速步兵 vs 快速飞行兽，AI 会优先追速度慢的目标
4. **防御战术**：用 "Archer Defense" 预设，蓝方弓箭手多 → 近战步兵会保护射手（走绿色路线到射手旁边）
5. **进攻战术**：红方全骑兵 → 不保护射手，直接冲锋

## 项目结构

```
battle-ai-demo/
├── main.py                统一入口（GUI / CLI）
├── headless.py            无头战斗引擎（被 main.py 调用）
├── pyproject.toml         项目配置
│
├── configs/               战斗配置文件（CLI 模式输入）
│   └── example.json
│
├── config/                纯数据常量（无逻辑）
│   ├── colors.py          调色板
│   ├── units.py           兵种定义
│   ├── presets.py         预设阵型
│   └── timing.py          动画/延迟常量
│
├── engine/                核心引擎（不依赖 pygame 渲染，可单独测试）
│   ├── hex_grid.py        六角格几何 + 寻路        ← battle_board.h/cpp
│   ├── unit.py            单位类                    ← battle_troop.h/cpp
│   ├── battle_state.py    战斗状态机 + 伤害公式      ← battle_arena.h/cpp
│   ├── battle_logger.py   战斗日志记录
│   └── actions.py         行动类型（Move/Attack/Skip）
│
├── ai/                    AI 决策系统（主要扩展方向）
│   ├── planner.py         顶层决策调度              ← ai_battle.cpp
│   ├── evaluation.py      局面分析（兵力对比、战术标志）← ai_battle.cpp:949
│   ├── scoring.py         威胁评分 + 位置评估        ← ai_battle.cpp (散布)
│   └── strategy.py        策略枚举（预留）
│
├── ui/                    渲染层（依赖 engine + config）
│   ├── game.py            Game 类：窗口、缩放、主循环
│   ├── fonts.py           字体系统 + team helpers
│   ├── renderer.py        共享绘制工具（Popup、按钮、单位）
│   └── screens/
│       ├── setup.py       布阵界面
│       └── battle.py      战斗界面 + 动画引擎
│
├── tests/                 自动化测试
├── log/                   战斗日志（自动生成，gitignore）
└── assets/                资源目录（预留贴图/音效）
```

**依赖方向：`config → engine → ai → ui`**，engine 和 ai 可以脱离 pygame 做纯逻辑单元测试。

## 后续发展方向

本项目专注**战术层 AI**——只做战场内的决策（单位移动、攻击、法术、士气等），不做冒险地图、英雄行动、城镇建设等战略层内容。

- [ ] **法术系统** (`engine/spells.py` + `ai/spells.py`)
  - 伤害法术（Magic Arrow, Lightning Bolt）
  - 辅助法术（Slow, Haste, Shield, Bless, Curse）
  - 召唤法术（Summon Earth/Fire/Water/Air Elemental）
  - AI 法术决策：何时施法 vs 普攻/移动
  - 对应 `ai_battle_spell.cpp`

- [ ] **士气/运气系统** (`engine/morale.py`)
  - 高士气 → 额外行动概率
  - 低士气 → 跳过行动概率
  - 运气 → 双倍伤害
  - AI 需要考虑期望值波动

- [ ] **撤退/投降** (`ai/retreat.py`)
  - 判断战局劣势时机
  - 评估撤退代价 vs 全灭代价
  - 对应 `ai_battle.cpp` 中的投降逻辑

- [ ] **阵型/编队** (`ai/formation.py`)
  - 开局布阵优化（前锋、射手后排、飞行侧翼）
  - 战中阵型调整（保持保护关系）

- [ ] **更多兵种**
  - 支持所有 HoMM2 原版兵种（约 60 种）
  - 特殊能力：反击次数、死亡凝视、吸血、自我治疗等

- [ ] **对抗性调优**
  - AI vs AI 批量对战 + 日志分析
  - 参数调优（scoring weights）
  - 对比 fheroes2 原版 AI 的决策

### 关键参考文件

| 本项目模块 | fheroes2 源码 | 功能 |
|-----------|--------------|------|
| `ai/planner.py` | `src/fheroes2/ai/ai_battle.cpp` | 战斗 AI 主逻辑 |
| `ai/spells.py` | `src/fheroes2/ai/ai_battle_spell.cpp` | 法术 AI |
| `ai/evaluation.py` | `ai_battle.cpp:949` | 局面分析 |
| `engine/hex_grid.py` | `src/fheroes2/battle/battle_board.cpp` | 六角格引擎 |
| `engine/battle_state.py` | `src/fheroes2/battle/battle_arena.cpp` | 战斗机制 |
| `engine/unit.py` | `src/fheroes2/battle/battle_troop.cpp` | 单位逻辑 |

## 技术栈

- **Python 3.11+**
- **pygame** — 渲染、输入、窗口管理
- **uv** — 包管理与依赖解析
- **PyInstaller** — 打包为单文件可执行
- 无第三方 AI/ML 框架，所有决策逻辑手工实现以忠实复刻原版算法

## 安全机制

- **200 回合上限**：超时按剩余 army strength 判定胜方，防止无限循环
- **`_next_unit()` 无递归**：避免栈溢出
