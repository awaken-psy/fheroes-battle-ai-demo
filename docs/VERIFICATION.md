# 验证清单 — M1–M5

> 分两部分：
> **第一部分 自动化验证**——可由脚本/测试构造并执行（pytest + arena），无需人看，CI 也跑这些。
> **第二部分 人工验证**——需要你亲自启动 GUI、用眼睛确认的视觉/交互项（动画、形状、飘字、撤退入局等）。
>
> 里程碑目标与退出标准见 [MILESTONES.md](MILESTONES.md)。

---

# 第一部分 · 自动化验证（Claude 可构造并执行）

```bash
uv sync --group dev      # 安装含 pytest 的开发依赖
```

## 全局检查 — 最近一次执行结果（2026-06-09）

| # | 验证项 | 命令 | 预期 | 实测 |
|---|---|---|---|---|
| A1 | 全部单测 | `uv run pytest` | 82 passed | ✅ 82 passed |
| A2 | engine/ai 零 pygame | `grep -rn "import pygame" ai/ engine/` | 无输出 | ✅ 零命中 |
| A3 | 无显示器导入 engine+ai | `pytest tests/test_planner.py::test_engine_and_ai_import_without_pygame` | returncode 0 | ✅ |
| A4 | 镜像公平 Balanced | `arena --preset Balanced --games 500 --mirror --seed 0` | 40–60% | ✅ 40.6% |
| A5 | 镜像公平 Archer Defense | 同上换预设 | 40–60% | ✅ 49.8% |
| A6 | 镜像公平 Flyer Threat | 同上换预设 | 40–60% | ✅ 55.0% |
| A7 | 法术 vs 无法术 | `arena --preset Balanced --games 400 --seed 0 --hero0` | 带法术方显著占优 | ✅ 100% vs 0% |
| A8 | 撤退可观测 | `arena --config configs/example.json --games 200 --seed 0 --hero0 --hero1 --difficulty Impossible` | 撤退率 > 0 | ✅ 18.0% |

一行复现：`uv run pytest && uv run python scripts/arena.py --preset Balanced --games 500 --mirror --seed 0`
> arena 命令前缀统一为 `uv run python scripts/arena.py`。

## 各里程碑 → 对应自动化测试

| 里程碑 | 验证点 | 测试 / 命令 |
|---|---|---|
| **M1** | 伤害拆分 expected/roll、planner 可复现、解耦守门 | `test_combat.py`、`test_planner.py`、A3 |
| **M2** | 战力公式量纲、threat 距离衰减、反僵局、交替出手 | `test_combat.py`(strength)、`test_scoring.py`、`test_battle_state.py` |
| **M3** | 6 法术效果、buff 到期、英雄每回合≤1 法、施法 AI 阈值/选靶 | `test_spells.py`、`test_spell_ai.py`、A7 |
| **M4** | 续战阈值、撤退结束+告别法术、狂暴、士气运气、planner 不评估 | `test_retreat.py`、`test_berserk.py`、`test_morale_luck.py`、A8 |
| **M5** | 无限反击/吸血/凝视/自愈、能力倍率、谨慎走位 | `test_abilities.py`、`test_cautious.py` |

> 共 12 个测试文件、82 项，全部不依赖显示器；CI 在 push/PR 上跑 `pytest` + arena smoke。

---

# 第二部分 · 人工验证（GUI / 视觉，需你亲自看）

启动：`uv run main.py`（GUI 会给双方默认英雄，故能看到法术）。逐条勾选。

> 说明：**士气/运气在 GUI 默认关闭**（morale/luck=0），不可视觉验证——它们是引擎层、走配置文件，已由自动化 `test_morale_luck.py` 覆盖。

## 布阵界面（Setup）

- [ ] **B1** 左侧调色板**一次只显示 5 个兵种**，右侧滚动条可滚到全部 **8 个**（含新增 Vampire / Troll / Medusa）；滚动方式：鼠标滚轮、拖动滑块、点轨道跳转；滚动条不与下方 Preset 按钮重叠；每行属性 `A D H S x数量` 正确
- [ ] **B2** 选兵种 → 点己方半场放置；右键移除；放置只允许在己方半场
- [ ] **B3** 顶部 **Team** 按钮切换蓝/红方（颜色随之变化）
- [ ] **B4** 三个 **Preset** 按钮可加载阵型（Balanced / Archer Defense / Flyer Threat）
- [ ] **B5** **Start Battle** 在双方都有兵时才变绿可点

## 战斗视觉（形状 / 血条 / 标记）

- [ ] **B6** 形状正确：● 近战、△ 射手、◇ 飞行。**Vampire=◇飞行、Griffin=◇飞行、Troll/Medusa=●**
- [ ] **B7** 符号字母正确：S A G P C **V T M**；血条颜色随血量（绿>50%、黄>25%、红）；单位旁 `数量/总HP`
- [ ] **B8** 黄色圆圈=当前行动单位；红色虚线 + 菱形描边=攻击目标；高亮格=移动路径

## 战斗动画

- [ ] **B9** 移动：单位沿路径平滑滑行到落点
- [ ] **B10** 近战：突刺前冲 + 回弹；被攻击方**反击**（攻击者也掉血飘字）
- [ ] **B11** 射击：黄色弹道线 + 黄点飞向目标
- [ ] **B12** 飘字：红色 `-N` 伤害上浮淡出；击杀显示黄色 `DEAD`；受击格红闪

## 法术（GUI 双方默认有英雄）

- [ ] **B13** 一局中能看到施法：目标格出现**大号青色飘字**（`法术名 -伤害`，如 `Lightning Bolt -75`），持续约 2 秒上浮淡出 + 较明显的红闪（0.4s），与红色近战伤害数字明显区分；调试栏/日志出现 `[CAST]`
- [ ] **B14** Lightning Bolt 等伤害法术能造成显著伤害（大额 `-N`，常一击秒杀一族）

## 撤退（需构造悬殊局面）

- [ ] **B15** 布一个**极悬殊**对局（如蓝方 1 弓箭手 vs 红方满编多骑兵），观察劣势方被打残后**英雄撤退** → 直接进入 Game Over（无需打到全灭）；日志含 `[RETREAT]`，撤退前可能有一发告别法术

## 操控

- [ ] **B16** `Space` 暂停/继续；`1/2/3` 慢/正常/快
- [ ] **B17** `F` 快进瞬间结算 → 回布阵（写日志）；`R` 中止 → 回布阵
- [ ] **B18** `D` 开关调试栏；调试栏显示 AI 决策文本（`[ATK]/[DEF]/[CAUT]` + 动作 + 双方 STR）
- [ ] **B19** `+/-` 缩放、`F11` 全屏、`F12` 截图到 `/tmp/demo-screenshot.png`

## 收尾 / 布局

- [ ] **B20** Game Over：显示胜方文字 + **Play Again** 按钮，点击回布阵
- [ ] **B21** 缩放窗口 / 全屏后，网格、单位、顶栏、调试栏**不错位**
- [ ] **B22**（行为观感，对应学习指南）"Archer Defense" 里蓝方近战会**走去护弓**（绿色路线贴到射手旁），红方全骑兵则**直接冲锋**；谨慎姿态下近战不会无脑冲进慢速敌人攻击范围

---

## 结论判定

- **第一部分**全绿（`uv run pytest` 82 passed + 三预设镜像 40–60% + A7/A8 符合）→ 逻辑层验证通过。
- **第二部分**勾选完毕 → 视觉/交互层验证通过。
- 两部分皆过即认为 M1–M5（v1.0，覆盖原版战斗 AI ~90%）验证通过；宽体单位/攻城等留待 M5b/M6。
