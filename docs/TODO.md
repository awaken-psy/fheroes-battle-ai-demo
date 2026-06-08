# TODO — 战斗 AI 复刻路线图

> 评价基准:原版 `src/fheroes2/ai/` 的**战斗 AI 本体** ——
> `ai_battle.cpp` (2091 行) + `ai_battle_spell.cpp` (958 行) ≈ 3000 行,纯规则式、单层贪心启发式,无搜索树/ML。
> (`ai_planner_*.cpp` 约 6000 行是冒险地图/王国经营 AI,**不在本项目复刻范围**。)
>
> 目标:在「战术层战斗 AI」范围内,逐步把复刻保真度从「战术核心子集」推进到「与原版决策行为可对照一致」。
>
> 当前完成度:近战攻防 + 射手决策 + 局面姿态评估(高保真),约覆盖原版战斗 AI 的 **~45%**。
> 战术决策骨架完整,缺**法术 / 撤退 / 攻城**三大块。
>
> 状态图例:`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成

---

## 现状盘点(已对照源码确认)

| 维度 | 实际情况 |
|---|---|
| 本 demo | ~1000 行 Python,纯战术层战斗 AI |
| 已复刻(高保真) | `planUnitTurn` 分发、射手决策(护弓/逃跑/被贴脸近战)、近战攻防、`analyzeBattleState` 姿态判断、`should_defend` 五条规则 |
| 主要缺口 | 法术系统、撤退/投降、攻城/城墙、宽体单位、士气运气、狂暴 |

**核对发现的关键偏差:**
1. **战力公式跑偏**(`engine/unit.py:54`):现为 `(atk+def)·count·damage·hp/200`,原版为 `(1+0.1·atk+0.05·def)×baseStrength×count`。所有姿态阈值(`×10`/`×6`/`0.66`)都建在 strength 量纲上,公式错则全盘漂移 —— **保真修正里最该先做的一处**。
2. **`threat()` 仍是占位级**(`ai/scoring.py:12`):固定倍率 1.2/1.1/1.3,离原版 `evaluateThreatForUnit` 的"双向伤害+距离衰减+反击折算"差距大。
3. **`strategy.py` 是 12 行空壳 —— 符合原版**:原版战斗 AI 没有独立 strategy 层,personality 影响很小。**不要为对齐而发明原版没有的策略系统**。
4. **从未做过 AI-vs-AI 验证**:所有"高保真"目前只靠人眼读代码核对 —— 这是当前最大隐患。

---

## 🔴 P0 — 先建验证闭环(最高优先)

> 已积累一批"声称对齐原版"的逻辑,但零验证。**继续加功能前,必须先能量化"对齐到什么程度"**,否则 P1/P2 改完无法判断是改对还是改错。
> ⚠️ 顺序约束:第 1 项(解耦 pygame)是其余三项的前置,必须先做。

- [ ] **解耦 `hex_grid.py` / engine 与 pygame**(前置)
  - 移除顶部 `import pygame`,纯几何逻辑下沉,渲染相关上移到 `ui/`
  - 兑现 README "engine/ai 可脱离 pygame 单测" 的承诺;不解耦则无法批量跑
- [ ] **自对弈 / AI-vs-AI 批量对战框架**
  - 基于 `headless.py` 扩成批量跑 N 局、镜像配队 + 轮换先手、输出胜率
  - 报告含样本量与置信区间(避免 10 局得 40% 这种统计噪声)
  - 入口建议:`scripts/arena.py --games 500 --mirror`
- [ ] **AI 决策回归测试**(`tests/test_planner.py`)
  - 固定战场快照 → 断言 `planner` 输出特定动作(护弓、追击、射手逃跑等)
  - 锁住当前已对齐原版的行为,防止重构时回归
- [ ] **伤害公式单元测试**(`tests/test_combat.py`)
  - 覆盖攻防差 +10%/-5%、3.0× 上限、射手近战 0.5 惩罚

## 🟠 P1 — 保真度修正(小改动,高回报)

> 三项都是**改现有代码**而非新增模块,工作量小但直接拉高保真度。紧跟 P0。

- [ ] **战力公式对齐原版量纲**(`engine/unit.py` / `ai/evaluation.py`)—— P1 内最优先
  - 原版:`GetMonsterStrength = (1 + 0.1·atk + 0.05·def) × monsterBaseStrength × count`
    (`monster.cpp:146`, `army_troop.cpp:76`)
  - 影响:所有阈值判断都建立在 strength 上,量纲不对会系统性偏移姿态判断
- [ ] **`threat()` 升级为双向伤害+反击估值**(`ai/scoring.py`)
  - 对齐 `Unit::evaluateThreatForUnit`(`battle_troop.cpp:1007`):
    潜在伤害 + 距离衰减 + 双击折算 + 阵营符号修正(友军 ×-2 / 已动敌 /1.25 / 镜像 ×10)
- [ ] **`isLimitOfTurnsExceeded` 反僵局**(替换当前 200 回合硬上限)
  - 原版:进攻方连续 `MAX_TURNS_WITHOUT_DEATHS=50` 回合无死亡 → 撤退(`ai_battle.cpp:630`)

## 🟡 P2 — 补齐三大缺失模块(从 ~45% 推到 ~85% 的主体)

> 建议顺序:先法术(最大单块,观感差距最明显)、再撤退。

- [ ] **法术系统**(`engine/spells.py` + `ai/spells.py`)— 原版 `ai_battle_spell.cpp` 全部 958 行
  - 伤害(Magic Arrow / Lightning Bolt)、增益(Haste/Bless/Shield)、减益(Slow/Curse)、召唤、复活
  - **AI 法术决策核心**:施法阈值 `myStr² / enemyStr × 0.04`、法力开方折价 `value/√(sp/3)`、
    各法术 ratio(Slow 0.1·失速 / Haste 0.05·增速 / Bless·Curse 0.15 …)与时长系数
  - buff/debuff 必须是**有时限的临时状态**(不可像简化版那样永久改属性)
- [ ] **撤退 / 投降决策**(`ai/retreat.py`)— 原版 `ai_battle.cpp:703-870`
  - 续战条件 `myStr × 难度系数 >= enemyStr`;否则按宝物/可否再雇佣/英雄主属性决定 Retreat / Surrender
  - 撤退前放一发"告别伤害法术"`farewellSpellcast`
- [ ] **狂暴 `berserkTurn`**(`ai_battle.cpp:508`)— 攻击/移向最近单位(不分敌我),小而独立可顺手做
- [ ] **士气 / 运气系统**(`engine/morale.py`)
  - 引擎层:高士气→额外行动、低士气→跳过、运气→双倍伤害
  - ⚠️ **原版 AI 决策本身不评估士气/运气**(`ai_battle.cpp:1289` 显式注释),仅引擎随机生效——
    复刻时不要让 planner 去算期望波动,否则即偏离基准

## 🟢 P3 — 战场机制与内容扩展(按需)

- [ ] **攻城 / 城墙 / 箭塔逻辑**
  - 守方塔战力计入射手、城墙远程惩罚、`cellsUnderWallsIndexes={7,28,49,72,95}`、地震法术
- [ ] **宽体单位(占两格)+ 双格攻击** `optimalAttackVector` / `doubleCellAttackValue`
- [ ] **谨慎进攻走位** `findOptimalPositionForSubsequentAttack`(`ai_battle.cpp:351`)
  - 沿路径累加敌威胁,选最安全前进点(当前 `_cautious` 只判定不优化落点)
- [ ] **更多兵种**:逼近原版数值表(约 60 种)+ 特殊能力(无限反击 / 死亡凝视 / 吸血 / 自愈)
- [ ] **阵型 / 编队**(`ai/formation.py`):开局布阵 + 战中保持保护关系
- [ ] **`strategy.py`** —— 仅在确认原版有对应行为时才做。原版战斗 AI 几乎不存在独立策略层,
      不要为凑模块而发明原版没有的系统(否则违背"以原版为基准"的初衷)

## 🔵 P4 — 对抗性调优(依赖 P0 闭环)

- [ ] **与原版 AI 行为对照**:同一战场快照,对比本项目 planner 与原版决策是否一致
- [ ] **scoring 权重调参**:在批量对战框架上做参数搜索
- [ ] (可选)引入 C++ 模拟器 / MCTS 作对手,互为陪练验证强度

---

## 总体路线(一句话)

先 **P0 建验证闭环 → P1 三处小修拉高保真 → P2 补法术/撤退两大块**;
做到这里即是"行为可与原版对照、覆盖率 ~85%"的高保真复刻,P3/P4 为锦上添花。

---

## 关键原版参考(速查)

| 本项目模块 | fheroes2 源码 | 功能 |
|---|---|---|
| `ai/planner.py` | `ai_battle.cpp:689` (`planUnitTurn`) | 单位回合主干 |
| `ai/evaluation.py` | `ai_battle.cpp:949` (`analyzeBattleState`) | 局面分析 / 战术姿态 |
| `ai/scoring.py` | `battle_troop.cpp:1007` (`evaluateThreatForUnit`) | 威胁打分 |
| `ai/spells.py`(待建) | `ai_battle_spell.cpp:71` (`selectBestSpell`) | 法术 AI |
| `ai/retreat.py`(待建) | `ai_battle.cpp:703-870` | 撤退 / 投降 |
| `engine/unit.py` | `battle_troop.cpp` / `monster.cpp:146` | 单位 / 战力公式 |
| `engine/hex_grid.py` | `battle_board.cpp` | 六角格引擎 |
