# 里程碑 — 战斗 AI 复刻

> 把 [`TODO.md`](TODO.md) 的 P0–P4 重组为可发版、可验收的里程碑。
> 每个里程碑都有明确的**退出标准(Exit Criteria)**:满足即可打 tag 发版,不满足不进下一个。
> 保真度百分比以「覆盖原版战斗 AI ~3000 行的决策行为」为分母,为粗略估计。

> 注:`v0.2.0` 是已发布的可玩 demo(GUI + 当前 AI),里程碑版本从 `v0.3.0` 起。

| 里程碑 | 主题 | 对应 TODO | 目标保真度 | 状态 |
|---|---|---|---|---|
| **M1** `v0.3` | 验证闭环可用 | P0 | ~45%(不变,但**可量化**) | ✅ 完成 |
| **M2** `v0.4` | 保真度修正 | P1 | ~55% | ✅ 完成 |
| **M3** `v0.5` | 法术系统 | P2(法术) | ~70% | ✅ 完成 |
| **M4** `v0.6` | 撤退 + 完整回合行为 | P2(其余) | ~85% | ✅ 完成 |
| **M5** `v1.0` | 特殊能力 + 谨慎走位 | P3(核心) | ~90% | ✅ 完成 |
| **M5b** | 宽体单位(2 格占位) | P3 | ~91% | ✅ 完成 |
| **M6a** | 兵种扩充(原版精确数值;先 Knight+Barbarian ~20) | P3 | ~92% | ✅ 完成 |
| **M6b** | 攻城系统(完整) | P4 | ~94% | 📋 待办 |
| **M6c** | 验证 / 调参 / 原版对照 | P4 | ~95% | 📋 待办 |

---

## M1 — 验证闭环可用 `v0.3`

> **没有验证就没有"保真"**。本里程碑不增强 AI,只让"对齐程度"第一次变得可测量。

任务:
- [x] 解耦 `hex_grid.py` / engine 与 pygame — 纯几何留 engine,像素+绘图入 `ui/hex_renderer.py`
- [x] `scripts/arena.py`:批量 N 局、镜像配队、轮换先手、Wilson 置信区间
- [x] `tests/test_planner.py`:固定快照断言关键动作 + 无-pygame 守门
- [x] `tests/test_combat.py`:伤害公式(攻防差 ±、3.0× 上限、0.3 下限、射手近战 0.5)
- [x] 伤害拆分 `expected_damage`(AI/测试,确定) vs `roll_damage`(执行,随机)
- [x] `.github/workflows/test.yml`:push/PR 跑 pytest + arena smoke

**退出标准:**
- `pytest` 全绿,engine + ai 可在无 pygame 环境下导入并测试
- `arena.py --games 500 --mirror` 能跑完并输出带置信区间的胜率
- 镜像对战胜率落在 **40–60%**(无重大站边偏差)

> 注:原定「50%±5%」过紧。odd-r 棋盘对列反射**非等距**(实测破坏 1820 对格子距离),
> 且先动方吃反击是结构性劣势,故完美 AI 在镜像下也到不了精确 50%。
> M1 中 arena 已暴露并修掉一个真实方向 bug:`should_defend` 硬编码 team0 视角
> (`unit.col >= cols//2`),导致 team1 永不防守 → 镜像 9.8%;修为按 team 判定后回到 ~44–54%。

---

## M2 — 保真度修正 `v0.4`

> 三处**改现有代码**的小改动,直接拉高保真度。改完用 M1 的 arena 验证未引入回归。

任务:
- [x] 战力公式对齐原版量纲(`engine/unit.py`)— `getMonsterBaseStrength` 算法(sqrt(dmg·hp)·special)+ `(1+0.1atk+0.05def)×base×count`
- [x] `threat()` 升级为伤害+距离衰减(`ai/scoring.py`)— `expected_damage / distMod`;双击/能力/符号修正留 M3/M4
- [x] `isLimitOfTurnsExceeded`:50 回合无死亡 → 进攻方(可配置,默认 team0)撤退,替换 200 回合硬上限
- [x] **(闭环发现)交替出手 turn order** — 对齐原版 `GetCurrentUnit` 归并:同速单位 A,B,A,B 交替而非整队先动

**退出标准:**
- [x] `pytest` 全绿(34 个;新增 strength/threat/stalemate/turn-order 用例)
- [x] `strength` 公式 = `(1+0.1·atk+0.05·def)×baseStrength×count`,有单测覆盖
- [x] 三预设镜像仍落 40–60%(40.6 / 49.6 / 55.0,均 PASS)
- [x] arena 不再撞 200 回合兜底(Ended early 0%,平均 9–25 回合)

> **闭环再次兑现价值**:M2 更锐利的 damage-based 集火暴露出 demo 的 turn order 不保真——
> 同速时「整队先动」导致后手方集火反击的巨大优势(镜像 35/70/79%)。原版 `GetCurrentUnit`
> 是两队速度队列归并、同速交替出手。改为交替后镜像回到 40–60%。这是 turn-order 保真缺口,
> 非 M2 三项之一,但因污染镜像测量而一并修掉(同 M1 修 should_defend 的处理)。

---

## M3 — 法术系统 `v0.5`

> 原版最大单块(958 行),也是观感差距最明显处。先做这块。

任务:
- [x] `engine/spells.py` + `engine/hero.py`:6 法术(伤害/增益/减益),buff/debuff 为**有时限临时状态**(Effect 逐回合 tick 失效);Hero{power, spell_points, spellbook},每回合≤1 法
- [x] `engine/unit.py`:`effects[]` + 有效 `speed`(base+Σdelta) + `damage_factor`(Bless/Curse) + `tick_effects`
- [x] `engine/battle_state.py`:heroes + `_cast` 执行 + damage_factor 入伤害 + start_round tick/reset;`CastAction`
- [x] `ai/spells.py`:`select_best_spell` 阈值 `myStr²/enemyStr×0.04`(射手×0.5/低法力×2)、法力开方折价 `value/√(cost/3)`、伤害启发式(击杀奖励/掉血%)、Haste/Slow/Bless/Curse ratio(对齐 `ai_battle_spell.cpp:71`)
- [x] 整合:`planner.maybe_cast_spell`(单位行动前)、headless/GUI 接入、config/presets 支持 heroes、arena `--hero0/--hero1`

**退出标准:**
- [x] 覆盖 Magic Arrow / Lightning Bolt / Haste / Slow / Bless / Curse
- [x] buff/debuff 到期自动失效(单测验证),不永久改属性
- [x] arena 带法术 vs 无法术胜率显著差异(team0 +hero ≈100%);镜像双英雄仍 40–60%(46.8%)
- [x] 全部 pytest 绿(52 个,新增 test_spells/test_spell_ai 18 个);GUI 无头含施法跑帧通过

> 注:Bless/Curse 原版为"打最大/最小伤害",demo 单伤害值→近似 ×1.2 / ×0.8。
> 召唤/复活/驱散/群体/AOE/特殊能力(死亡凝视等)留待后续。arena 的 per-config 英雄未接入
> (用 `--hero0/--hero1` 旗标),配置文件英雄走 `run_battle`/GUI 路径。

---

## M4 — 撤退 + 完整回合行为 `v0.6`

> 补齐回合内剩余的原版行为,达到"一局完整战斗的每一步都有原版对应逻辑"。

任务:
- [x] `ai/retreat.py`:续战条件 `myStr×ratio ≥ enemyStr`(难度系数 Easy16.7/Normal13.3/Hard11.8/Imp10)+ 告别伤害法术(`select_best_spell(retreating=True)`)。**仅 Retreat**(Surrender 依赖 demo 没有的王国/金币/宝物/主属性)
- [x] `berserkTurn`:轻量 `Berserk` 效果 + `planner._berserk` 攻击/移向最近单位(不分敌我),射手射最近
- [x] 士气/运气引擎层(军队级,默认 0=关):运气好×2/坏×0.5(在 `roll_damage`)、士气好→额外行动/坏→跳过(循环);**`expected_damage` 与 planner 决策完全不含士气运气**(单测断言)

**退出标准:**
- [x] 劣势方撤退而非战至全灭(arena 可观测:Impossible 难度悬殊配队撤退率 ~18%);撤退前放告别法术
- [x] 狂暴/士气/运气均有单测;planner 决策代码中无士气运气计算(单测验证一致)
- [x] 全部 pytest 绿(70 个,新增 test_retreat/test_berserk/test_morale_luck);GUI 无头跑帧含撤退;镜像 Balanced 1000 局 42.1%(40–60 PASS)

> 注:撤退续战比很大(~13×),正常对战极少初始触发,多在单位被打残、strength 跌破阈值后中途撤退。
> Retreat 在单位行动前(Step 2,先于施法 Step 3);headless/run_battle 共用 `_take_unit_turn`,GUI 动画路径仅运气生效、不做士气额外/跳过(默认关)。

---

## M5 — 特殊能力 + 谨慎走位 `v1.0` ✅

> 收尾:补特殊能力与新兵种、把谨慎走位从"判定"升级为"优化落点"。
> 范围按确认聚焦核心两项;宽体单位/攻城**显式留作 M5b/M6**。

任务:
- [x] 特殊能力:无限反击/吸血(hp_drain)/死亡凝视(death_gaze,近似 enemy_halving)/自愈(self_heal);战斗钩子在 `battle_state` + base_strength/threat 倍率接上 M2/M3 预留位
- [x] 新兵种:Griffin 补无限反击(经典特性)+ Vampire(吸血,飞)/Troll(自愈)/Medusa(死亡凝视)
- [x] `findOptimalPositionForSubsequentAttack` 谨慎走位:`_safest_step_on_path` 沿路径选威胁最低且仍前进的落点(`cautious` 时启用)
- [x] 验证:新功能单测 + 现有回归(确认范围,不做 arena 调参/原版对照)
- [ ] **(已细化为下方 M5b→M6c)** 宽体单位 / 攻城 / 兵种扩充 / arena 调参 / 原版决策对照;~~阵型/MCTS~~(已剔除:属 AI 算法增强而非规则复刻,留待 DL 阶段)

**退出标准:**
- [x] 4 个能力均生效且有单测;base_strength/threat 能力倍率接上
- [x] 谨慎走位在慢速敌人场景选更安全落点(单测:落点威胁 141→0)
- [x] 全部 pytest 绿(82 个,新增 test_abilities/test_cautious 12 个);镜像仍 40–60(Balanced 40.6 / Flyer 55.0);GUI 含新兵种跑帧正常

> 注:death_gaze 近似为额外斩杀 `max(1,count//10)`;hp_drain **不复活**(只补残血),弱于原版可复活版;self_heal 每回合补一名存活单位的血(不复活)。Bless/Curse 仍 M3 的 ×1.2/×0.8 近似。

---

# M5b/M6 — 规则复刻收尾(冻结规则,为 DL 铺路)

> M5b→M6c 是 v1.0 之后的规则补全。目标是把游戏规则**做到接近原版并冻结**,
> 之后才进 R2+ 训练脚手架(规则不冻结,观测/动作编码会反复返工)。
> **范围决策(2026-06-09 确认)**:宽体单位完整 2 格占位、攻城完整版、兵种用原版精确数值、
> 源码级审计对照;**剔除 MCTS/阵型**(属 AI 算法增强,非规则复刻,留待 DL 阶段)。
>
> **依赖顺序**:`M5b 宽体(地基) → M6a 兵种 → M6b 攻城 → M6c 验证`。
> 原版近 1/3 兵种本身是宽体,故宽体必须先于兵种扩充。
>
> **安全网(每个子里程碑通用)**:老的单格 / 野战对战指纹保持逐字不变(零回归),
> 新行为由新单测覆盖。唯一例外是 M6a 切换原版精确数值时**一次性、有意地**重建指纹基线。
>
> **源码对照基准**:本地 `/home/awaken/projects/AI/fheroes2/src/fheroes2/` 有完整 C++ 源码。

## M5b — 宽体单位(结构性地基)✅

> 原版规则:宽体单位占**同一行相邻两格**(head + tail),head 永远朝敌、tail 在身后;
> 移动需两格都放得下;近战邻接 head 与 tail 都算。**简化决策**:朝向固定由队伍决定
> (team0 head 在右/tail=col-1,team1 head 在左/tail=col+1),不实现原版倒退翻转(AI 几乎只向前)。
> 对照源码:`battle/battle_cell.cpp`(`Position::Set`)、`battle/battle_troop.cpp`(GetHead/TailIndex,
> `_isReflected = headIdx < tailIdx`)、`battle/battle_board.cpp`、`battle/battle_pathfinding.cpp`。

任务:
- [x] `engine/unit.py`:加 `is_wide`(默认 False)、`tail_cell`(由 head+team 朝向推导)、`occupied_cells()`(单格返回 `{pos}`,与旧逻辑逐字等价)
- [x] `engine/hex_grid.py`:`reachable`/`find_path`/`nearest_cell_next_to` 加 `tail_dir` 参数,`_tail_ok` 校验尾格在界内且空闲;单格路径逐字不变
- [x] `engine/battle_state.py`:`occupied()`/`unit_at()` 改 footprint 并集(单格等价)
- [x] `ai/classic/planner.py`:加 `_tail_dir`/`_dist`/`_pos_dist`/`_attack_cells` 几何 helper(均对单格等价),全调用点 footprint 化 + `tail_dir` 串入寻路
- [x] `ui/renderer.py`:`draw_unit` 宽体先画连到 tail 格的身体(head 之下层,动画跟随)
- [x] `config/units.py`+`config/presets.py`:新增宽体兵种 `Champion` + `Wide Clash` 预设(老预设不动)
- [x] `scripts/fingerprint.py`:提交可复现指纹脚本,基线钉死在 3 个单格预设(与 PRESETS 增删解耦)

**退出标准:**
- [x] 宽体单位移动 / 攻击 / 反击 / 渲染正确,有单测(`tests/test_wide.py` 12 个)
- [x] 纯单格预设战斗指纹与 M5b 前**逐字一致** = `49b740ae1e7e90d3`(零回归安全网,贯穿 6 步不变)
- [x] 全部 pytest 绿(94 个);GUI 宽体局渲染 202 帧无错正常分胜负;arena 镜像 Balanced 仍 PASS(40–60%)

> 注:朝向固定为近似(原版 `UpdateDirection` 倒退翻转未实现);宽体 ENEMY 的 `pos_value`/`threat`
> 评分仍按 head 近似(scoring.py 未动以守指纹),M6a 接原版数值后再评估是否细化。

## M6a — 兵种扩充(原版精确数值)✅

> 把兵种数值切到 **fheroes2 原版精确表**(现有简化数值作废),此刻**重建指纹基线**
> (有意的一次性行为变更,M6a 唯一允许破 `49b740ae1e7e90d3` 的点)。
> **范围决策(2026-06-09)**:先做 **Knight + Barbarian 两阵营(~20)** 打通管线,
> 其余 4 阵营 + 中立作机械式后续扩充。数量用**预设可指定 + 默认 `grown`**;
> 能力**易做的实现、其余进档案表,全部进 strength 公式**。
> 对照源码:`monster/monster_info.cpp`(`battleStats`/`generalStats` 数据表 + 行 391+ 的能力注入、
> `getMonsterBaseStrength`)、`kingdom/speed.h`(Speed 枚举=整数,AVERAGE=4)。

**名册**:Knight = Peasant/Archer/Ranger/Pikeman/Veteran Pikeman/Swordsman/Master Swordsman/
Cavalry/Champion/Paladin/Crusader;Barbarian = Goblin/Orc/Orc Chief/Wolf/Ogre/Ogre Lord/
Troll/War Troll/Cyclops。

**能力覆盖表**(本两阵营用到的):

| 能力 | 兵种 | 本轮 |
|---|---|---|
| `DOUBLE_HEX_SIZE`(宽体) | Cavalry, Champion, Wolf | ✅ 已实现(M5b) |
| `HP_REGENERATION` | Troll, War Troll | ✅ 已实现(self_heal) |
| `DOUBLE_SHOOTING` | Ranger | ✅ 做(射手每回合射 2 次) |
| `DOUBLE_MELEE_ATTACK` | Paladin, Crusader, Wolf | ✅ 做(近战连击 2 次,反击后触发) |
| `TWO_CELL_MELEE_ATTACK` | Cyclops | ✅ 做(命中目标 + 其身后一格,反击前触发) |
| `DOUBLE_DAMAGE_TO_UNDEAD` | Crusader | ✅ 做(×1.15 进公式;本两阵营无亡灵) |
| `SPELL_CASTER`(20% 麻痹) | Cyclops | 📄 档案(需麻痹状态,暂缺战斗钩子) |
| `IMMUNE_TO_CERTAIN_SPELL`(诅咒) | Crusader | 📄 档案(法术免疫判定,暂缺) |

任务:
- [x] **伤害模型 min/max**:`Unit.damage`→`damage_min/damage_max`;`expected_damage`=`count·(min+max)/2·mult`,`roll_damage` 在区间内取(**移除非忠实的 ±15%**),`_compute_base_strength` 用均值
- [x] **`config/units.py` 重构**:精确数据表 + `race`/`level`/`grown` 字段,标 `is_wide`/`shots`/`abilities`
- [x] **能力**:补全 strength 公式能力项(no_enemy_retaliation×1.4、double_shooting×2、double_melee×1.75、two_cell×1.2、double_damage_to_undead×1.15);战斗钩子做 double_shooting/double_melee/two_cell;其余进档案
- [x] **数量规则**:扩展预设格式支持可选 `count`,未写默认 = `grown`;`from_type` 支持 count 覆写
- [x] **预设迁移**:老预设迁到新兵种 + 加 Knight vs Barbarian / Clash of Titans 预设;Balanced 换 Ogre Lord(非宽体镜像公平)
- [x] **测试**:批量创建全兵种、strength 按 cost 单调性 sanity、宽体/射手标记、新能力 combat hook (20 新测试)
- [x] **重建指纹基线**:`1f54c421b0f7f078` 替换 `49b740ae`,更新脚本历史注释

**退出标准:**
- [x] Knight+Barbarian 20 兵种可创建且通过 strength sanity;能力覆盖表入档
- [x] 新指纹基线已记录(`1f54c421b0f7f078`),后续以它为零回归基准
- [x] 全部 pytest 绿(115 个);arena 镜像 4 预设均 PASS(Balanced 50.0/Archer Defense 51.7/Flyer Threat 45.7/Knight vs Barbarian 48.7)

> 注:移除 ±15% 改 min/max + 老预设迁精确数值,会改变 arena 镜像胜率,需重新确认(M6a 预期内重校准)。
> 其余 4 阵营 + 中立(Sorceress/Warlock/Wizard/Necromancer + Griffin/Phoenix/Hydra/各色龙等)
> 待本轮管线打通后机械式补入,届时再评估新能力(DRAGON/AREA_SHOT/UNDEAD/ELEMENTAL 等)。

## M6b — 攻城系统(完整版)

> fheroes2 最大的单一子系统。对照源码:`battle/battle_arena.cpp`、`battle/battle_catapult.cpp`、
> `battle/battle_tower.cpp`、`battle/battle_bridge.cpp`、`battle/battle_board.cpp`(墙/河格位)、`castle/castle.cpp`。

任务:
- [ ] `engine/castle.py` 数据层:4 段城墙(多状态 hp:完好/破损/摧毁)、城门吊桥(开/关/毁)、护城河格、箭塔(≤3,射手伪单位)、投石车
- [ ] 几何 + 规则:城墙阻挡移动与射击 LOS;护城河停止移动 + 防御惩罚;吊桥开关控制入城;守军加成
- [ ] 回合机制:箭塔自动射击、投石车每回合自动砸墙(攻方)、隔墙射击惩罚
- [ ] `ai/classic` 攻城决策:攻方优先破墙 / 破桥,守方据墙,箭塔选目标,投石车选目标启发式(对齐原版 Arena AI)
- [ ] `ui` + `config/presets.py`:渲染城堡 / 墙 / 塔 / 河 + 一个攻城预设

**退出标准:**
- [ ] 城墙 / 护城河 / 箭塔 / 投石 / 吊桥 / 守军加成各部件生效且有单测
- [ ] 攻 / 守 AI 在攻城场景表现合理
- [ ] 野战指纹不受影响(攻城为独立场景);GUI 攻城跑帧正常

## M6c — 验证与调参(打 ~95% tag)

> 对照源码:`ai/` 下的战斗 AI(`AIBattle`/`AIToBattle` 相关)。审计方式为**源码级逐条比对**,
> 产出「行为差异清单」,不做原版运行时注入对照(工程上不现实)。

任务:
- [ ] `scripts/arena.py` 扩展:支持新兵种 + 攻城场景
- [ ] arena 权重调参一遍,向原版行为靠拢
- [ ] 源码级审计对照:逐条比对原版 AI 源码,产出 `docs/` 行为差异清单
- [ ] 文档更新,保真度 ~95%,打 tag

**退出标准:**
- [ ] arena 支持新兵种 + 攻城并跑完
- [ ] 行为差异清单入档(逐条标注已对齐 / 近似 / 暂缺)
- [ ] 全部 pytest 绿;镜像仍 40–60%;打 tag

---

# 路线图 — AI 可插拔化(面向未来深度学习)

> 目标:把整套 AI 改写成**可插拔**形式,未来能用深度学习训练的 `DeepAI` 替换原版 AI,
> 同时**原版规则 AI 完整保留**,两者可同台对战(天然互为基准)。
> 训练本身不急;先铺骨架,等游戏规则冻结后再做训练相关的重武器。

## 设计原则

1. **零行为变化** — 原版 AI 逻辑一行不改,只搬位置 + 加一层接口;重构前后所有测试逐字一致。
2. **依赖方向不变** — 继续 `ai → engine` 单向,引擎/UI 不知道有几种 AI。
3. **接口稳定、实现可换** — 调用方只认抽象接口 `AIPlayer`,不认具体类。
4. **本阶段不引入 DL 依赖** — 不碰 numpy / gym / torch,纯软件工程重构,与继续写规则零冲突。

## 解耦现状(2026-06-09 审查结论)

- ✅ 分层健康:`config → engine → ai → ui`,`engine` 零反向依赖,`ai` 不依赖 config/ui/pygame。
- ✅ AI 输出是干净的 `Action` 对象,引擎用 `battle.execute()` 消费。
- ⚠️ 替换 DL 的 4 个障碍:(1) 无抽象接口+工厂,8 处硬编码 `BattleAI()`;(3) 无观测编码层;
  (4) 无形式化动作空间+合法掩码+训练环境。障碍 2(三段式接口)向前兼容,DL 可内部委托统一策略。

## R1 — 可插拔骨架(障碍 1)✅ 完成

> 本阶段只解决障碍 1。规则可继续随意改,接口不变。

任务:
- [x] `ai/base.py`:`AIPlayer` 抽象基类,签名照搬现有三方法(`check_retreat` / `maybe_cast_spell` / `decide`)
- [x] `git mv` 原版 6 文件(planner/evaluation/scoring/spells/retreat/strategy)进 `ai/classic/`,内部相对 import 整组搬动后仍成立
- [x] `BattleAI` 继承 `AIPlayer` 并重命名 `ClassicAI`(`ai/classic/__init__.py` 保留 `BattleAI` 别名防旧引用)
- [x] `ai/factory.py`:`create_ai(kind)` + 注册表 + `register_ai` / `available_ais`,注册 `"classic" → ClassicAI`
- [x] 调用点(headless ×2 / GUI)改走 `create_ai("classic")`;8 个测试改 import 路径到 `ai.classic.*`
- [x] 预留空 `ai/deep/` 占位目录(含未来 DeepAI 注册说明)

**退出标准:**
- [x] `pytest` 全绿(82 个),结果与重构前一致
- [x] 同 seed 的 headless 战斗指纹重构前后逐字相同(30 局 `HASH 545c41fcb8481a63`,零行为变化安全网)
- [x] 切换 AI 只需改 `create_ai("...")` 的字符串;`create_ai("deep")` 清晰报错列出可用项

## R2+ — 训练脚手架(障碍 3/4,规则冻结后做,保持框架无关)

> 以下依赖「游戏规则已基本冻结」,过早做会随规则反复返工,故**显式留到规则稳定后**。

- [ ] **R2 观测编码** `ai/observation.py`:`BattleState` → 定长数值向量(numpy,框架无关)
- [ ] **R3 动作空间** `ai/action_space.py`:动作编号 + 合法性掩码 + 编号↔Action 互转
- [ ] **R4 环境适配** `ai/env.py`:通用 `reset/step/reward` 三件套;是否套 Gymnasium 适配层到时再定(适配层单独隔离)
- [ ] **R5 DeepAI** `ai/deep/`:实现 `DeepAI(AIPlayer)` 注册进工厂,`create_ai("deep")` 与原版同台对战
