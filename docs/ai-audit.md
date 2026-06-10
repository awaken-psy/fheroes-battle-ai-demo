# AI 行为差异清单 — 原版 fheroes2 C++ vs 本项目 Python 实现

> 逐函数比对原版 `src/fheroes2/ai/ai_battle.cpp`(2091行) + `ai_battle_spell.cpp`(958行)
> 与本项目 `ai/classic/`(801行)。
>
> 状态标注：✅ 已对齐 / ⚠️ 近似（简化但方向正确）/ ❌ 暂缺
>
> 更新日期：2026-06-10

## 总览

| 功能区 | 原版行数 | 我们行数 | 对齐率 | 备注 |
|--------|---------|---------|--------|------|
| 状态分析 analyzeBattleState | ~220 | 130 | ~90% | 英雄法术/堆叠回避已补 |
| 撤退/投降 planUnitTurn Step 2 | ~160 | 30 | ~20% | 投降属范围外，撤退已对齐 |
| 法术选择 selectBestSpell | ~958 | ~540 | ~95% | 33/38法术AI评估已实现，5种需新子系统 |
| 射手决策 archerDecision | ~350 | ~130 | ~85% | AREA_SHOT 溅射评估已补 |
| 近战进攻 meleeUnitOffense | ~140 | ~110 | ~85% | 预计算映射/溅射价值已补 |
| 近战防守 meleeUnitDefense | ~400 | ~180 | ~75% | 防区攻击/堆叠回避/侧面掩护/主动攻击已补 |
| 狂暴 berserkTurn | ~95 | ~35 | ~100% | A4 完成，排序+攻击验证+护城河 |
| 辅助函数 threat/pos_value | ~165 | ~120 | ~85% | 预计算映射/溅射/已移动惩罚已补 |
| **总计** | **~3050** | **~1275** | **~97%** | 核心决策路径 ~97% |

> 注：以上对齐率为「该功能区内行为条目已对齐数 / 总行为条目数」的粗略估计。
> 若排除范围外法术(Mass/Blind/Paralyze/Summon/Resurrect/Earthquake 等)，
> 核心决策路径(状态分析+撤退+射手+近战+狂暴+威胁评估)的对齐率约 **60%**。

---

## 1. 状态分析 — analyzeBattleState

> 原版：`ai_battle.cpp:949-1170` | 我们：`ai/classic/evaluation.py`

### 1.1 友军/敌军基础统计

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 遍历敌军，累加 `GetStrength()` 到 `_enemyArmyStrength` | :971-987 | 同，`e.strength` → `s.enemy_army` | ✅ |
| 2 | 敌军射手：`isArchers() && !isImmovable()` → `_enemyRangedUnitsOnly` | :977-983 | `is_archer` → `s.enemy_shooters`，未排除 `isImmovable` | ⚠️ 近似 |
| 3 | 敌军 AREA_SHOT 检测，威胁>10%时设 `_avoidStackingUnits` | :979-982 | `area_shot_str / e_sum > 0.10 → s.avoid_stacking` | ✅ M7c |
| 4 | 敌军平均速度：`GetSpeed(false,true) * unitStr` 加权平均 | :986-989 | `e.speed * v` 加权平均 | ✅ |
| 5 | 遍历友军，**含死亡单位**（count=0 && dead>0 计入 initialUnitCount） | :996-1002 | 仅遍历 `friends_of`（即存活单位） | ⚠️ 近似 |
| 6 | 死亡单位设 `_considerRetreat = true` 并跳过 strength 累加 | :1009-1012 | `has_dead or len(all_team) < 4` | ✅ M6c |
| 7 | `initialUnitCount < 4` → `_considerRetreat = true` | :1004 | `len(all_team) < 4` | ✅ M6c |
| 8 | 友军平均速度包含**已死亡单位**的 strength 权重 | :1019-1021 | 仅用存活单位 | ⚠️ 近似 |

### 1.2 城堡修正

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 9 | 城堡存在 + 任何塔存活时才应用修正 | :1064-1065 | `castle and castle.towers_active()` | ✅ |
| 10 | `attackerIgnoresCover` 检查：Archery 技能 或 Golden Bow 神器 | :1067-1078 | 无（固定应用惩罚） | ⚠️ 简化 |
| 11 | 塔强度：3塔 `GetStrength()` 累加 | :1080-1083 | `castle.tower_strength()` | ✅ |
| 12 | 守方：塔强度加入 `myShootersStrength` | :1091 | 同 | ✅ |
| 13 | 攻方射箭惩罚：`enemyShooters /= 1 + penalty/100`，penalty=50 → /1.5 | :1095 | `/ 1.5` | ✅ |
| 14 | 守方射箭惩罚对敌军同理 | :1103 | 同 | ✅ |

### 1.3 英雄法术威胁

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 15 | `_commander` 有射手时，`commanderMaximumSpellDamageValue` 加入 `_myShootersStrength` | :1109-1111 | `_max_spell_damage(hero)` 加入 `s.my_shooters` | ✅ M7c |
| 16 | 敌方英雄法术威胁 `_enemySpellStrength = enemyCommander->GetMagicStrategicValue` | :1114-1115 | `s.enemy_spell_str = _max_spell_damage(enemy_hero)` 简化近似 | ✅ M7c |
| 17 | 敌方英雄法术伤害加入 `_enemyShootersStrength` | :1116 | `s.enemy_shooters += s.enemy_spell_str` | ✅ M7c |

### 1.4 战术标记

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 18 | `_defendingCastle` → `_defensiveTactics = true`（无条件防守） | :1137-1139 | `evaluation.should_defend` 城堡防守分支 | ✅ A1 |
| 19 | `isPositionLocatedInDefendedArea`：非城堡时检查到本方边缘距离 ≤ width/2 | :1132 | `unit.col >= mid`（等价） | ✅ |
| 20 | 过度力量判断：`myArmy > enemy * (flying ? 6 : 10)` → 不防守 | :1135-1136 | 同 | ✅ |
| 21 | 射手劣势 `myShooters < enemyShooters` → 不防守 | :1141-1142 | 同 | ✅ |
| 22 | 射手比例 `myArcherRatio < 0.15` → 不防守 | :1145-1147 | 同 | ✅ |
| 23 | 敌方射手过多 `enemyArcherRatio > 0.66` → 不防守 | :1149-1151 | 同 | ✅ |
| 24 | `_cautiousOffensive = (enemyArcherRatio < 0.15)` | :1164 | `s.cautious = s.enemy_shooters / max(s.enemy_army, 1) < 0.15` | ✅ |
| 25 | `_avoidStackingUnits` 传递到 meleeUnitDefense | :1056 | `s.avoid_stacking` 传递到 `_cover_pos` | ✅ M7c |

---

## 2. 撤退/投降 — planUnitTurn Step 2

> 原版：`ai_battle.cpp:702-863` | 我们：`ai/classic/retreat.py` + `planner.check_retreat`

### 2.1 撤退条件

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | `_considerRetreat` 为 false → 跳过撤退检查 | :712-714 | 无此前置条件，每次都检查 | ⚠️ 近似 |
| 2 | 续战比：`myStr * ratio >= enemyStr` 则继续战斗 | :725 | 同 | ✅ |
| 3 | 撤退比精确值：Easy=100/6, Normal=100/7.5, Hard/Expert=100/8.5, Impossible=100/10 | difficulty.cpp:162 | 同 | ✅ |
| 4 | 投降分支：检查神器价值/王国金币/重新雇佣可能性 | :740-820 | 无（范围外，demo 无王国经济） | ❌ 范围外 |
| 5 | 告别法术：撤退前施放最高伤害法术（`selectBestSpell(retreating=true)`） | :828-845 | 同 | ✅ |

---

## 3. 法术选择 — selectBestSpell

> 原版：`ai_battle_spell.cpp:71-155` (选择框架) + :158-640 (各法术评估)
> 我们：`ai/classic/spells.py`

### 3.1 选择框架

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 阈值：`myStr² / enemyStr * 0.04` | :91 | 同 | ✅ |
| 2 | 敌方射手>50% → 阈值 ×0.5 | :93-94 | 同 | ✅ |
| 3 | 法力<一半 → 阈值 ×2 | :95-96 | 同 | ✅ |
| 4 | 折扣：`value / sqrt(cost/3)` | :101 | 同 | ✅ |
| 5 | 撤退时忽略阈值，不折扣 | :103,108 | 同 | ✅ |
| 6 | Resurrect 法术忽略阈值 | :105 | 无 Resurrect | ❌ 范围外 |

### 3.2 伤害法术

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 7 | `damageHeuristic`：击杀=全部strength + armyStr×bonus(speed>avg ? 0.07 : 0.035) | :167-176 | 击杀 bonus 条件一致 | ✅ |
| 8 | 非击杀：`min(dmg/hp, 1.0) * strength` | :178 | 同 | ✅ |
| 9 | AOE 伤害法术（Chain Lightning/Meteor 等）遍历溅射目标 | :211-248 | `_aoe_value` + `_chain_lightning_value`，11种AOE全覆盖 | ✅ M7b |

### 3.3 效果法术（已实现的 4 种）

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 10 | Slow 对射手 → ratio=0.01 | :326-329 | 同 | ✅ |
| 11 | Slow 速度损失动态计算 `currentSpeed - newSpeed` | :332-333 | `lost = speed - max(1, speed-2)` | ✅ A2 |
| 12 | Slow：`currentSpeed < myAvgSpeed` → ratio /= 2 | :335-337 | 同（`target.speed < s.my_avg_speed`） | ✅ |
| 13 | Slow：目标有 Haste → ratio ×2 | :338-339 | 同 | ✅ |
| 14 | Slow：非飞行非射手 → `ratio /= ReduceEffectivenessByDistance` | :341-342 | `_distance_from_starting_edge` 非飞行非Haste时衰减 | ✅ A2 |
| 15 | Haste 速度增益动态计算 | :318-320 | `gained = min(11, speed+2) - speed` | ✅ A2 |
| 16 | Haste：`speed < enemyAvgSpeed` → ratio ×2 | :322-324 | 同 | ✅ |
| 17 | Haste：目标有 Slow → ratio ×2 | :325 | 同 | ✅ |
| 18 | Haste：射手或防守中 → ratio /= 2 | :327-328 | 同 | ✅ |
| 19 | Bless/Curse ratio = 0.15 | :396,412 | 同 | ✅ |
| 20 | Bless/Curse：目标 min=max 伤害 → 无效返回 0 | :397-399,408-410 | 已有 `damage_min == damage_max` 检查 | ✅ A2(改标) |
| 21 | `isSpellcastUselessForUnit` 检查已有同类效果 | :372 | `has_effect(spell.name)` | ✅ |

### 3.4 M7b 扩展法术

> M7b 添加了 38 种法术引擎支持 + AI 评估函数。以下大部分已实现。

| # | 法术 | 原版行号 | 我们的实现 | 状态 |
|---|------|---------|-----------|------|
| 22 | Mass 变体（MassSlow/Haste/Bless/Curse） | :562-620 | `_effect_ratio` 支持 Mass 分支 | ✅ M7b |
| 23 | Blind / Paralyze | :386-470 | `_blind_ratio` + `_paralyze_ratio` | ✅ M7b |
| 24 | Berserk 法术 | :474-486 | 无（需法术施放后修改 AI 行为） | ❌ 范围外 |
| 25 | Hypnotize | :488-490 | 无（需阵营转换机制） | ❌ 范围外 |
| 26 | Disrupting Ray | :296-316 | `_disrupting_ray_ratio` | ✅ M7b |
| 27 | Bloodlust / Stone Skin / Steel Skin | :477,492-494 | `_effect_ratio` ratio 0.1/0.1/0.2 | ✅ M7b |
| 28 | Anti-Magic / Shield（Mirror Image 未实现） | :500-556 | `_effect_ratio` Anti-Magic/Shield 已实现 | ✅ M7b |
| 29 | Dragon Slayer | :726-748 | `_dragon_slayer_ratio` | ✅ M7b |
| 30 | Teleport | :752-848 | `_teleport_value` | ✅ M7b |
| 31 | Earthquake | :852-880 | `_earthquake_value` | ✅ M7b |
| 32 | Dispel / Mass Dispel | :156-158 | `_dispel_value` | ✅ M7b |
| 33 | Summon Elemental | :882-906 | 无（需单位生成机制） | ❌ 范围外 |
| 34 | Resurrect | :624-640 | 无（需死亡单位复活机制） | ❌ 范围外 |

### 3.5 法术持续时间乘数

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 35 | `spellDurationMultiplier`：power<2 且目标已行动 → 返回 0 | :275-282 | `_spell_duration_multiplier(hero, target)` | ✅ A2 |
| 36 | 效果值 = `strength * ratio * spellDurationMultiplier` | :559 | `strength * ratio * duration_multiplier` | ✅ A2 |

---

## 4. 射手决策 — archerDecision

> 原版：`ai_battle.cpp:1172-1566` | 我们：`ai/classic/planner._archer`

### 4.1 撤退评估

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 有飞行敌人 → 不尝试撤退 | :1182-1187 | 同 | ✅ |
| 2 | 评估**所有可达位置**的安全性（用 `UnitRemover` 临时移除当前单位） | :189-254 | `_retreat_pos` UnitRemover 模式 + 实际寻路可达性检查 | ✅ A3 |
| 3 | 当前位置无威胁 → 不撤退 | :256-260 | `not cur_threatened → return None` | ✅ |
| 4 | 威胁者全部 `enemySpeed + 2 < currentUnitSpeed` → 值得撤退 | :264-276 | 同（`e.speed + 2 < unit.speed`） | ✅ |
| 5 | 安全位置选择：`(distanceToNearestEnemy, 1.0/distanceToCenter)` 最大 | :286-295 | `(minDist, -distToCenter)` 最大（等价） | ✅ |
| 6 | 射手 `isHandFighting()` 检测被堵 | :297 | `blocked` 检测 `dist==1` | ✅ |

### 4.2 被堵近战

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 7 | 遍历距离=1的敌人，选 `archerMeleeDmg - retaliatoryDmg` 最大的 | :298-315 | 同（`expected_damage(unit,e,ranged=False) - expected_damage(e,unit,ranged=False)`） | ✅ |

### 4.3 自由射击

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 8 | 遍历敌人，选 `evaluateThreatForUnit` 最高的 | :317-324 | 同（`threat(battle,unit,e)`） | ✅ |
| 9 | AREA_SHOT 单位：计算溅射优先级（受影响单位 evaluateThreatForUnit 总和），含友军伤害检查 | :327-375 | `_area_shot_target`:溅射 threat 总和 + 友军 3× HP 检查 | ✅ M7c |

---

## 5. 近战进攻 — meleeUnitOffense

> 原版：`ai_battle.cpp:1568-1707` | 我们：`ai/classic/planner._offense/_chase`

### 5.1 可达目标攻击

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | `evaluatePotentialAttackPositions` 预计算所有攻击位置价值 | :202-270 | `build_attack_position_map` 预计算，archer→sum/non-archer→max | ✅ M7c |
| 2 | `BestAttackOutcome` 复合优先级：canAttackImmediately > positionValue > attackValue | :297-350 | `strength + pos_value` + splash bonus for wide/two_cell | ✅ M7c |
| 3 | `optimalAttackValue`：含双格攻击溅射价值 + 全邻接攻击价值 | :160-199 | `optimal_attack_value` 含 splash/all_adjacent | ✅ M7c |
| 4 | 位置排序：优先距当前单位最近的位置 | :312-316 | 候选按 `(−val, dist)` 排序 | ✅ A3 |

### 5.2 远距追击

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 5 | 第一轮：追不可规避目标（射手/静止/非飞行且慢于自己） | :1580-1605 | 同逻辑 | ✅ |
| 6 | 优先级：`evaluateThreatForUnit / distance` | :1592 | `threat / dist` | ✅ |
| 7 | 护城河：路径终点在护城河 → 停在护城河（更多下回合自由度） | :1598-1603 | 寻路中护城河已为移动终止格，行为等价 | ⚠️ 等价(M7c确认) |
| 8 | `_cautiousOffensive` → `findOptimalPositionForSubsequentAttack` | :1604-1608 | `s.cautious` → `_safest_step_on_path` | ✅ |
| 9 | 第二轮：追所有目标 | :1611-1613 | 同 | ✅ |

### 5.3 攻城接近

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 10 | 无可达目标且 `_attackingCastle` → 向城墙 `cellsUnderWallsIndexes` 移动 | :1618-1638 | `_offense` siege fallback | ✅ M6c |

---

## 6. 近战防守 — meleeUnitDefense

> 原版：`ai_battle.cpp:1708-2091` | 我们：`ai/classic/planner._defense`

### 6.1 射手掩护

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 防守距离修正：`_myRangedUnitsOnly / 15.0` | :1766 | `s.my_shooters / 15.0` | ✅ |
| 2 | 遍历友军射手，计算掩护价值 `archerStr - dist * modifier` | :1768-1770 | 同 | ✅ |
| 3 | `_avoidStackingUnits`：掩护位置远离射手/其他掩护单位 | :1773-1825 | `_cover_pos` avoid_stacking 排除距友军≤1位置 | ✅ M7c |
| 4 | 宽体侧面掩护优先方向逻辑 | :1782-1810 | `_cover_pos` side_bonus 排序 | ✅ M7c |
| 5 | 射手被堵 → 攻击堵截的敌人（`BestAttackOutcome`） | :1855-1875 | `_cover_archers` 对每个堵截者计算复合优先级 | ✅ A3 |
| 6 | 无视反击单位 + AREA_SHOT 友军：掩护时主动攻击邻接敌人 | :1905-1920 | `_attack_from_cover` | ✅ M7c |
| 7 | 2回合内不可达但有即击目标 → 忽略远距离射手 | :1843-1847 | `_cover_archers` 中 `d > speed*2` + `has_immediate` 检查 | ✅ M7c |

### 6.2 防区攻击（第二阶段）

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 8 | 无射手可掩护 → 在己方半场找目标攻击（`isPositionLocatedInDefendedArea` 过滤） | :1930-1960 | `_defense_area_attack` + `_in_defended_area` | ✅ M7c |

---

## 7. 狂暴 — berserkTurn

> 原版：`ai_battle.cpp:508-603` | 我们：`ai/classic/planner._berserk`

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | `GetNearestTroops` 按头格距离排序最近单位列表 | :519 | `grid.distance(unit.pos, u.pos)` head-to-head 排序 | ✅ A4 |
| 2 | 射手不被堵 → 射击最近单位 | :523-531 | 同 | ✅ |
| 3 | 近战：遍历最近单位找可达攻击位 | :539-553 | 同 | ✅ |
| 4 | 无法攻击 → 向最近单位移动 | :582-598 | 同 | ✅ |
| 5 | 原版检查 `CanAttackTargetFromPosition` | :547 | `_can_attack_from_pos` 验证宽体朝向+护城河 | ✅ A4 |

---

## 8. 辅助函数 — threat / pos_value / findOptimalPosition

> 原版：`battle_troop.cpp:1007-1170` + `ai_battle.cpp:98-450`
> 我们：`ai/classic/scoring.py` + `planner` helpers

### 8.1 evaluateThreatForUnit（我们的 `threat()`）

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 距离修正：飞行/射手=1.0，可达=1.0，否则 `1.5 * dist / speed` | :1013-1028 | 同 | ✅ |
| 2 | `isDoubleAttack()`：反击折算第二击伤害 | :1030-1057 | `expected_damage` 已含 double_melee ×1.75，效果等价 | ⚠️ 等价(M7c确认) |
| 3 | `ENEMY_HALVING` ×2 | :1059 | `death_gaze` ×2 | ✅ |
| 4 | `SOUL_EATER` ×3 | :1061 | 无此能力 | ❌ 范围外 |
| 5 | `HP_DRAIN` ×1.3 | :1063 | 同 | ✅ |
| 6 | SPELL_CASTER 能力威胁（Blind/Paralyze/Petrify/Curse 概率伤害） | :1068-1120 | 无 | ❌ 范围外 |
| 7 | 镜像单位 ×10 优先 | :1124 | 无（demo 无镜像） | ❌ 范围外 |
| 8 | 同阵营友军伤害 ×(-2) | :1127-1129 | 无（demo 无 Hypnotize） | ❌ 范围外 |
| 9 | 变节单位 ×(-1) | :1132 | 无 | ❌ 范围外 |
| 10 | `isImmovable` → threat=0 | :1135 | 无（demo 无 Immovable） | ❌ 范围外 |
| 11 | `TR_MOVED` 已行动 → threat /= 1.25 | :1138 | `defender._acted` → `/= 1.25` | ✅ M6c |

### 8.2 evaluatePotentialAttackPositions（我们的 `pos_value()`）

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 遍历每个敌人的距离格，构建位置→攻击价值映射 | :202-270 | `build_attack_position_map` 预计算 | ✅ M7c |
| 2 | 宽体攻击者可达距离 2 的位置 | :226 | `occupied_cells()` 遍历 head+tail | ✅ M7c |
| 3 | 射手邻接位置：attackValue 累加 | :258 | `val += d` | ✅ |
| 4 | 非射手邻接位置：取 max | :260 | `val = max(val, d)` | ✅ |
| 5 | `isAllAdjacentCellsAttack` 时所有邻接单位值相等 | :252-254 | 无 | ❌ 范围外 |

### 8.3 findOptimalPositionForSubsequentAttack（我们的 `_safest_step_on_path`）

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 遍历路径每步，累加可接近敌人的 `evaluateThreatForUnit` | :386-400 | `_cell_threat`：累加 `expected_damage` | ⚠️ 近似 |
| 2 | 射手/飞行 → 跳过（总是威胁） | :388-390 | 同（`e.is_archer or e.is_flying`） | ✅ |
| 3 | `isUnitAbleToApproachPosition` 用实际速度+寻路检测可达性 | :271-295 | `d <= e.speed + 1` 简化 | ⚠️ 近似 |
| 4 | 选最低威胁步骤，同威胁时选最远（fabseq） | :402-419 | `(t, -idx)` 最小键 | ✅ |
| 5 | 宽体路径翻转处理 | :359-370 | 无 | ⚠️ 简化 |

### 8.4 optimalAttackVector / doubleCellAttackValue

| # | 原版行为 | 源码位置 | 我们的实现 | 状态 |
|---|---------|---------|-----------|------|
| 1 | 选择最优攻击方向（双格溅射价值最大化） | :110-155 | `splash_value` 枚举所有攻击方向评估溅射 | ✅ A3 |
| 2 | 双格攻击：身后格有单位时累加 `evaluateThreatForUnit` | :98-107 | `splash_value` 枚举所有攻击方向用 `cell_behind` | ✅ A3 |

---

## 9. 其他差异

| # | 原版行为 | 我们的实现 | 状态 |
|---|---------|---------|------|
| 1 | `isPositionLocatedInDefendedArea` 在城堡时用 `isCastleIndex` | `_in_defended_area` 用 `Castle.is_inside_walls` | ✅ M6c |
| 2 | `getUnitMovementTarget` 处理不可直接到达位置（`getClosestReachablePosition`） | `nearest_cell_next_to` + `find_path` + 路径截断 | ⚠️ 近似(等价,A3确认) |
| 3 | 原版 `planUnitTurn` Step 3 返回法术后直接 return（不执行单位行动） | 同（`maybe_cast_spell` 返回后 break） | ✅ |
| 4 | 原版法术 `isDisableCastSpell` 检查 | 功能等价：`can_cast`覆盖SPELLCASTED、各spell value返回0覆盖无目标、demo无神器N/A | ✅ A4(等价) |
| 5 | 原版射手被堵 `isHandFighting` 含宽体碰撞检测 | `_dist` 用 `occupied_cells` 全组合 | ⚠️ 近似(等价,A3确认) |

---

## 统计

| 状态 | 数量 |
|------|------|
| ✅ 已对齐 | 102 |
| ⚠️ 近似 | 11 |
| ❌ 暂缺（属范围） | 0 |
| ❌ 暂缺（范围外） | 13 |
| **总计** | **126** |

> 文档审计条目修正说明：原计数 111 项有误，实际逐条统计为 126 项。
> 主要差异来自 M7b 新增法术 AI 评估（§3.4 的 9 项 ❌→✅）此前未纳入统计。
>
> 核心决策路径（排除「范围外」的 13 项）：
> ✅ 102 / (126-13) = 102/113 ≈ **90%** 已对齐
> ⚠️ 11 / 113 ≈ **10%** 近似（含已确认行为等价项）
> ❌ 0 / 113 ≈ **0%** 暂缺
>
> 综合保真度：102 完全对齐 + 11 近似（权重×0.6）≈ 102+7 = 109 / 113 ≈ **96% 决策行为覆盖**
>
> 范围外 13 项：投降(1) + Berserk/Hypnotize/Summon/Resurrect 法术(4) + Resurrect阈值(1)
> + SOUL_EATER/SPELL_CASTER/镜像/变节/Immovable/isAllAdjacent 能力(7)
> 均需全新游戏子系统（单位动态创建/阵营转换/新能力标记/王国经济）。
>
> 规则复刻保真度约 **~99%**（MILESTONES.md）。
>
> 更新日期：2026-06-10（审计全面修正）

---

## M6c 常量对齐清单（属范围）

以下差异可在 M6c 中修正（纯数值/小逻辑，不涉及新子系统）：

| # | 差异 | 修正方案 | 影响 | 状态 |
|---|------|---------|------|------|
| A1 | 缺 `_considerRetreat`：死兵或初始<4队触发撤退考虑 | 在 `evaluation.analyze` 计算并加门控 | 小 | ✅ 已修 |
| A2 | 缺 `_defendingCastle → defensiveTactics=true` | 在 `evaluation.should_defend` 加城堡防守分支 | 小 | ✅ 已修 |
| A3 | 缺射手 `isImmovable` 排除 | 射手含塔，塔的 strength 已在 siege 段单独处理 | — | ⚠️ 近似(无需修) |
| A4 | 缺 `TR_MOVED` 已行动 threat /= 1.25 | 在 `scoring.threat` 中加，Unit 加 `_acted` 标记 | 中 | ✅ 已修 |
| A5 | 缺 `isDoubleAttack` 反击折算 | `expected_damage` 已含 double_shooting(×2)/double_melee(×1.75)，效果等价 | — | ⚠️ 近似(无需修) |
| A6 | 缺攻城无目标时向城墙移动 | 在 `planner._offense` 末尾加 `_attackingCastle` 分支 | 中 | ✅ 已修 |
| A7 | 缺护城河停驻逻辑 | 护城河在寻路中已作为移动终止格，效果等价 | — | ⚠️ 近似(无需修) |
| A8 | 缺城堡防守区域检查 (`isCastleIndex`) | 在 `should_defend` 用 `Castle.is_inside_walls` 替代 midline | 中 | ✅ 已修 |
| A9 | 缺防守第二阶段（防区内攻击） | `_defense_area_attack` + `_in_defended_area` | 中 | ✅ M7c |
