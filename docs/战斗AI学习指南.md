# 战斗 AI 学习指南

> 源码：[ai_battle.h/cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp)（2,091 行）+ [ai_battle_spell.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp)（958 行），共约 3,050 行
>
> 核心思想：**每个单位轮到行动时，评估局面 → 选最优行动（攻击/移动/施法/撤退），没有搜索树。**
>
> 与战略 AI（第 1-4 章）完全独立——战略 AI 只决定"去哪、打谁"，战斗开始后由战斗引擎直接调用 `BattlePlanner`，中间不传任何战略层上下文。

---

## 学习路线

```
§1 战斗入口：BattlePlanner 如何被调用
  ↓
§2 局面评估：analyzeBattleState() 计算哪些全局变量
  ↓
§3 单位决策主循环：planUnitTurn() 的四步决策
  ↓
§4 射手决策：archerDecision() 的三态逻辑
  ↓
§5 近战进攻：meleeUnitOffense() 的三级目标选择
  ↓
§6 近战防御：meleeUnitDefense() 的护弓逻辑
  ↓
§7 法术选择：selectBestSpell() 的价值评估
  ↓
§8 特殊机制：狂暴、撤退/投降、反僵局
```

---

## §1 战斗入口：BattlePlanner 如何被调用

**代码位置**：[ai_battle.h:73](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.h#L73)（类定义）、[battle_arena.cpp:438,508](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_arena.cpp#L438)（调用点）

`BattlePlanner` 是单例（`Get()` 返回 `static` 局部变量），战斗引擎（`Battle::Arena`）在两个时刻调用它：

```
[battle_arena.cpp](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/battle/battle_arena.cpp)
│
├── Arena::StartBattle()  第 438 行
│   └── BattlePlanner::Get().battleBegins()    ← 重置计数器
│
└── Arena::turnLoop()     第 508 行（每回合每个单位）
    └── BattlePlanner::Get().BattleTurn(arena, currentUnit, actions)
        ├── isLimitOfTurnsExceeded()  ← 反僵局检查
        └── planUnitTurn()            ← 核心决策
```

**`battleBegins()` 做的事**：把回合计数器 `_currentTurnNumber` 归零，反僵局计数器 `_numberOfRemainingTurnsWithoutDeaths` 设为 50，双方死亡计数归零。

**接口极度简洁**：战斗引擎只传 `arena`（战场状态）、`currentUnit`（当前行动单位）、`actions`（输出行动列表）。战斗 AI 不需要知道英雄的角色（CHAMPION 还是 SCOUT），也不需要知道为什么要打这场仗。

### BattlePlanner 的成员变量

战斗 AI 不做持久化的战略决策，但每回合需要记录大量临时状态。这些变量在 `analyzeBattleState()` 中计算，后续所有决策函数消费：

```
                    ┌─────────── 分析阶段写入 ───────────┐
                    │                                     │
敌方信息            │  _enemyArmyStrength                 │  我方信息
_enemyShootersStr  │  _enemyRangedUnitsOnly              │  _myArmyStrength
_enemySpellStr     │  _enemyAverageSpeed                 │  _myShootersStrength
                    │                                     │  _myRangedUnitsOnly
                    │                                     │  _myArmyAverageSpeed
                    │                                     │
                    ├─────────── 战术标志 ────────────────┤
                    │  _attackingCastle  攻城方            │
                    │  _defendingCastle  守城方            │
                    │  _defensiveTactics  防御战术         │
                    │  _cautiousOffensive  谨慎进攻        │
                    │  _avoidStackingUnits  避免聚堆       │
                    │  _considerRetreat  考虑撤退          │
                    └─────────────────────────────────────┘
```

---

## §2 局面评估：analyzeBattleState()

**代码位置**：[ai_battle.cpp:949-1170](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L949-L1170)

每个单位轮到行动时，第一步不是"决定做什么"，而是"重新理解战场"。`analyzeBattleState()` 遍历所有敌我单位，计算上述全部成员变量。

### 2.1 军力统计

对每个活着的单位累加 `GetStrength()`（综合攻击/防御/血量/数量的军力值），同时分类统计射手强度和远程单位强度：

```
for 每个敌方单位:
    _enemyArmyStrength += unitStr
    if 是射手且非固定单位:
        _enemyRangedUnitsOnly += unitStr
        if 有范围射击能力且未被近战阻挡:
            areaAttackThreat += unitStr
    _enemyAverageSpeed += speed × unitStr   ← 军力加权平均速度

_enemyShootersStrength = _enemyRangedUnitsOnly
```

**注意**：对我方单位的遍历**不跳过死亡单位**——死亡单位的信息用于判断是否考虑撤退。

### 2.2 攻城修正

如果战场有城堡（且至少有一个箭塔存活），射手强度需要修正：

```
守城方:
    _myShootersStrength += 箭塔总强度      ← 箭塔也算"射手"
    _enemyShootersStrength /= (1 + 城墙射击惩罚%)  ← 城墙挡住敌人射击

攻城方: 反过来
```

箭塔总强度 = 中央塔 + 左塔 + 右塔的 `GetStrength()` 之和。如果攻方有箭术技能或无射击惩罚宝物，则不做城墙惩罚修正。

### 2.3 英雄法力强度

```
if (我方英雄存在):
    _myShootersStrength += 英雄最大伤害法术的预期伤害值
if (敌方英雄存在):
    _enemySpellStrength = 敌方英雄的魔法战略价值（基于我方军力）
    _enemyShootersStrength += 敌方英雄最大伤害法术的预期伤害值
```

这把英雄的法术能力也折算进"射手力量"——因为法术本质上也是一种远程攻击。

### 2.4 战术决策

`analyzeBattleState()` 的最后计算三个战术标志，它们直接决定后续单位的行动模式：

#### `_considerRetreat`（考虑撤退）

```
任意一个我方单位被打死（count==0 且 dead>0）  → true
或者初始单位总数 < 4                           → true
```

#### `_defensiveTactics`（防御战术）

```
条件链（全部满足才启用）:
  1. 当前单位还在己方半场（未越过中线）
  2. 我方军力不是碾压级（< 飞行单位 ×6 / 地面单位 ×10 倍）
  3. 我方射手 ≥ 敌方射手
  4. 不是"我方射手太弱"（射手比例 < 15%）
  5. 不是"敌方射手太多"（敌方射手比例 > 66%）
  6. 如果守城 → 直接 true（有城墙保护）
```

**设计意图**：防御战术的核心场景是"我方射手比对方强，应该在己方半场等对方冲过来"。如果对方射手也很多，那不能等——必须主动冲上去压制对方射手。

#### `_cautiousOffensive`（谨慎进攻）

```
敌方射手比例 < 15%  → true
```

敌方几乎没有射手时，不急着冲锋，走一步算一步——反正对方射不到你。

#### `_avoidStackingUnits`（避免聚堆）

```
敌方范围射击军力占敌方总军力 > 10%  → true
```

敌方有范围攻击（如火元素/法师的面积射击）时，己方单位不要站太近，避免被一发打中多个。

---

## §3 单位决策主循环：planUnitTurn()

**代码位置**：[ai_battle.cpp:689-947](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L689-L947)

```
planUnitTurn(arena, currentUnit)
│
├── 特殊：狂暴状态？→ berserkTurn()，直接返回
│
├── Step 1: analyzeBattleState()          ← 重算所有全局变量
│
├── Step 2: 撤退/投降决策                  ← 只在 _considerRetreat 时检查
│   ├── 能撤退？有宝物？能重新雇佣？       ← 多因素决策
│   ├── 决定撤退 → 先放 farewell spell → RETREAT
│   └── 决定投降 → 先放 farewell spell → SURRENDER
│
├── Step 3: 法术决策                       ← 英雄还没施法时才检查
│   ├── selectBestSpell() 找最优法术
│   └── 找到了？→ SPELLCAST，直接返回      ← 施法占一个回合的行动
│
├── Step 4: 单位行动决策树
│   ├── 是射手？→ archerDecision()
│   └── 是近战？→
│       ├── _defensiveTactics？→ meleeUnitDefense()
│       └── 否则？→ meleeUnitOffense()
│       └── 最终: ATTACK / MOVE / SKIP
│
└── 没有行动？→ SKIP（跳过回合）
```

### Step 2 详解：撤退/投降决策

这是战斗 AI 中最长的单个代码块（[planUnitTurn() 的第 703-870 行](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L703-L870)），决策逻辑如下：

```
前提: _considerRetreat 为 true（有单位阵亡或初始单位太少）
      且不是人类控制的英雄在自动战斗
      且我方军力 × 难度系数 < 敌方军力
│
├── 有值钱宝物？→ 撤退（别让宝物落入敌手，尤其联盟战）
├── 无法重新雇佣？（最后一个英雄且没有城堡/正在守最后一座城）→ 不撤退
├── 英雄属性总和 ≥ 10？→ 撤退（保护经验值）
└── 否则 → 继续打
```

如果决定撤退/投降，会先执行 **farewell spellcast**——用 `selectBestSpell(retreating=true)` 选伤害最高的法术放一炮再走。撤退模式下只考虑伤害法术，且不需要超过法术阈值。

**撤退 vs 投降的区别**：
- 撤退（Retreat）：不需要花钱，英雄消失后可以从酒馆重新雇佣，但**丢失全部宝物和军队**
- 投降（Surrender）：需要付金币（敌方军队价值的 50%），英雄保留宝物，军队返回最近的城堡

选择优先级：如果能撤退就撤退（免费），不能撤退再考虑投降（看有没有钱）。

---

## §4 射手决策：archerDecision()

**代码位置**：[ai_battle.cpp:1172-1537](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1172-L1537)

射手有三条路，按优先级排列：

```
archerDecision()
│
├── 1. 能逃跑？→ 退到安全位置，不攻击
│   └── 条件：当前被威胁 + 存在安全位置 + 比威胁方快
│
├── 2. 被近战堵住且无法逃跑？→ 近战攻击相邻敌人
│   └── 选净伤害差（我方伤害 - 反击伤害）最大的目标
│
└── 3. 可以射击？→ 射击最高价值目标
    └── 对每个敌方单位计算 evaluateThreatForUnit()
```

### 4.1 逃跑评估

这是射手决策中最复杂的部分（[第 1180-1379 行](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1180-L1379)）：

**Step 1**：是否值得尝试逃跑？

```
if 敌方有飞行单位 → 不值得（飞单位追得上）→ 返回 -1
```

**Step 2**：评估所有可能位置的威胁

临时把当前单位从战场"拿走"（`UnitRemover` RAII），遍历所有可达位置，对每个位置计算：
- `threateningEnemiesIndexes`：哪些敌方单位能到达这个位置
- `distanceToNearestEnemy`：离最近敌方单位的距离

**Step 3**：当前位置是否安全？

```
if 当前位置没有威胁 → 不需要逃跑 → 返回 -1
```

**Step 4**：即使当前位置受威胁，逃跑值得吗？

```
对所有威胁当前单位的敌人:
    该敌人速度 + 2 < 我方速度？  ← 必须明显比我慢
if 全部满足 → 值得尝试
else → 不值得（跑不掉，不如打）→ 返回 -1
```

**Step 5**：在安全位置中选最优

```
对每个安全位置（无威胁的可达位置）:
    排序标准 = (离敌人距离[越大越好], 1/离中心距离[越小越好])
选排序最高的
```

**为什么偏爱中心而非角落？** 角落容易被敌方单位堵死——一旦被逼到角上，射手再也无法逃脱。中心位置有更多逃跑路线。

### 4.2 射击目标选择

不被近战阻挡时，对每个敌方单位计算 `evaluateThreatForUnit(currentUnit)`，选最高威胁的。

**范围射击（AREA_SHOT）的特殊处理**：
- 评估目标周围所有受影响的单位（敌我都有）
- 如果友军受伤 ≥ 3 倍敌军受伤 → 标记为危险射击，跳过
- 射击宽体目标时，分别计算打头和打尾的溅射范围

### 思考题

- 射手的逃跑逻辑要求"敌人速度 + 2 < 我方速度"，这个 +2 的缓冲区有什么用？如果去掉这个条件会怎样？
- 射手被堵住时只考虑"净伤害差"，不考虑击杀价值。你能想到什么情况下这会导致次优决策？
- 射击目标选择用的是 `evaluateThreatForUnit()`（敌方对我方的威胁），而不是 `getPotentialDamage()`（我能打多少）。为什么？

---

## §5 近战进攻：meleeUnitOffense()

**代码位置**：[ai_battle.cpp:1568-1706](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1568-L1706)

```
meleeUnitOffense()
│
├── 1. 已在攻击范围内的目标
│   └── getMeleeBestOutcome() → 选最优攻击
│
├── 2. 不在范围内：选远程目标（两轮筛选）
│   ├── 第一轮：追"逃不掉"的目标
│   │   └── 射手 / 固定单位 / 速度比我慢的非飞行单位
│   └── 第二轮：追任何目标
│   └── 评分 = threatForUnit / distance
│
└── 3. 攻城时没有目标？→ 走向城墙
```

### 5.1 第一级：已在攻击范围内

`getMeleeBestOutcome()` 对每个敌方单位调用 `BestAttackOutcome()`，计算复合评分：

```
MeleeAttackOutcome {
    canAttackImmediately   ← 本回合能否打到？
    positionValue          ← 攻击位置本身的价值（能挡多少射手等）
    attackValue            ← 目标的威胁值
}

IsOutcomeImproved() 比较规则:
    能立即攻击的 > 不能立即攻击的（首要条件）
    同等条件下: 位置价值大的优先
    位置价值也相同: 目标威胁大的优先
```

### 5.2 第二级：选择远程目标

对不在攻击范围内的敌人，用距离加权评分：

```
priority = enemy.evaluateThreatForUnit(currentUnit) / dist
```

**两轮筛选的设计意图**：

```
第一轮: 追"逃不掉"的目标
  ├── 射手 → 必追（即使对方更快，至少能挡住它）
  ├── 固定单位（Speed::STANDING）→ 必追
  └── 非飞行单位且速度 < 我方 → 追得上

如果第一轮没找到 → 第二轮: 追任何目标
```

为什么第一轮优先追"逃不掉"的？因为追一个跑得掉的单位是浪费回合——你走到它上回合的位置，它下回合又跑了。先追跑不掉的，确保至少能打到东西。

### 5.3 谨慎进攻模式

当 `_cautiousOffensive = true`（敌方射手比例 < 15%）时，不走到路径尽头，而是用 `findOptimalPositionForSubsequentAttack()` 选一个更安全的位置：

```
对路径上每一步:
    累加所有能到达该位置的敌方地面单位的威胁值
选威胁最低且最靠近目标的位置
```

即：在前进的同时尽量躲在敌方近战单位的攻击范围之外。

### 5.4 城墙冲撞

攻城战且没有其他目标时，走向城墙下的格子（`cellsUnderWallsIndexes = {7, 28, 49, 72, 95}`），准备下回合破门。

### 思考题

- 第一轮优先追"逃不掉"的目标。但如果有一个速度极慢的高威胁单位和一只快被消灭的弱小单位，AI 会追谁？这是最优的吗？
- 谨慎进攻模式只考虑地面单位的威胁（跳过飞行和射手）。为什么？

---

## §6 近战防御：meleeUnitDefense()

**代码位置**：[ai_battle.cpp:1708-2065](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1708-L2065)

防御战术的核心逻辑：**保护己方射手**。

```
meleeUnitDefense()
│
├── 1. 保护己方射手（最复杂）
│   ├── 对每个己方射手:
│   │   ├── 找最佳掩护位置（射手旁边的可达格子）
│   │   ├── 找射手旁边的敌方单位（正在堵射手的）
│   │   ├── 计算射手价值 = shooterStrength - dist × (totalRanged / 15)
│   │   ├── 选最有价值的射手去保护
│   │   └── 如果射手旁边有敌人 → 攻击那个敌人
│   └── （最终: 移动到掩护位置 / 攻击堵射手的敌人）
│
└── 2. 没有射手需要保护？→ 在己方半场打敌人
```

### 6.1 射手价值计算

```
archerValue = frnd.GetStrength() - dist × defenseDistanceModifier
其中: defenseDistanceModifier = _myRangedUnitsOnly / 15.0
```

这个公式的含义：每个射手有一个基础价值（`GetStrength()`），距离越远减分越多。减分速率取决于己方总射手力量——射手越多，每个射手的保护越重要（因为减分慢，所以即使远也值得去保护）。

**源码注释中的例子**：
- 射手 A 强度 200（格子 0），射手 B 强度 100（格子 88）
- modifier = (200+100)/15 = 20
- 从格子 66（离 A 约 5 格，离 B 约 1 格）看：
  - A 的价值 = 200 - 20×5 = 100
  - B 的价值 = 100 - 20×1 = 80
  - 结论：去保护 A（更强的那个）

### 6.2 掩护位置选择

不是简单地站到射手旁边。AI 会考虑：

1. **宽体单位优先从侧面掩护**——因为从正面掩护会挡住射手的视线（虽然 HoMM2 中这个影响有限）
2. **避免聚堆模式**（`_avoidStackingUnits`）时，掩护单位不能和其他掩护单位紧挨——至少隔一格
3. 有方向优先级：根据射手朝向和掩护单位是否宽体，决定先尝试哪些方向的格子

### 6.3 攻击堵射手的敌人

如果射手旁边有敌方单位（距离=1），这是最紧急的情况——射手无法射击。防御模式下的近战单位会优先攻击这些"堵射手"的敌人。

额外细节：
- 无视反击的单位（如精灵/幽灵）即使不直接堵射手也会顺手攻击相邻敌人
- 如果己方射手有范围射击能力（`AREA_SHOT`），掩护单位更应该帮忙清理旁边的敌人（否则射手不敢开枪怕误伤）

### 6.4 己方半场防御

如果没有任何射手需要保护，防御模式退化为"只在己方半场打敌人"——通过 `isPositionLocatedInDefendedArea()` 过滤攻击位置。

```
己方半场定义:
  普通战斗 → 格子到己方边缘的 X 距离 ≤ 半个战场宽度
  守城战   → 城堡内部格子（isCastleIndex）
```

### 思考题

- 防御模式下近战单位优先保护射手。但如果有个敌人正在拆我方城墙（攻城战），AI 不会去阻止。这是不是一个设计缺陷？
- `defenseDistanceModifier` 用 `_myRangedUnitsOnly / 15.0` 计算。为什么 15 这个常数能让"两个射手时优先保护强的那个"这个策略生效？

---

## §7 法术选择：selectBestSpell()

**代码位置**：[ai_battle_spell.cpp:71-156](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L71-L156)

法术选择是战斗 AI 中最独立的子系统——它有自己的价值评估体系，且一旦决定施法，整个单位回合就结束了（施法 = 行动）。

### 7.1 选择流程

```
selectBestSpell(arena, currentUnit, retreating)
│
├── 计算阈值 spellValueThreshold
│
├── 遍历英雄所有法术:
│   ├── 跳过: 非战斗法术 / 被禁用 / 法力不够
│   ├── 退缩模式只看伤害法术
│   │
│   ├── damage      → spellDamageValue()
│   ├── dispel      → spellDispelValue()
│   ├── summon      → spellSummonValue()
│   ├── resurrect   → spellResurrectValue()
│   ├── dragonslayer→ spellDragonSlayerValue()
│   ├── teleport    → spellTeleportValue()
│   ├── earthquake  → spellEarthquakeValue()
│   ├── buff(友方)  → spellEffectValue(friendly)
│   └── debuff(敌方)→ spellEffectValue(enemies)
│
├── 每个法术的原始价值 → 归一化: value / sqrt(cost / 3)
│
└── 超过阈值？→ 更新最佳法术
```

### 7.2 法术阈值

AI 不会随便施法——只有法术价值超过阈值才放：

```
spellValueThreshold = _myArmyStrength² / _enemyArmyStrength × 0.04

修正:
  敌方射手比例 > 50%  → 阈值 × 0.5（更容易施法——被射很痛）
  法力 < 最大法力的一半 → 阈值 × 2.0（省着用法力）
```

阈值的含义：军力相当（比值≈1）时，阈值约为我方军力的 4%。一个法术至少要有"打掉我方 4% 军力"的效果才值得放。

### 7.3 价值归一化

```
归一化价值 = outcome.value / sqrt(spellCost / 3)
退缩模式: 不除以消耗（反正要走了，法力不用白不用）
复活法术: 忽略阈值检查
```

用 `sqrt` 做非线性衰减——3 点法力（1 级法术）的基准比为 1:1，12 点法力（3 级法术）的比率为 1:2。这确保高消耗法术不会仅仅因为数值大就被优先选择。

### 7.4 伤害法术评估：spellDamageValue()

**代码位置**：[ai_battle_spell.cpp:158-273](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L158-L273)

```
对每个受影响单位:
    actualDamage = spellDamage × (100 - magicResist) / 100

    if damage == 0 → 免疫，跳过

    if 能秒杀:
        value = unit.GetStrength() + armyStrength × bonus
                bonus = 7%（快单位）/ 3.5%（慢单位）
                ← 击杀快单位的奖励更高

    if 打醒沉睡单位（如被催眠的）:
        value = killPercentage × strength + (killPercentage - 1) × strength
                ← 惩罚：只打了 30% 意味着 70% 是负面效果

    otherwise:
        value = (damage / hitpoints) × unit.GetStrength()
```

**群体伤害法术**（如火球/闪电链）会考虑友军伤害——从总价值中减去打到的友军价值。如果退缩时打死自己当前单位，直接跳过这个法术。

### 7.5 增益/减益法术：spellEffectValue()

**代码位置**：[ai_battle_spell.cpp:368-560](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L368-L560)

每种法术有一个基础 `ratio`，最终价值 = `target.GetStrength() × ratio × spellDurationMultiplier`。

| 法术 | ratio | 修正条件 |
|------|------:|---------|
| Hypnotize | 1.50 | 最高基础比例——夺取控制权 |
| Berserker | 0.85 | 非射手时按距离衰减；最后一个敌人无效 |
| Blind | 0.80 / 0.40 | 最后一个敌人且已反击或无限反击 → 降低 |
| Paralyze | 0.85 / 0.50 | 同 Blind 的条件 |
| Disrupting Ray | 0.20× | 按目标防御比例和敌我军力比调整 |
| Curse / Mass Curse | 0.15 | 目标伤害 max==min 时无效 |
| Bless / Mass Bless | 0.15 | 目标伤害 max==min 时无效；驱散 Curse 时 ×2 |
| Slow / Mass Slow | 0.10×lostSpeed | 对射手几乎无效；已加速的目标 ×2 |
| Haste / Mass Haste | 0.05×gainedSpeed | 速度低于敌军时 ×2 |
| Steel Skin | 0.20 | |
| Bloodlust | 0.10 | 固定值 |
| Stone Skin | 0.10 | |
| Shield / Mass Shield | 动态 | enemyRangedRatio × 0.3 |
| Anti-Magic | 动态 | 按敌方法术强度映射到 0-0.9 |
| Mirror Image | 0.33-1.0 | 射手 1.0，飞行 0.55，其他 0.33；速度 < 敌均速时 /5 |

#### `spellDurationMultiplier()`

```
duration = heroPower + 宝物加成
if duration < 2 且目标已行动过 → return 0（只持续不到 1 轮，没用）
else → return 1
```

#### `isSpellcastUselessForUnit()`

跳过已经拥有该状态的单位（如已 Haste 的不再施 Haste）、固定单位（已 Blind 的不再施其他状态法术，除非 Anti-Magic）。

### 7.6 复活法术：spellResurrectValue()

```
value = missingHP × unitMonsterStrength / monsterHitPoints

if 我方军力 > 敌方 且不是普通 Resurrect → value × 2
    （优势时复活是永久保留的）
```

先考虑活着的单位，再考虑墓地里的单位。

### 7.7 召唤法术：spellSummonValue()

```
value = 召唤物.GetStrengthWithBonus(heroAttack, heroDefense)
if 我方军力 > 敌方 × 2 → value / 2（已经碾压，不需要召唤）
```

只有英雄旁边有空位时才能召唤。

### 思考题

- 法术阈值公式 `myStrength² / enemyStrength × 0.04` 意味着军力比越大阈值越高。这合理吗？什么场景下这个公式会导致 AI 不该省法力时却省了？
- Blind 对最后一个敌人 ratio 只有 0.40（而非 0.80），因为被致盲的单位被攻击后会醒来。但如果我方有一个能一击秒杀这个单位的兵种，是不是应该先致盲再秒杀（避免反击）？AI 能考虑这种配合吗？
- Mirror Image 对射手比例 1.0、对地面近战比例 0.33。这个差异合理吗？什么情况下给地面单位镜像比给射手更有价值？

---

## §8 特殊机制

### 8.1 狂暴状态：berserkTurn()

**代码位置**：[ai_battle.cpp:508-602](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L508-L602)

被 Berserker 法术控制的单位不经过正常的决策流程，而是：

```
找最近的单位（不分敌我，GetNearestTroops）
if 是射手 → 射击最近的
if 是近战 → 走到最近的旁边攻击 / 走向最近的
没有可达单位 → SKIP
```

狂暴单位的行为是"无脑攻击最近的"——它甚至可能打自己的队友。

### 8.2 反僵局机制：isLimitOfTurnsExceeded()

**代码位置**：[ai_battle.cpp:630-687](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L630-L687)

```
MAX_TURNS_WITHOUT_DEATHS = 50

每回合（仅攻方回合检查）:
    if 双方都无人死亡 → 计数器 -1
    if 有人死亡 → 计数器重置为 50

    计数器归零时:
        自动战斗模式 → 关闭自动战斗（玩家接管）
        快速战斗模式 → 攻方英雄撤退
```

**为什么需要这个？** 两个飞行单位互相绕圈，或者一个慢速单位永远追不到对方，战斗会无限持续。50 回合无人死亡的兜底防止 AI 陷入死循环。

### 8.3 双格攻击：optimalAttackVector()

**代码位置**：[ai_battle.cpp:110-158](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L110-L158)

宽体攻击者（如龙/比蒙）攻击时可能打到目标背后的另一个单位。`optimalAttackVector()` 尝试所有攻击方向，选背后有额外目标的那个——相当于白打一个单位。

全范围攻击（如九头蛇 `isAllAdjacentCellsAttack()`）直接累加所有相邻敌方单位的威胁值。

### 8.4 位置价值评估：evaluatePotentialAttackPositions()

**代码位置**：[ai_battle.cpp:202-269](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L202-L269)

对战场上的每个攻击位置（敌方单位旁边），计算该位置的"价值"：

```
对每个敌方单位旁边的每个可达位置:
    如果该单位是射手 → 位置价值 += 该单位威胁（鼓励去挡射手）
    如果该单位是近战 → 位置价值 = max(位置价值, 该单位威胁)
```

射手的贡献用加法（挡住多个射手更有价值），近战的贡献用 max（打近战主要看威胁最高的那个）。

---

## 全局数据流总图

```
[battle_arena.cpp]                   BattlePlanner                     输出
───────────────                     ──────────────                    ────

battleBegins()  ───→ 重置计数器

每回合每单位:
BattleTurn()
  │
  ├── isLimitOfTurnsExceeded() ──→ RETREAT / 关闭自动战斗
  │
  └── planUnitTurn()
       │
       ├── analyzeBattleState()
       │    ├── _myArmyStrength / _enemyArmyStrength
       │    ├── _myShootersStrength / _enemyShootersStrength
       │    ├── _defensiveTactics / _cautiousOffensive
       │    └── _considerRetreat
       │                        │
       ├── 撤退/投降决策 ←──────┘──→ RETREAT / SURRENDER
       │
       ├── selectBestSpell()  ──→ SPELLCAST
       │    ├── spellDamageValue()
       │    ├── spellEffectValue()
       │    ├── spellSummonValue()
       │    └── ...
       │                        │
       └── 单位行动 ←──────────┘
            ├── 射手: archerDecision()
            │    ├── 逃跑（MOVE）
            │    ├── 被堵近战（ATTACK）
            │    └── 射击（ATTACK）
            ├── 近战进攻: meleeUnitOffense()
            │    ├── 打范围内的（ATTACK）
            │    ├── 追远程目标（MOVE/ATTACK）
            │    └── 走向城墙（MOVE）
            └── 近战防御: meleeUnitDefense()
                 ├── 保护射手（MOVE/ATTACK）
                 └── 半场防御（ATTACK）
```

---

## 文件索引

| 文件 | 行数 | 主要功能 |
|------|------|---------|
| [ai_battle.h](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.h) | 152 | BattlePlanner 类定义、成员变量、SpellSelection/SpellcastOutcome 结构体 |
| [ai_battle.cpp:1-603](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1-L603) | 603 | 辅助函数：MeleeAttackOutcome、optimalAttackVector、berserkTurn 等 |
| [ai_battle.cpp:605-947](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L605-L947) | 343 | 核心流程：Get()、battleBegins()、BattleTurn()、planUnitTurn() |
| [ai_battle.cpp:949-1170](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L949-L1170) | 222 | 局面评估：analyzeBattleState() |
| [ai_battle.cpp:1172-1537](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1172-L1537) | 366 | 射手决策：archerDecision() |
| [ai_battle.cpp:1539-1706](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1539-L1706) | 168 | 近战进攻：getMeleeBestOutcome()、meleeUnitOffense() |
| [ai_battle.cpp:1708-2091](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle.cpp#L1708-L2091) | 384 | 近战防御：meleeUnitDefense()、isPositionLocatedInDefendedArea() |
| [ai_battle_spell.cpp:71-156](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L71-L156) | 86 | 法术选择入口：selectBestSpell() |
| [ai_battle_spell.cpp:158-273](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L158-L273) | 116 | 伤害法术：spellDamageValue() |
| [ai_battle_spell.cpp:275-366](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L275-L366) | 92 | 法术比例：spellDurationMultiplier()、getSpellSlowRatio()、getSpellHasteRatio() 等 |
| [ai_battle_spell.cpp:368-560](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L368-L560) | 193 | 增益/减益法术：spellEffectValue()（单目标版，switch-case） |
| [ai_battle_spell.cpp:562-574](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L562-L574) | 13 | 群体法术：spellEffectValue()（多目标版） |
| [ai_battle_spell.cpp:576-672](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L576-L672) | 97 | 驱散/复活法术：spellDispelValue()、spellResurrectValue() |
| [ai_battle_spell.cpp:675-751](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L675-L751) | 77 | 召唤/屠龙法术：spellSummonValue()、spellDragonSlayerValue() |
| [ai_battle_spell.cpp:754-958](https://github.com/ihhub/fheroes2/blob/master/src/fheroes2/ai/ai_battle_spell.cpp#L754-L958) | 205 | 法术辅助：isSpellcastUselessForUnit()、spellTeleportValue()、spellEarthquakeValue() |

---

## 综合思考题

1. **战斗 AI 没有搜索树**，每步只看当前最优。你能构造一个场景，让战斗 AI 因为贪心而输掉一场"两步配合就能赢"的战斗吗？（提示：致盲 + 秒杀的配合）

2. **防御战术的触发条件**要求"我方射手 ≥ 敌方射手"。如果双方射手军力相当但我方射手全是 1 级弓箭手，对方全是 5 级法师，AI 仍然选择防御。这合理吗？

3. **法术阈值** `myStrength² / enemyStrength × 0.04` 意味着劣势时（比值小）阈值更低，更容易施法。这是好设计还是坏设计？优势时省法力、劣势时拼命——直觉上说得通，但有没有反例？

4. **射手的逃跑决策**要求"敌人速度 + 2 < 我方速度"。假设一个速度 6 的精灵射手面对一个速度 3 的步兵。精灵跑了之后，步兵下回合还能追上吗？那精灵不是白跑了吗？

5. **狂暴单位**的 AI 只是攻击最近的单位，不分敌我。如果你是设计师，你会如何改进狂暴 AI 使其更聪明（但仍保持"失控"的特质）？

6. 战斗 AI 和战略 AI 完全独立——战斗 AI 不知道战略 AI 派这个英雄来打的目的是什么（抢矿？攻城？清怪？）。如果战斗 AI 能获得这个信息，哪些决策可以做得更好？
