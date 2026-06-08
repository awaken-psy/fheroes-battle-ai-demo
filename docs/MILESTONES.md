# 里程碑 — 战斗 AI 复刻

> 把 [`TODO.md`](TODO.md) 的 P0–P4 重组为可发版、可验收的里程碑。
> 每个里程碑都有明确的**退出标准(Exit Criteria)**:满足即可打 tag 发版,不满足不进下一个。
> 保真度百分比以「覆盖原版战斗 AI ~3000 行的决策行为」为分母,为粗略估计。

> 注:`v0.2.0` 是已发布的可玩 demo(GUI + 当前 AI),里程碑版本从 `v0.3.0` 起。

| 里程碑 | 主题 | 对应 TODO | 目标保真度 | 状态 |
|---|---|---|---|---|
| **M1** `v0.3` | 验证闭环可用 | P0 | ~45%(不变,但**可量化**) | ✅ 完成 |
| **M2** `v0.4` | 保真度修正 | P1 | ~55% | 📋 待办 |
| **M3** `v0.5` | 法术系统 | P2(法术) | ~70% | 📋 待办 |
| **M4** `v0.6` | 撤退 + 完整回合行为 | P2(其余) | ~85% | 📋 待办 |
| **M5** `v1.0` | 战场机制与调优 | P3 + P4 | ~95% | 📋 待办 |

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
- [ ] 战力公式对齐原版量纲(`engine/unit.py` / `ai/evaluation.py`)
- [ ] `threat()` 升级为双向伤害+反击估值(`ai/scoring.py`)
- [ ] `isLimitOfTurnsExceeded`:50 回合无死亡撤退,替换 200 回合硬上限

**退出标准:**
- `strength` 公式 = `(1+0.1·atk+0.05·def)×baseStrength×count`,有单测覆盖
- `should_defend` 的姿态阈值在新量纲下重新校准,回归测试更新且通过
- arena 对战不再出现 200 回合僵局触顶的情况

---

## M3 — 法术系统 `v0.5`

> 原版最大单块(958 行),也是观感差距最明显处。先做这块。

任务:
- [ ] `engine/spells.py`:伤害 / 增益 / 减益 / 召唤 / 复活,buff/debuff 为**有时限临时状态**
- [ ] `ai/spells.py`:施法阈值 `myStr²/enemyStr×0.04`、法力开方折价 `value/√(sp/3)`、各法术 ratio 与时长系数(对齐 `ai_battle_spell.cpp:71` `selectBestSpell`)

**退出标准:**
- 至少覆盖 Magic Arrow / Lightning Bolt / Haste / Slow / Bless / Curse
- buff/debuff 到期自动失效(单测验证),不永久改属性
- arena 中带法术英雄 vs 无法术,胜率有显著、合理的差异

---

## M4 — 撤退 + 完整回合行为 `v0.6`

> 补齐回合内剩余的原版行为,达到"一局完整战斗的每一步都有原版对应逻辑"。

任务:
- [ ] `ai/retreat.py`:续战条件 + Retreat/Surrender 决策 + 告别伤害法术(`ai_battle.cpp:703-870`)
- [ ] `berserkTurn`:狂暴单位攻击/移向最近单位(不分敌我)
- [ ] 士气/运气引擎层(高士气额外行动 / 低士气跳过 / 运气双倍);**planner 不评估士气运气**

**退出标准:**
- 劣势方在合适阈值下会撤退而非战至全灭(arena 可观测到撤退率)
- 狂暴/士气/运气均有单测;planner 决策代码中无士气运气期望计算

---

## M5 — 战场机制与调优 `v1.0`

> 锦上添花:扩展战场内容,并在 arena 闭环上做对抗性调优。

任务:
- [ ] 攻城 / 城墙 / 箭塔逻辑
- [ ] 宽体单位(占两格)+ 双格攻击
- [ ] `findOptimalPositionForSubsequentAttack` 谨慎走位优化落点
- [ ] 更多兵种(~60 种)+ 特殊能力(无限反击/死亡凝视/吸血/自愈)
- [ ] 与原版 AI 同快照行为对照;scoring 权重在 arena 上参数搜索
- [ ] (按需)阵型编队;(可选)C++ 模拟器/MCTS 陪练

**退出标准:**
- 攻城战可完整进行(含城墙/箭塔)
- 同战场快照下,本项目 planner 与原版决策一致率 ≥ 目标阈值(待 M5 启动时定标)
