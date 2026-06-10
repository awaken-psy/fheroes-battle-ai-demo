# 里程碑 — 战斗 AI 复刻

> 规则层 M1–M7e 已全部完成，~99% 保真度，298 测试。
> 当前重心：A 系列 AI 决策层对齐原版（87% → 96%，A3 完成）。
>
> 详细规则对照见 [`docs/rules-audit.md`](rules-audit.md)（303 项）。
> AI 行为审计见 [`docs/ai-audit.md`](ai-audit.md)（111 条）。

---

## 规则层 ✅ 完成

| 里程碑 | 主题 | 保真度 | commit |
|---|---|---|---|
| **M1** | 验证闭环（arena + 测试 + turn order） | ~45% | — |
| **M2** | 保真度修正（strength/threat/撤退/交替出手） | ~55% | — |
| **M3** | 法术系统（6 法术 + Hero + buff/debuff） | ~70% | — |
| **M4** | 撤退 + 士气/运气 + 狂暴 | ~85% | — |
| **M5** | 特殊能力 + 谨慎走位 | ~90% | — |
| **M5b** | 宽体单位（2 格占位 + tail 几何） | ~91% | 49b740ae |
| **M6a** | 兵种扩充（Knight+Barbarian 20 种 + min/max 伤害） | ~92% | 1f54c421 |
| **M6b** | 攻城系统（墙/河/塔/车/桥完整） | ~94% | — |
| **M6c** | 源码审计对齐（111 条行为差异清单） | ~95% | abb209d4 |
| **M7** | 兵种全阵营（6 阵营 + 中立，63 种） | ~96% | 5029ff98 |
| **M7b** | 法术扩充（38 种战斗法术 + AOE/控制/功能） | ~97% | 5e2dd88 |
| **M7c** | AI 行为精细化（8 审计项收尾，87% 决策覆盖） | ~98% | 3b1c223 |
| **M7d** | 英雄战斗技能（Archery/Ballistics/Leadership/Luck） | ~98% | — |
| **M7e** | 规则保真度收尾（英雄攻防/d24/Golem/BoneDragon/Genie） | ~99% | 13dabf3 |

> 规则层成果：298 测试全绿，63 种兵种，38 种法术，完整攻城系统，7 种能力钩子，
> 英雄技能 + 主属性，士气/运气 d24/d12，Golem 减伤，Bone Dragon 被动。
> 规则正式冻结，不再做规则层工作（6 复杂法术留作独立扩展）。

---

## AI 架构层

### R1 — 可插拔骨架 ✅ 完成

- `ai/base.py`: `AIPlayer` 抽象基类
- `ai/classic/`: 原版 AI 重命名为 `ClassicAI`
- `ai/factory.py`: `create_ai(kind)` + 注册表
- 调用点统一走工厂；`ai/deep/` 占位

### R2+ — 训练脚手架（下一步）

> 规则已冻结，可以开始。保持框架无关（纯 numpy，不依赖 torch/gym）。

- [ ] **R2 观测编码** `ai/observation.py`: `BattleState` → 定长数值向量
- [ ] **R3 动作空间** `ai/action_space.py`: 动作编号 + 合法性掩码 + 编号↔Action 互转
- [ ] **R4 环境适配** `ai/env.py`: `reset/step/reward` 三件套
- [ ] **R5 DeepAI** `ai/deep/`: `DeepAI(AIPlayer)` 注册进工厂，与 ClassicAI 同台对战

---

## AI 决策层对齐 — A 系列

> 目标：29 条 ⚠️ 近似项中，21 条实质修复 + 4 条验证改标 ✅ + 4 条已确认等价改标 ✅
> 覆盖率：87% → ≥95%
> 对照源码：`ai_battle.cpp` + `ai_battle_spell.cpp`

### A1 — 状态分析与威胁评分基础 ✅ 完成

- [x] §1.1-#2：敌军射手排除 `isImmovable`
- [x] §1.1-#5：死亡单位纳入初始部队计数
- [x] §1.1-#8：友军平均速度包含死亡单位的 strength 权重（确认等价）
- [x] §1.2-#10：Archery 技能 → 跳过攻城射箭惩罚
- [x] §8.3-#1/3/5：动态朝向 + `_precompute_enemy_reach` + 路径尾格威胁
- [x] §2.1-#1/§8.1-#2/§5.2-#7：4 项改标 ✅

> 7 项修复 + 4 项改标 + 动态朝向基础，14 新测试，312 全绿，覆盖率 87% → 91%。

### A2 — 法术启发式精度 ✅ 完成

- [x] §3.3-#11：Slow 速度损失动态计算 `lost = speed - max(1, speed-2)`
- [x] §3.3-#14：Slow 非飞行非Haste → 距离衰减 `ReduceEffectivenessByDistance`
- [x] §3.3-#15：Haste 速度增益动态计算 `gained = min(11, speed+2) - speed`
- [x] §3.5-#35：`spellDurationMultiplier`：power<2 且目标已行动 → 乘 0
- [x] §3.5-#36：效果值 × 持续乘数
- [x] §3.3-#20：Bless/Curse min=max 检查（改标 ✅）

> 5 项修复 + 1 项改标，21 新测试，319 全绿，覆盖率 91% → 93%。

### A3 — 移动与目标选择 `planner.py` ✅ 完成

- [x] §4.1-#2：射手撤退评估——`UnitRemover` 模式 + 实际寻路可达性
- [x] §5.1-#4：攻击位置按距当前单位距离排序
- [x] §6.1-#5：被堵射手用 `BestAttackOutcome` 复合优先级
- [x] §8.4-#2：双格攻击方向评估（`splash_value` 枚举攻击方向）
- [x] §9-#2：`getClosestReachablePosition`（改标 ✅ — 等价）
- [x] §9-#5：宽体 `isHandFighting` 碰撞检测（改标 ✅ — 等价）

> 4 项修复 + 2 项改标，18 新测试，337 全绿，覆盖率 93% → ~96%。

### A4 — 狂暴、边缘情况与最终审计

- [ ] §7-#1/5：狂暴精确化
- [ ] §9-#4：`isDisableCastSpell`
- [ ] 最终审计更新 `ai-audit.md`，目标覆盖率 ≥95%
