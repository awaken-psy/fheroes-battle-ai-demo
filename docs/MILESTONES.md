# 里程碑 — 战斗 AI 复刻

> **当前阶段：规则层 + 经典 AI 决策层全部完成，准备进入 R 系列训练脚手架。**
>
> - 规则层 M1–M7e：~99% 保真度，63 兵种，38 法术，298 测试
> - AI 决策层 A1–A4：~97% 决策行为覆盖（126 条审计，104✅ / 11⚠️ / 11❌范围外），356 测试
> - AI 架构层 R1：可插拔骨架完成
>
> 详细规则对照见 [`docs/rules-audit.md`](rules-audit.md)（318 项）。
> AI 行为审计见 [`docs/ai-audit.md`](ai-audit.md)（126 条）。

---

## ✅ 已完成

### 规则层 M1–M7e

~99% 保真度，298 测试，63 种兵种，38 种法术，完整攻城系统，7 种能力钩子，
英雄技能 + 主属性，士气/运气 d24/d12。规则正式冻结（6 种复杂法术留作独立扩展）。

### AI 架构层 R1 — 可插拔骨架

`ai/base.py` AIPlayer 抽象基类 · `ai/classic/` ClassicAI · `ai/factory.py` 工厂 + 注册表

### AI 决策层 A1–A4 — 经典 AI 复刻

逐函数比对 fheroes2 C++ 源码（`ai_battle.cpp` + `ai_battle_spell.cpp`），126 条审计项逐条对齐。
覆盖率从 87% 提升至 97%，68 个新测试。11 项范围外全部被规则层缺失（6 种法术/投降）阻塞，
无遗漏。

---

## 📋 下一步 — R 系列训练脚手架

> 规则层 + 经典 AI 已冻结，可以开始。保持框架无关（纯 numpy，不依赖 torch/gym）。
> 目标：让 DeepAI 与 ClassicAI 同台对战，验证学习效果。

- [ ] **R2 观测编码** `ai/observation.py`: `BattleState` → 定长数值向量
- [ ] **R3 动作空间** `ai/action_space.py`: 动作编号 + 合法性掩码 + 编号↔Action 互转
- [ ] **R4 环境适配** `ai/env.py`: `reset/step/reward` 三件套
- [ ] **R5 DeepAI** `ai/deep/`: `DeepAI(AIPlayer)` 注册进工厂，与 ClassicAI 同台对战
