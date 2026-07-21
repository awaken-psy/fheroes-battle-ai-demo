# R 系列管线重构

Last updated: 2026-07-21 14:22

## 概述

fheroes-battle-ai-demo 项目的 RL 训练管线（R 系列）经历了全面重构。原始管线有 4 个严重设计缺陷，导致 T 系列训练困难。重构修复了这些缺陷并验证了改进效果。

## 当前状态

R 系列重构已完成，代码在 `feat/training-v2` 分支上。846 个测试通过，200K 步训练验证完成（纯 sparse 结果不理想，计划回退到 dense+sparse 课程）。

## 关键知识

### 修复的 4 个根本问题

1. **Replay buffer 无效（严重 bug）**：`trainer.py update()` 中 `T = len(self.buffer)` 只含 fresh 数据，minibatch 循环 `range(0, T, ...)` 不覆盖 replay 部分。修复：改为 `total_len = len(grids)` + `randperm(total_len)`。T9b 以来的所有"replay 防遗忘"结论都不可靠。

2. **动作空间过大（13,566 维）**：Attack 子空间 `pos × target = 99 × 99 = 9,801` 维占 72%，但任何时刻合法的只有 ~30 个。重构为 `enemy_index × position = 7 × 8 = 56` 维。ACTION_DIM 13566→3821，policy head 参数 5.23M→1.47M。

3. **英雄施法被压缩为二选一**：env.step() 每次只执行一个 action，agent 选了 Cast 就不能攻击。修复：拆为 cast phase + unit phase，CastAI 和 DeepAI 对等。

4. **当前行动单位标记不足**：CNN 无法定位"哪个单位在行动"。添加第 36 通道（_CH_SELECTED），在当前单位格子上标 1.0。

### 其他修复

- DeepAI 无法加载 MoE checkpoint（从 state_dict 检测 num_experts）
- Windows 路径不兼容（/tmp/ → pytest tmp_path fixture）
- bottleneck 重复计算 3 次（forward 返回 bottleneck 三元组，1 次复用）
- _ResidualExpertBlock 命名误导（→_ExpertMLPBlock）
- moe_hidden_dim 默认值 128→384
- train.py 硬编码 13566 改为 ACTION_DIM

### 训练验证结果

| 配置 | T6 (dense+sparse) | v2 (纯 sparse) |
|------|-------------------|----------------|
| example | 83% | 0% |
| even_clash | 95% | 22% |
| mage_duel | 51% | 100% |
| dragon | 0% | 0-3% |
| avg | 57% | ~14% |

mage_duel 100% 说明双阶段施法有效，但纯 sparse 在长回合配置上 credit assignment 太难。

## 待办/下一步

- 回退到 dense+sparse 课程（phase1=10K, phase2=30K）
- ClassicAI 比例 33%→15%
- replay buffer 10→20, update_epochs 2→4
- 目标：4 配置 avg ≥ 57%（匹配 T6），验证管线改进是否有效
