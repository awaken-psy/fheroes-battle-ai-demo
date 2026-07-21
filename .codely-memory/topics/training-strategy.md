# 训练策略与 RL 决策

Last updated: 2026-07-21 14:22

## 概述

fheroes-battle-ai-demo 项目训练 DeepAI（神经网络）击败 ClassicAI（规则 AI）的训练策略设计。经历了从纯自博弈到混合对手的演进，以及对奖励设计、遗忘对策、配置扩展等维度的系统分析。

## 当前状态

training-v2 分支实现了混合对手策略（33% ClassicAI + 33% 自博弈 + 33% 对手池），200K 步纯 sparse 训练验证完成。结果显示纯 sparse 不可行，计划回退到 dense+sparse 课程。

## 关键知识

### RL 算法选择

- **PPO** 是正确的选择，稳定可控。不是瓶颈。
- DPPO（分布式 PPO）是多 worker 并行版本，当前未使用。
- ER-PPO 是 PPO + 经验回放混合（我们 T9b 的做法），破坏了 on-policy 假设但缓解遗忘。

### 自博弈策略选择

- **纯自博弈**：同一个 policy 打双方，参数共享。容易收敛到互怼策略。
- **FSP（虚构自博弈）**：维护历史版本池，随机采样对手。我们的 FIFO 池是简化版。
- **PFSP（优先级 FSP）**：按胜率优先采样~50%胜率的对手。AlphaStar 用这个。我们未实现。
- **联盟训练**：主 agent + 联盟快照 + 剥削者三类。太复杂。
- **当前选择**：混合对手（ClassicAI + 自博弈 + FIFO 池），33% each。

### 奖励设计实验

- **纯 sparse（±1 胜负）**：200K 步 avg 14%。value_loss 不收敛，长回合 credit assignment 太难。
- **dense→sparse 课程**（T6）：200K 步 avg 57%。dense reward（HP delta）提供中间信号。
- **结论**：纯 sparse 在当前设置下不可行。需要 dense reward 作为 bootstrap。但 dense 可能导致 agent 学"刷伤害"而非"赢"，需要课程衰减。

### 遗忘对策

- **Replay buffer**：T9b 引入但 bug 导致无效。R-refactor 修复后容量 10 不够（~10% rollout）。
- **降低 update_epochs**：4→2 减少过拟合，但学习速度也变慢。
- **MoE**：T9 系列花了 7 个子里程碑，有效但复杂。应该作为最后手段。
- **逐步扩展配置**：T7 一次性 16 配置导致崩溃。应该 4→8→16。

### 200K 步纯 sparse 训练数据分析

- vs ClassicAI：31 局 mean reward -0.074（全负，agent 从未学会打败 ClassicAI）
- 自博弈：mean +0.144（能赢自己）
- 对手池：mean +0.205（打历史版本赢得最多）
- entropy：3.556→3.450（策略在收敛但不够）
- value_loss：0.046→0.047（不收敛，value function 无效）
- 回合长度：example 120 步（太长，sparse credit assignment 困难）

## 待办/下一步

- 回退到 dense+sparse 课程 + ClassicAI 15% + replay 20 + update_epochs 4
- 目标 avg ≥ 57%
- 如果达到，逐步减少 dense 比例做消融实验
- 如果遗忘仍严重，考虑 PFSP（按胜率优先采样历史版本）
