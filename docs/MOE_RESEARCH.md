# MoE (Mixture of Experts) 深度调研报告

> 调研日期：2026-06-12
> 目标：为 fheroes-battle-ai-demo 项目选择合适的 MoE 架构，解决灾难性遗忘问题

---

## 1. 项目背景与问题

### 当前状态
- **模型架构**: 13.1M 参数 CNN + Unit-type Embedding (T8)
- **训练方法**: PPO + 经验回放 (T9b)
- **核心问题**: 灾难性遗忘 — 在 `even_clash` 配置上从 100% 胜率骤降至 2.5%
- **硬件限制**: RTX 3070 Laptop (8.2GB VRAM), 训练速度 ~45-53 steps/s

### 为什么选择 MoE
在 5 种反遗忘策略对比中 (CLEAR / EWC / Progressive Networks / MoE / AlphaStar League)：

| 策略 | 优点 | 缺点 | 适合度 |
|------|------|------|--------|
| CLEAR | 简单 | 仅适合同分布数据 | 低 |
| EWC | 即插即用 | 超参敏感，大模型效果差 | 低 |
| Progressive Networks | 零遗忘 | 参数爆炸 (N个任务→N倍参数) | 低 |
| **MoE** | **参数高效、可扩展、可微分** | **需设计路由策略** | **高** |
| AlphaStar League | 最强 | 工程复杂度极高 | 中 |

MoE 在参数效率、可微分训练、与现有 PPO 流程兼容性方面最优。

---

## 2. 核心论文与项目

### 2.1 Soft MoE for Deep RL (Google DeepMind, ICML 2024)

**论文**: *Soft Mixture of Experts for Deep Reinforcement Learning* (arXiv: 2402.08609)

**核心发现**:
- **Soft MoE 在 RL 中完全可行**，且优于标准 MoE (top-k gating)
- 在 Rainbow DQN + 8 experts 下，Atari 平均提升 **~20%**
- 关键：Soft MoE 将 dormant neurons (死神经元) 降至接近零
- PerConv tokenization (每个空间位置作为一个 token) 效果最优

**Soft MoE vs Top-k MoE**:
```
标准 Top-k MoE:
  - 每个 token 只选 top-k 个 expert
  - 需要 load balancing loss 防止路由崩塌
  - 梯度只流过被选中的 expert → 训练不稳定

Soft MoE:
  - 所有 expert 对所有 token 进行加权组合 (softmax over all)
  - 完全可微分，无需 load balancing loss
  - 梯度均匀流过所有 expert → 训练更稳定
```

**对我们项目的意义**: 我们已有 CNN 提取空间特征，Soft MoE 替换倒数第二层全连接即可。

---

### 2.2 M3DT: MoE in Decision Transformer (ICML 2025)

**论文**: *M3DT: Multi-Task Multi-Domain Decision Transformer with Mixture of Experts* (arXiv: 2505.24378)

**核心贡献**:
- 三阶段训练流程 (backbone → experts → router)
- Task grouping: 相似任务共享 expert，减少梯度冲突
- Softmax router over all experts (非 top-k)

**三阶段训练 (关键)**:
```
Stage 1: Backbone Pretraining
  - 在 1-2 个简单任务上训练共享层
  - 目标: 学到通用的状态表示
  - 50K-100K steps

Stage 2: Per-Expert Training
  - 冻结 backbone，每个 expert 独立训练
  - 按 task grouping 分配 expert 到任务组
  - 30K-50K steps per expert

Stage 3: Router Training
  - 冻结 backbone + experts，只训练 router
  - 在所有任务上联合训练
  - 20K-30K steps
```

**对我们项目的意义**: 
- 三阶段训练可以直接迁移到我们的多配置场景
- 每个 battle configuration 对应一个"任务"
- Expert 按"兵种类型"或"阵型特征"分组

---

### 2.3 CP-MoE: Consistency-Preserving MoE (2025)

**论文**: *Continual Learning with Mixture of Experts via Consistency-Preserving Routing* (arXiv: 2605.20247)

**核心贡献**:
- Transient Expert: 临时 expert 用于保护性更新
- CKA-based consistency routing: 基于 CKA 相似度路由
- 近零遗忘 (0.62%) 在 split-CIFAR100 上

**Consistency-Preserving Routing**:
```
1. 计算 CKA(current_features, historical_prototypes)
2. 高相似度 → 路由到已知 expert (保守)
3. 低相似度 → 路由到 transient expert (探索)
4. Transient expert 的更新经蒸馏后合并回主 experts
```

**对我们项目的意义**: 如果三阶段训练后仍有遗忘，可以加入 CP 机制作为后续优化。

---

### 2.4 MOORE: Orthogonal Experts via Gram-Schmidt (ICLR 2024)

**论文/代码**: https://github.com/AhmedMagdyHendawy/MOORE

**核心思想**:
- 通过 Gram-Schmidt 正交化强制 expert 表示多样性
- 防止 expert 退化到相同行为 (mode collapse)
- 在多个 continual learning benchmark 上优于标准 MoE

**对我们项目的意义**: 作为 expert 多样性的正则化手段，如果 expert 出现 mode collapse 可以引入。

---

### 2.5 Soft MoE PyTorch 实现参考

**仓库**: https://github.com/fkodom/soft-mixture-of-experts

**关键代码结构** (简化版):
```python
class SoftMoELayer(nn.Module):
    def __init__(self, dim, num_experts, slots_per_expert):
        self.router = nn.Linear(dim, num_experts * slots_per_expert)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(num_experts)
        ])
        # slots: learnable dispatch/combine weights
        self.slots = nn.Parameter(torch.randn(num_experts, slots_per_expert, dim))
    
    def forward(self, x):
        # x: (batch, tokens, dim)
        # 1. Router computes soft assignment
        # 2. Each expert processes weighted input
        # 3. Weighted combination of expert outputs
        ...
```

---

## 3. 技术对比

### Expert 路由策略对比

| 策略 | 可微分 | 需要负载均衡 | Expert 利用率 | 训练稳定性 |
|------|--------|-------------|--------------|-----------|
| Top-k Gating | 部分 | 是 | 低 (k/N) | 中 |
| **Soft MoE** | **完全** | **否** | **100%** | **高** |
| Hash Routing | 否 | 否 | 固定 | 高 |
| CKA Routing | 部分 | 否 | 自适应 | 中 |

### 与现有架构的集成难度

| 组件 | 集成方式 | 难度 |
|------|---------|------|
| CNN Backbone | 不变，仅替换 penultimate layer | 低 |
| PPO Trainer | 仅需调整 forward pass | 低 |
| Experience Replay | 兼容，replay data 可用于 expert 训练 | 低 |
| Unit-type Embedding | 不变，在 MoE 层之前拼接 | 低 |

---

## 4. 推荐架构设计

### 4.1 模型架构

```
输入: (B, 2, 11, 15) — 双方战场状态
  ↓
Unit-type Embedding (unchanged)
  ↓
Conv Block 1: Conv2d(2+E, 32, 3, padding=1) + BatchNorm + ReLU
  ↓
Conv Block 2: Conv2d(32, 64, 3, padding=1) + BatchNorm + ReLU
  ↓
Conv Block 3: Conv2d(64, 128, 3, padding=1) + BatchNorm + ReLU
  ↓
Flatten → (B, 128*11*15) = (B, 21120)
  ↓
Shared Linear: 21120 → 512 + ReLU     ← Backbone (共享)
  ↓
Soft MoE Layer:
  ├── Router: Linear(512 → num_experts)
  ├── Expert 0: Linear(512, 256) + ReLU  ← 专精: 近战兵种配置
  ├── Expert 1: Linear(512, 256) + ReLU  ← 专精: 远程兵种配置
  ├── Expert 2: Linear(512, 256) + ReLU  ← 专精: 混合配置A
  └── Expert 3: Linear(512, 256) + ReLU  ← 专精: 混合配置B
  → Weighted combination → (B, 256)
  → Linear(256, 512) + ReLU
  ↓
Policy Head: Linear(512, 11) — action logits
Value Head:  Linear(512, 1)  — state value
```

### 4.2 参数量估算

| 组件 | 参数量 |
|------|--------|
| CNN Backbone (unchanged) | ~大部分参数不变 |
| Shared Linear (21120→512) | ~10.8M |
| Router (512→4) | ~2K |
| 4× Expert (512→256) | ~524K |
| Merge Linear (256→512) | ~131K |
| **MoE 新增总计** | **~657K (~2.5MB VRAM)** |

### 4.3 Expert 分组策略

```
Group A: 纯近战 (Swordsman vs Swordsman)
Group B: 近程+远程 (Archers in the mix)
Group C: 强力单位 (Paladins, Crusaders)
Group D: 复杂混合 (Multi-unit compositions)

Router 根据输入状态自动选择加权组合，
不需要手动指定分组——通过 Stage 3 router training 学习。
```

---

## 5. 三阶段训练策略 (适配自 M3DT)

### Stage 1: Backbone Pretraining
```
配置: 1-2 个简单配置 (如 even_clash)
步数: 50K-100K steps
目标: CNN + Shared Linear 学到通用战场特征表示
冻结: 无 (全部训练)
```

### Stage 2: Per-Expert Training
```
配置: 按 expert 分组，每组 1-2 个配置
步数: 每个 expert 30K-50K steps
目标: 每个 expert 专精一类战斗配置
冻结: CNN + Shared Linear (backbone 冻结)
策略: 顺序训练每个 expert，或并行 (如果 VRAM 允许)
```

### Stage 3: Router Training
```
配置: 所有配置混合训练
步数: 20K-30K steps
目标: Router 学会根据输入正确分配 expert 权重
冻结: CNN + Shared Linear + 所有 Experts (只训练 router)
```

### 经验回放集成
- Stage 1: 使用简单配置的 replay data 加速收敛
- Stage 2: 每个 expert 的 replay buffer 独立维护
- Stage 3: 全部 replay data 混合训练 router

---

## 6. 风险分析

### 高风险
1. **Expert Mode Collapse**: 所有 expert 退化为相同行为
   - **缓解**: Soft MoE (加权组合而非 top-k) 已大幅降低此风险
   - **后备**: 引入 MOORE 正交化约束

2. **路由崩塌**: Router 总是分配相同权重
   - **缓解**: 三阶段训练确保 expert 各自有独特初始化
   - **后备**: 添加 entropy bonus 到 router 输出

### 中风险
3. **三阶段训练调度复杂**: 需要精确控制冻结/解冻
   - **缓解**: 分阶段实现，每阶段独立验证
   - **后备**: 简化为两阶段 (backbone+experts 同时训练 → router)

4. **VRAM 增加**: ~2.5MB 额外 VRAM
   - **缓解**: 对于 8.2GB VRAM 完全可接受
   - **后备**: 减少 expert 数量 (4→2)

### 低风险
5. **与现有 PPO 流程兼容性**
   - Soft MoE 完全可微分，不影响 PPO 梯度计算
   - 只需修改 model forward pass，trainer 基本不变

---

## 7. 实施路线图

### Phase 1: Soft MoE 层实现 (T9c)
- [ ] 实现 `SoftMoELayer` nn.Module
- [ ] 替换 model 的 penultimate layer
- [ ] 验证 forward/backward 正确性
- [ ] 单配置训练验证不退化

### Phase 2: 三阶段训练框架 (T9c continued)
- [ ] 实现 freeze/unfreeze 工具函数
- [ ] 实现 Stage 1: backbone pretraining
- [ ] 实现 Stage 2: per-expert training
- [ ] 实现 Stage 3: router training
- [ ] 全配置评估

### Phase 3: 优化与扩展 (T9d, 可选)
- [ ] Expert 分组策略优化 (自动 vs 手动)
- [ ] MOORE 正交化约束 (如果 mode collapse)
- [ ] CP-MoE 持续学习机制 (如果仍有遗忘)
- [ ] Expert 数量调参 (2/4/8)

---

## 8. 参考文献

1. **Soft MoE for Deep RL** — Puigcerver et al. (2024). *Soft Mixture of Experts for Deep Reinforcement Learning*. ICML 2024. [arXiv:2402.08609](https://arxiv.org/abs/2402.08609)

2. **M3DT** — Chen et al. (2025). *M3DT: Multi-Task Multi-Domain Decision Transformer with Mixture of Experts*. ICML 2025. [arXiv:2505.24378](https://arxiv.org/abs/2505.24378)

3. **CP-MoE** — Lee et al. (2025). *Continual Learning with Mixture of Experts via Consistency-Preserving Routing*. [arXiv:2605.20247](https://arxiv.org/abs/2605.20247)

4. **MOORE** — Hendawy et al. (2024). *MOORE: Mixture of Orthogonal Experts for Continual Learning*. ICLR 2024. [GitHub](https://github.com/AhmedMagdyHendawy/MOORE)

5. **Soft MoE PyTorch** — fkodom. *Soft Mixture of Experts in PyTorch*. [GitHub](https://github.com/fkodom/soft-mixture-of-experts)

6. **Soft MoE (Original)** — Puigcerver et al. (2023). *From Sparse to Soft Mixtures of Experts*. ICLR 2024. [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)

---

## 9. 结论

**推荐方案**: Soft MoE + 三阶段训练 (M3DT 流程)

**核心理由**:
1. Soft MoE 在 RL 中已被 Google DeepMind 验证有效 (~20% 提升)
2. 三阶段训练在多任务 DT 中已被 ICML 2025 验证
3. 与现有架构完全兼容，仅需替换一层
4. 参数开销极小 (~657K 新增参数, ~2.5MB VRAM)
5. 训练流程可增量实现，风险可控

**预期效果**:
- 解决灾难性遗忘: 每个 expert 负责一类配置，参数不互相覆盖
- 良好泛化: Soft MoE 的加权组合使模型能处理训练中未见过的配置
- 可扩展: 后续增加新配置只需训练新 expert，不影响已有 expert
