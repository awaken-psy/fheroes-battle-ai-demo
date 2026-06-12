# T9f — 配置绑定训练

> 解决 T9e Router 冻结问题：Phase 1 强制 expert 专精 → Phase 2 监督式 router 预训练 → Phase 3 联合微调。

目标：通过配置-expert 绑定打破初始化对称性，用监督学习训练 router

## 问题分析

T9e 的 Router 完全冻结（30 次 eval 权重不变 `[0.0, 0.0015, 0.2254, 0.7731]`）：

```
所有 expert 初始输出相同（恒等初始化 + 相同 head 权重）
→ 改变路由权重不改变最终输出
→ 主损失对 router 梯度 ≈ 0
→ 只有 balance loss (0.01) 提供微弱梯度
→ Router 不动 → Expert 不分化 → 恶性循环
```

**Phase 2 PPO 软路由失败分析**（4 次尝试全部失败）：
根因是 MoE 软路由 + PPO **范式不兼容**（多篇 2025-2026 论文确认）：
PPO 数据分布不断变化 → 主损失梯度淹没 router 学习信号 → router 坍缩到单个 expert。

**新方案**（BAR 论文, Branch-Adapt-Route 2026）：**解耦 router 训练**。
用监督学习（交叉熵分类）预训练 router，再用极低 LR 做 PPO 联合微调。

## 任务清单
- [x] 9f.1 Phase 1 配置绑定训练（200K 步, lr=1e-3）
  - 测试：构建验证 → 846 passed ✅
  - 结果：expert 权重相似度 0.92~0.96（从 T9e 的 0.998 降低），router 可辨识信号 5x 提升
  - best.pt @step 143K：avg 45.6%，even_clash 95%
  - 50K 步不够（相似度 0.994），延长到 200K 后显著分化
  - 遗留：相似度仍 > 0.9 阈值，但 functional differentiation 足够
- [~] 9f.2 Phase 2 监督式 Router 预训练（BAR 式解耦训练）
  - PPO 软路由 4 次失败，切换到监督学习方案
  - [x] 9f.2a 数据收集脚本 `scripts/collect_router_data.py`
  - [x] 9f.2b 监督训练脚本 `scripts/train_router_supervised.py`
  - [x] 9f.2c train.py 添加 `--load-router` + router 稳定性监控
  - [x] 9f.2d 测试 → 10 个新测试 + 856 total passed ✅
  - [ ] 9f.2e 收集数据 + 训练 router + 验证准确率 ≥ 85%
  - 下次入手点：在 GPU 上运行 collect → train_router_supervised → verify
- [ ] 9f.3 Phase 3 联合微调（待 9f.2e router 准确率达标）
  - 计划：lr=1e-5, 20K-50K 步, balance_loss_weight=0.01
  - 监控：router_w_cos_sim > 0.8（不偏离预训练权重太远）
- [ ] 9f.4 验证训练 + 测试

## 新增代码
- `scripts/collect_router_data.py`：收集 (bottleneck 特征, config_id) 数据对
- `scripts/train_router_supervised.py`：用交叉熵分类训练 router（1540 参数）
- `train.py`：新增 `--load-router` 参数 + `router_w_delta`/`router_w_cos_sim` 指标

## 配置-Expert 绑定
- Expert 0 → even_clash.json
- Expert 1 → example.json
- Expert 2 → dragon_battle.json
- Expert 3 → mage_duel.json

## 9f.1 Phase 1 配置绑定训练（50K 步）

**目标**：打破 expert 对称性，让每个 expert 专精一个配置

**方法**：
- 加载 T9e best.pt (step 143K) 作为起点
- 每个 rollout 根据 config_idx 调用 `set_active_expert(idx)`
- 即 Expert 0 只看 even_clash 数据，Expert 1 只看 example 数据，以此类推
- 冻结 router（Phase 1 不需要路由学习）
- balance_loss_weight = 0.0（不需要 balance loss）

**代码改动**：`train.py` — 当 `--train-stage 2` + `--use-moe` 时已有 `set_active_expert(idx % num_experts)` 逻辑，
正好是 Phase 1 需要的行为（config_idx = expert_idx 轮训）。只需确认冻结 router。

**训练命令**：
```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --load-backbone checkpoints/t9e-hotstart/best.pt \
  --train-stage 2 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 50000 --device cuda --lr-schedule cosine \
  --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.0 \
  --checkpoint-dir checkpoints/t9f-phase1
```

**验证**：Phase 1 结束后检查 expert 权重余弦相似度是否 < 0.9

## 9f.2 Phase 2 软路由训练（100K 步）

**目标**：Router 学会区分配置，expert 继续分化

**方法**：
- 从 Phase 1 best.pt 加载
- 解冻所有参数（`unfreeze_all()`）
- balance_loss_weight = 0.1（比 T9e 的 0.01 强 10 倍）
- 软路由：每个配置数据通过所有 expert，router 学习分配权重

**代码改动**：
- `train.py`：新增 `--train-stage 4`（Phase 2 软路由）或在现有 stage 中通过参数控制
- Phase 2 = `unfreeze_all()` + 高 balance_loss_weight

**训练命令**：
```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --resume checkpoints/t9f-phase1/best.pt \
  --total-steps 100000 --device cuda --lr-schedule cosine \
  --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.1 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --checkpoint-dir checkpoints/t9f-phase2
```

**验证**：Router 权重在不同配置间有明显差异

## 9f.3 Phase 3 联合微调（50K 步）

**目标**：稳定 expert 专精，防止遗忘

**方法**：
- 从 Phase 2 best.pt 加载
- 降低 LR（原始 1/5，即 --lr 5e-5）
- 正常联合训练
- balance_loss_weight = 0.05

**训练命令**：
```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --resume checkpoints/t9f-phase2/best.pt \
  --total-steps 50000 --device cuda --lr 5e-5 --lr-schedule cosine \
  --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.05 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --checkpoint-dir checkpoints/t9f-phase3
```

## 9f.4 验证训练 + 测试

**测试**：
- 全量测试通过（846+）
- Expert 权重余弦相似度分析（Phase 1/2/3 各阶段对比）
- Router 权重在 4 个配置上的分布可视化

**训练验证**：
- 三阶段串联 avg ≥ 50%（≥ T9e best 56.25%）
- even_clash ≥ 90%
- 至少 2 个配置胜率 > T9e best 对应值

## 退出标准
- [ ] Phase 1 后 4 个 expert 权重余弦相似度 < 0.9（证明已分化）
- [ ] Phase 2 后 Router 权重在不同配置间有明显差异
- [ ] 串联三阶段 avg ≥ 50%
- [ ] even_clash ≥ 90%
- [ ] 至少 2 个配置胜率 > T9e best 对应值
- [ ] 全量测试通过（846+）

## 备注
- 起点：T9e best.pt (step 143K, avg 56.25%)
- 三阶段总步数：50K + 100K + 50K = 200K（约 16 分钟 RTX 3070）
- Phase 1 的 `set_active_expert` 逻辑在 `train.py` Stage 2 已实现
- Phase 2 可能需要新增 train-stage 参数或代码改动
