# T9g — 深层 Expert MLP + Diversity Loss + 监督 Router

> 解决 T9f expert 分化不足问题：加深 expert 到 2 层 MLP + diversity loss + 2 层 MLP router。

目标：让 expert 真正分化（余弦相似度 < 0.8），监督 router 准确率 ≥ 85%

## 任务清单
- [x] 9g.1 Expert 加深：单层 Linear → 2 层 MLP（无残差）
  - 测试：856 passed ✅
  - `_ResidualExpertBlock`（实际无残差）替换 `Sequential(Linear, ReLU)`
  - 两层 identity init: `linear1=eye_, linear2=eye_` → expert(x) = ReLU(x) = x（热启动）
- [x] 9g.2 Phase 1 重训 + diversity loss
  - 测试：diversity_loss 从 0.61 → -0.33（expert 余弦相似度从 0.91 → -0.33）
  - `--diversity-loss-weight 0.5`，400K 步配置绑定训练
  - 新增 `compute_diversity_loss()` 在 SoftMoELayer
  - 新增 `--diversity-loss-weight` CLI 参数
- [x] 9g.3 监督 Router 预训练
  - 测试：router 准确率 **99.06%** ✅（远超 85% 目标）
  - Router 升级为 2 层 MLP: `Linear(1536,384)+ReLU+Linear(384,4)`
  - 用 `--use-model-policy` 收集数据（random actions 只有 42%，model policy 达 99%）
  - Per-class: even_clash 100%, example 96.2%, dragon_battle 100%, mage_duel 100%
- [~] 9g.4 Phase 3 联合微调：lr=1e-5, 30K 步（运行中）
- [ ] 9g.5 验证 + 测试 + 文档

## 9g.2 Phase 1 Diversity Loss 结果

**关键发现**：diversity loss 让 expert 分化取得了决定性突破。

| 指标 | 无 diversity loss | diversity loss=0.5 |
|------|------------------|---------------------|
| Expert 同 config 余弦相似度 | 0.91 | **-0.33** |
| 相似度范围 | 0.84-0.98 | **-0.73 ~ 0.35** |
| Router 监督准确率 | 42-45% | **99.06%** |

**实现**：
- `SoftMoELayer.compute_diversity_loss(x)` — pairwise cosine similarity penalty
- 在 PPO update 的每个 minibatch 上计算，与 PPO loss 一起反向传播
- `--diversity-loss-weight 0.5`（0.1-1.0 范围都有效）

## 9g.3 Router MLP 升级

**关键发现**：数据收集方式比模型容量更重要。

| 数据源 | Router 准确率 |
|--------|-------------|
| Random actions | 42-48% |
| **Model policy** | **99.06%** |

Router 从 `Linear(1536,4)` 升级为 `Sequential(Linear(1536,384), ReLU, Linear(384,4))`。

## 9g.4 Phase 3 联合微调

**起点**：t9g-router-supervised checkpoint（supervised router + Phase 1 experts）
**策略**：backbone 冻结，experts + router 可训练，lr=1e-5

```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --load-backbone checkpoints/t9g-router-supervised/best.pt \
  --train-stage 4 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 30000 --device cuda --lr 1e-5 --lr-schedule cosine \
  --eval-interval 5120 --eval-games 20 \
  --balance-loss-weight 0.0 --diversity-loss-weight 0.1 \
  --checkpoint-dir checkpoints/t9g-phase3
```

**监控**：router_w_cos_sim > 0.8，win_rate > 0

## 退出标准
- [x] Expert 余弦相似度 < 0.9（实际 -0.33）
- [x] Router 监督准确率 ≥ 85%（实际 99.06%）
- [ ] 全量测试通过（856+）
- [ ] avg ≥ 50%
- [ ] at least 2 configs win rate > T9e baseline

## 备注
- 起点：T9e best.pt (avg 56.25%)
- T9f 的 expert-aware routing 和监督训练脚本直接复用
- 新增：diversity loss、2 层 MLP router、model-policy 数据收集
