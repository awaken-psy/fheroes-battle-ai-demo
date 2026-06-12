# T9g — 深层 Expert 残差 MLP + 重训

> 解决 T9f expert 分化不足问题：加深 expert 到 2 层残差 MLP，重新训练 Phase 1。

目标：让 expert 真正分化（余弦相似度 < 0.9），为监督 router 训练提供可区分特征

## T9f 经验总结

T9f 探索了两条路：
1. **PPO 软路由** → 4 次全部失败（PPO 梯度淹没 router）
2. **BAR 监督 router** → 42% 准确率（expert hidden 特征余弦相似度 0.98-0.99）

**根因**：expert 层太浅（`Linear(384,384)+ReLU` 单层），无法产生有意义的分化。
即使给了 expert hidden 特征给 router，线性分类器也区分不了。

**已完成的代码基础**（T9f 产出，T9g 复用）：
- Expert-aware routing 架构（router 看 expert hidden 拼接）
- `scripts/collect_router_data.py` — 数据收集脚本
- `scripts/train_router_supervised.py` — 监督 router 训练
- `--load-router` + router 稳定性监控

## 任务清单
- [ ] 9g.1 Expert 加深：单层 Linear → 2 层残差 MLP
- [ ] 9g.2 重训 Phase 1：400K 步配置绑定训练（lr=2e-3）
- [ ] 9g.3 监督 Router 预训练：准确率 ≥ 85%
- [ ] 9g.4 Phase 3 联合微调：lr=1e-5, 20K-50K 步
- [ ] 9g.5 验证 + 测试

## 9g.1 Expert 加深

**当前**：
```python
expert_i = Sequential(Linear(384, 384), ReLU)
```

**目标**：
```python
expert_i(x) = x + Linear2(ReLU(Linear1(x)))
# Linear1: identity 初始化（保持热启动）
# Linear2: zero 初始化（初始时 expert = pass-through）
```

**设计要点**：
- 第 1 层 identity 初始化 + 第 2 层 zero 初始化 = 初始时 expert 输出 ≈ 输入
- 残差连接保证热启动不被破坏
- 2 层 MLP 比单层有指数级更强的表达能力
- 参数增加：每个 expert 从 384*384+384=148K 增加到 (384*384+384)*2=296K

**代码改动**：
- `ai/deep/model.py` — `SoftMoELayer.__init__()` 修改 expert 构造
- `ai/deep/model.py` — `init_identity_experts()` 更新初始化逻辑
- 测试更新

## 9g.2 重训 Phase 1

**起点**：T9e best.pt (avg 56.25%)
**配置**：400K 步（T9f 200K 的 2 倍），lr=2e-3（T9f 的 2 倍）
**方法**：配置绑定训练（复用 Stage 2 逻辑）

```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --load-backbone checkpoints/t9e-hotstart/best.pt \
  --train-stage 2 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 400000 --device cuda --lr 2e-3 --lr-schedule cosine \
  --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.0 \
  --checkpoint-dir checkpoints/t9g-phase1
```

**验证**：expert 权重余弦相似度 < 0.9（T9f 只到 0.958）

## 9g.3 监督 Router 预训练

复用 T9f 的脚本：
```bash
uv run python scripts/collect_router_data.py \
    checkpoints/t9g-phase1/best.pt \
    --configs configs/even_clash.json configs/example.json \
              configs/dragon_battle.json configs/mage_duel.json \
    --output data/router_dataset_t9g.pt --device cuda

uv run python scripts/train_router_supervised.py \
    checkpoints/t9g-phase1/best.pt data/router_dataset_t9g.pt \
    --output checkpoints/t9g-phase2-supervised/best.pt \
    --epochs 50 --lr 1e-3 --device cuda
```

**成功标准**：准确率 ≥ 85%，每配置 ≥ 70%

## 9g.4 Phase 3 联合微调

```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --resume checkpoints/t9g-phase2-supervised/best.pt \
  --total-steps 30000 --device cuda --lr 1e-5 --lr-schedule cosine \
  --eval-interval 5120 --eval-games 40 \
  --balance-loss-weight 0.01 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --checkpoint-dir checkpoints/t9g-phase3
```

**监控**：router_w_cos_sim > 0.8，avg ≥ Phase 1 水平

## 9g.5 验证 + 测试

- 全量测试通过（856+）
- Expert 余弦相似度 < 0.9
- Router 监督准确率 ≥ 85%
- avg ≥ 50%（超 T9e best 56.25%）
- even_clash ≥ 90%

## 退出标准
- [ ] Expert 余弦相似度 < 0.9
- [ ] Router 监督准确率 ≥ 85%
- [ ] avg ≥ 50%
- [ ] even_clash ≥ 90%
- [ ] 至少 2 个配置胜率 > T9e best 对应值
- [ ] 全量测试通过（856+）

## 备注
- 起点：T9e best.pt (avg 56.25%)
- 预计训练时间：Phase1 400K ~20min + Phase3 30K ~3min = ~25min RTX 3070
- T9f 的 expert-aware routing 和监督训练脚本直接复用
- 如果 Phase 1 400K 后相似度仍 > 0.9，考虑增加到 800K 或用更大 LR
