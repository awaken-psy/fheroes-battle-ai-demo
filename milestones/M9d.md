# T9d — MoE 架构打磨 ❌ 训练失败

> 代码正确（827 测试通过），但 per-expert heads 冷启动导致训练失败。
> 详见 T9e 的热启动方案。

目标：解决 T9c 三个结构性问题，提升平均胜率

## 任务清单
- [x] 9d.1 Per-expert policy/value heads
- [x] 9d.2 Top-K 稀疏路由
- [x] 9d.3 Router auxiliary loss (负载均衡)
- [x] 9d.4 训练管线适配
- [x] 9d.5 新增测试
- [ ] 9d.6 验证训练 + 训练报告

## 测试记录
- [x] 9d.1 Per-expert policy/value heads
  - 测试：单测 827 全通过 → policy_heads/value_heads 形状/梯度/freeze 正确
  - 验证：MoE 模式 policy_head=None, value_head=None；per-expert heads 各自独立
- [x] 9d.2 Top-K 稀疏路由
  - 测试：单测 → top_k=2 时非选中权重为0，top_k=4 退化为 soft routing，top_k=1 只选1个
- [x] 9d.3 Router auxiliary loss
  - 测试：单测 → 均匀分布 loss=2.0 为最低，集中分布 loss > 2.0；返回标量
- [x] 9d.4 训练管线适配
  - 测试：单测 827 全通过 → trainer balance_loss 集成、train.py CLI + eval 适配
- [x] 9d.5 新增测试
  - 测试：test_moe.py 重写为 43 个用例（T9c 的 30→43），覆盖 shapes/Top-K/balance/gradient/freeze/compat/params/roundtrip/partial-load
  - 全量 827 测试通过，0 失败

## 9d.1 Per-expert policy/value heads

**改动文件**: `ai/deep/model.py`

**当前架构** (T9c):
```
bottleneck(384) → SoftMoELayer → merge(128→384) → shared policy_head(384→13566) + value_head(384→1)
```

**新架构** (T9d):
```
bottleneck(384) → [expert_i(384→128) → head_i_policy(128→13566) + head_i_value(128→1)] × E
                 → router(384→E) → softmax → weighted sum of logits/values
```

**具体改动**:

SoftMoELayer.__init__:
- 新增 `self.policy_heads = nn.ModuleList([Linear(hidden_dim, action_dim) for _ in range(num_experts)])`
- 新增 `self.value_heads = nn.ModuleList([Linear(hidden_dim, 1) for _ in range(num_experts)])`
- 删除 `self.merge = nn.Linear(hidden_dim, input_dim)` — 不再需要

SoftMoELayer.forward:
- 每个 expert 计算: `feat_i = expert_i(x)` → `logits_i = policy_heads[i](feat_i)` → `value_i = tanh(value_heads[i](feat_i))`
- Router weights 加权合并: `final_logits = Σ w_i * logits_i`, `final_value = Σ w_i * value_i`
- 返回 `(final_logits, final_value, weights)` — 注意：不再返回 384-dim features

SoftMoELayer.__init__ 新增参数: `action_dim: int`

BattleNet.__init__:
- 当 MoE 启用时，heads 移入 SoftMoELayer，BattleNet 不再创建 `self.policy_head` / `self.value_head`
- 保留 no-MoE 路径的 `self.policy_head` / `self.value_head`（向后兼容）

BattleNet.forward:
- MoE 模式: `logits, value, weights = self.moe(x)` → 直接返回（mask 处理在外面）
- 无 MoE: `logits = self.policy_head(x)`, `value = tanh(self.value_head(x))`（不变）

freeze 相关方法适配:
- `set_active_expert(idx)`: 冻结其他 expert + 其他 policy/value heads + router + merge(已删除)
- `freeze_experts_and_merge()`: 冻结所有 expert + 所有 policy/value heads，只留 router
- `freeze_backbone()`: 不变

**参数量变化**:
- 删除: merge(128→384) = 49.5K, shared policy_head(384→13566) = 5.23M, shared value_head(384→1) = 385
- 新增: 4 × policy_head(128→13566) = 6.96M, 4 × value_head(128→1) = 516
- 净增: ~1.73M（13.35M → ~15.08M）
- VRAM 预估: +~7MB，8.2GB 完全够用

**向后兼容**:
- `num_experts=0` 时行为完全不变（使用 BattleNet 自己的 shared heads）
- 旧 checkpoint 加载：load_backbone_weights 只加载 CNN+embedding+bottleneck，per-expert heads 随机初始化

## 9d.2 Top-K 稀疏路由

**改动文件**: `ai/deep/model.py`

**SoftMoELayer 新增参数**: `top_k: int = 2`

**forward 逻辑变更**:
```python
# 1. Router logits
router_logits = self.router(x)  # (B, E)

# 2. Top-K selection
if self.top_k < self.num_experts:
    topk_logits, topk_idx = router_logits.topk(self.top_k, dim=-1)
    topk_weights = F.softmax(topk_logits, dim=-1)  # (B, K)
    
    # 构造稀疏权重矩阵（非 top-K 位置为 0）
    weights = torch.zeros_like(router_logits)
    weights.scatter_(1, topk_idx, topk_weights)
else:
    weights = F.softmax(router_logits, dim=-1)

# 3. 计算 expert outputs（全部计算，通过 weights=0 等效 mask）
# 注：expert 很小(384→128)，计算开销可忽略

# 4. 加权合并（与 9d.1 的 per-expert heads 流程一致）
```

**梯度隔离**:
- `weights=0` 的 expert 对应 head 不贡献梯度（乘以 0）
- `set_active_expert(idx)` 中已经冻结其他 expert，确保 Stage 2 只更新 active expert

**CLI**: `--routing-topk {1,2,4}`，默认 2

## 9d.3 Router auxiliary loss (负载均衡)

**改动文件**: `ai/deep/model.py`, `ai/deep/trainer.py`

**SoftMoELayer 新增方法**:
```python
def balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
    """Switch Transformer 风格负载均衡损失"""
    # f_i: expert i 被 top-K 选中的比例
    topk_idx = router_logits.topk(self.top_k, dim=-1).indices
    selected = torch.zeros_like(router_logits).scatter_(1, topk_idx, 1.0)
    f = selected.mean(dim=0)  # (E,)
    
    # P_i: expert i 的平均 router softmax 概率
    P = F.softmax(router_logits, dim=-1).mean(dim=0)  # (E,)
    
    # balance_loss = E * Σ(f_i * P_i)
    return self.num_experts * (f * P).sum()
```

**forward 返回值新增**: `balance_loss` 作为第四个返回值

**trainer.py 集成**:
```python
# 在 update() 的 forward pass 中：
logits, value_pred, balance_loss = self.model(mb_grid, mb_global, mb_mask)

# 在 total_loss 中：
loss = (policy_loss 
        + self.value_coeff * value_loss 
        - self.entropy_coeff * entropy
        + self.balance_loss_weight * balance_loss)
```

**CLI**: `--balance-loss-weight`, 默认 0.01

## 9d.4 训练管线适配

**改动文件**: `ai/deep/trainer.py`, `scripts/train.py`

**trainer.py**:
- `PPOTrainer.__init__` 新增 `balance_loss_weight: float = 0.0` 参数
- `update()` 中提取 `balance_loss` 并加入 total_loss
- `update()` 返回值新增 `balance_loss` key
- `_select_action()` 适配新的 forward 返回格式（3 个值）
- `collect_rollout()` 中 bootstrap value 的 forward 调用适配

**scripts/train.py**:
- 新增 CLI: `--routing-topk`（默认 2）、`--balance-loss-weight`（默认 0.01）
- `BattleNet` 构造传入 `top_k` 参数
- eval 中 router weight 日志适配（weights 现在从 forward 返回）
- Stage 2/3 freeze 逻辑适配 per-expert heads

## 9d.5 新增测试

**新增/修改文件**: `tests/test_moe.py`（扩展）

**测试用例**:
1. Per-expert heads 独立性：冻结 expert_0 后，expert_0 的 head 梯度为 None，其他 expert 的 head 不受影响
2. Forward 输出形状：logits (B,13566), value (B,1), weights (B,E)
3. Top-K 路由：K=2 时，非 top-K 的权重为 0，top-K 权重和为 1
4. Top-K=4（=num_experts）时退化为 soft routing
5. Balance loss 计算：均匀分布时最小，集中分布时大
6. Balance loss 梯度：仅通过 router 参数
7. 向后兼容：num_experts=0 时输出形状不变
8. No-MoE 模式：不传 action_dim 时行为正确
9. 冻结/解冻：各 freeze 方法的参数 requires_grad 正确性
10. Head 参数初始化：orthogonal init 正确应用

## 9d.6 验证训练

**训练方案**:
```bash
# Stage 2: per-expert training (300K 步)
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --train-stage 2 \
  --load-backbone checkpoints/t9b-replay/best.pt \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 300000 --device cuda --lr-schedule cosine \
  --tensorboard --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.01 \
  --checkpoint-dir checkpoints/t9d-stage2

# Stage 3: router-only training (100K 步)
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --train-stage 3 \
  --resume checkpoints/t9d-stage2/best.pt \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 100000 --device cuda --lr-schedule cosine \
  --tensorboard --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.01 \
  --checkpoint-dir checkpoints/t9d-stage3
```

**对比维度** (T9c vs T9d):
| 指标 | T9c Stage 2 | T9d Stage 2 目标 |
|------|------------|-----------------|
| even_clash 最终 | 95% | ≥95%（不退化） |
| example 最终 | 17.5% | ≥40% |
| dragon 最终 | 15% | ≥30% |
| mage_duel 最终 | 0% | ≥20% |
| 平均胜率 | 32.5% | ≥45% |

**训练报告**: `docs/t9d-training-report.md`

## 退出标准
- [ ] Per-expert heads 实现完整，各 expert 输出互不干扰
- [ ] Top-K 路由实现，K 可配置
- [ ] Router auxiliary loss 实现且可调权重
- [ ] 4 配置训练 avg ≥ 45%（T9c 为 32.5%）
- [ ] 遗忘率 < 10%（T9c 为 15%）
- [ ] Router top-2 权重差距 > 0.15（T9c 为 0.02）
- [ ] 训练报告写入 `docs/t9d-training-report.md`
- [ ] 全量测试通过（850+）
