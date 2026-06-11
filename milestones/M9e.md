# T9e — MoE 热启动训练

目标：通过 hidden_dim=384 + 恒等初始化 + 共享 head 权重转移，解决 T9d 冷启动问题

## 任务清单
- [x] 9e.1 Expert hidden_dim 升级到 384
- [x] 9e.2 Expert 恒等初始化
- [x] 9e.3 共享 head 权重转移逻辑
- [x] 9e.4 联合训练（backbone 冻结）
- [ ] 9e.5 验证训练 + 测试

## 测试记录
- [x] 9e.1 Expert hidden_dim 升级到 384
  - 测试：单测 test_moe_hidden_dim_384_parameter_count → 参数量 16,262,785 正确
- [x] 9e.2 Expert 恒等初始化
  - 测试：单测 4 项 → 恒等矩阵/非匹配时跳过/非负输入直通/BattleNet 自动应用
- [x] 9e.3 共享 head 权重转移逻辑
  - 测试：单测 3 项 → 权重复制/维度不匹配跳过/转移后所有 expert 输出一致
- [x] 9e.4 联合训练（代码就绪，无新代码改动）
  - 测试：全量 846 测试通过，0 失败
  - 遗留：无

## 9e.1 Expert hidden_dim 升级到 384

**改动文件**: `ai/deep/model.py`

**当前**: expert = Linear(384→128) + ReLU, head = Linear(128→13566)
**目标**: expert = Linear(384→384) + ReLU, head = Linear(384→13566)

**具体改动**:
- `SoftMoELayer.__init__`: experts 改为 `nn.Linear(input_dim, input_dim)` 当 `hidden_dim == input_dim` 时
  - 或直接用传入的 hidden_dim=384
- `SoftMoELayer.__init__`: policy_heads 改为 `nn.Linear(384, action_dim)`
- `SoftMoELayer.__init__`: value_heads 改为 `nn.Linear(384, 1)`

**参数量变化**:
- 删除: 4 × expert(384→128) = 197K, 4 × policy_head(128→13566) = 6.96M, 4 × value_head(128→1) = 516
- 新增: 4 × expert(384→384) = 591K, 4 × policy_head(384→13566) = 20.89M, 4 × value_head(384→1) = 1.54K
- 净增: ~14.3M (15.1M → ~29.4M)

**CLI**: `--moe-hidden-dim 384`（从默认 128 改为 384）

## 9e.2 Expert 恒等初始化

**改动文件**: `ai/deep/model.py`

**原理**: bottleneck 输出已经过 `F.relu(self.fc_bottleneck(x))`，所有特征非负。
恒等矩阵初始化 expert 的 weight → expert(x) = ReLU(W·x + b) = ReLU(x + 0) = x。

**具体改动**:
- `SoftMoELayer.__init__` 中，当 `hidden_dim == input_dim` 时：
  ```python
  nn.init.eye_(self.experts[i][0].weight)  # 384×384 单位矩阵
  nn.init.zeros_(self.experts[i][0].bias)
  ```
- 当 hidden_dim ≠ input_dim 时保持原有 orthogonal 初始化（向后兼容）

**验证**: 初始化后 forward 应与 shared heads 输出近似一致

## 9e.3 共享 head 权重转移逻辑

**改动文件**: `scripts/train.py`（或在 `ai/deep/pipeline.py` 中新增函数）

**逻辑**:
1. 加载 backbone checkpoint（T9b best.pt）
2. 从旧 checkpoint 读取 `policy_head.weight`, `policy_head.bias`, `value_head.weight`, `value_head.bias`
3. 复制到 MoE 的每个 per-expert head：
   ```python
   old_ckpt = torch.load(path)
   for i in range(model.num_experts):
       model.moe.policy_heads[i].weight.data.copy_(old_ckpt['model']['policy_head.weight'])
       model.moe.policy_heads[i].bias.data.copy_(old_ckpt['model']['policy_head.bias'])
       model.moe.value_heads[i].weight.data.copy_(old_ckpt['model']['value_head.weight'])
       model.moe.value_heads[i].bias.data.copy_(old_ckpt['model']['value_head.bias'])
   ```
4. 维度匹配检查：只有当 `hidden_dim == _BOTTLENECK_DIM` 时才执行转移

**位置**: 在 `--load-backbone` 加载完成后执行，仅在 `--use-moe` 且 checkpoint 为非 MoE 时

## 9e.4 联合训练

**不需要代码改动**，只需组合正确的 CLI 参数：

```bash
uv run python scripts/train.py --use-moe --num-experts 4 --routing-topk 2 \
  --moe-hidden-dim 384 \
  --load-backbone checkpoints/t9b-replay/best.pt \
  --train-stage 2 \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 300000 --device cuda --lr-schedule cosine \
  --tensorboard --eval-interval 10240 --eval-games 40 \
  --balance-loss-weight 0.01 \
  --checkpoint-dir checkpoints/t9e-hotstart
```

**注意**: `--train-stage 2` 冻结 backbone 但不冻结 router（不同于 T9d 的 set_active_expert）。
需要验证当前 Stage 2 逻辑是否正确——它只调 `model.freeze_backbone()`，不调 `set_active_expert`。

## 9e.5 验证训练 + 测试

**测试**:
- test_moe.py 参数量测试需要更新（hidden_dim=384 时参数量不同）
- 新增测试：恒等初始化验证（forward 输出与 shared heads 近似）
- 新增测试：权重转移验证（转移后 per-expert head 权重 == old shared head 权重）
- 全量 827 测试通过

**训练验证**:
- 300K 步完成后，各配置胜率对比 T9c baseline
- Router weights 是否分化
- Balance loss 趋势

## 退出标准
- [ ] Expert hidden_dim=384 + 恒等初始化实现
- [ ] 共享 head 权重成功转移到 per-expert heads
- [ ] 4 配置训练 avg ≥ 45%（T9c 为 32.5%）
- [ ] even_clash ≥ 90%（不退化）
- [ ] Router 权重分化，top-2 差距 > 0.10
- [ ] 全量测试通过（827+）
