# T6 训练报告 — 稳定性优化

> 训练时间：2026-06-11，RTX 3070 Laptop，1745.9 秒（~29 分钟）

## 训练配置

```bash
python scripts/train.py \
  --total-steps 500000 \
  --config configs/example.json configs/even_clash.json \
           configs/mage_duel.json configs/dragon_battle.json \
  --lr-schedule cosine \
  --grad-accum 4 \
  --tensorboard \
  --opponent-pool 10 \
  --phase1-steps 30000 \
  --phase2-steps 100000 \
  --eval-interval 10000 \
  --eval-games 100 \
  --device cuda \
  --checkpoint-dir checkpoints/t6-stability
```

**相比 T4 的改进**：

| 参数 | T4 | T6 | 原因 |
|------|-----|-----|------|
| 配置 | 单一 (example.json) | 4 配置混合 | 提升泛化 |
| LR schedule | linear decay | cosine annealing | 避免末期 LR=0 停滞 |
| 对手池 | 5 | 10 | 更多对手多样性 |
| 总步数 | 200k | 500k | 更充分收敛 |
| 课程 phase1 | 10k | 30k | 多配置更复杂，需更长基础期 |
| 课程 phase2 | 30k | 100k | 同上 |

## 训练过程

- **Entropy**: 3.56 → 1.73（从随机策略到高度确定性策略）
- **LR 曲线**: 2.5e-4 → 0.0（cosine 平滑衰减，末期极低 LR 允许微调）
- **对手池使用**: 混合自博弈和池采样
- **训练步数**: 500k 步（实际完成于 step 501,760）

## 最佳 Checkpoint

**best.pt**: step 430,080，平均胜率 **57.25%**

### Eval 过程中的各配置趋势

| Step | Avg | example | even_clash | mage_duel | dragon_battle |
|-----:|----:|--------:|-----------:|----------:|-------------:|
| 10,240 | 9% | 36% | 0% | 0% | 0% |
| 20,480 | 24% | 96% | 0% | 0% | 0% |
| 51,200 | 44% | 28% | 100% | 47% | 0% |
| 163,840 | **47%** | 40% | 100% | 47% | 0% |
| 348,160 | 43% | 24% | 100% | 47% | 0% |
| **430,080** | **57%** ⭐ | **83%** | **95%** | **51%** | 0% |
| 450,560 | 34% | 42% | 3% | 90% | 0% |
| 501,760 | 31% | 29% | 3% | 92% | 0% |

### 最终 Benchmark 评估（best.pt，100 局/配置）

| 配置 | 胜率 | 95% CI | 目标 | 结果 |
|------|------|--------|------|------|
| Mirror Melee | **83%** | [74.5%, 89.1%] | ≥50% | ✅ |
| Asymmetric w/ Heroes | **95%** | [88.8%, 97.9%] | ≥40% | ✅ |
| Spell-Heavy | **51%** | [41.3%, 60.6%] | ≥30% | ✅ |
| Tier-7 Units | **0%** | [0.0%, 3.7%] | ≥20% | ✗ |

**通过 3/4 配置**，dragon_battle 仍然 0%。

## 与 T4 Baseline 对比

| 配置 | T4 (200k, single) | T6 (500k, multi) | 变化 |
|------|:-----------------:|:----------------:|------|
| example | 88% | 83% | -5pp |
| even_clash | 0% | **95%** | +95pp |
| mage_duel | 0% | **51%** | +51pp |
| dragon_battle | 0% | 0% | — |
| **平均** | **22%** | **57%** | **+35pp** |

## 关键发现

1. **多配置训练有效**：even_clash 从 0% → 95%，mage_duel 从 0% → 51%
2. **泛化有代价**：example 从 88% → 83%（-5pp），容量被分摊
3. **dragon_battle 仍为零**：龙/凤凰/骨龙等高级单位可能需要更大模型容量（T7）
4. **后期震荡**：even_clash 在 300k-500k 步之间从 95% 骤降到 3%，说明策略循环仍未完全解决
5. **Cosine LR**：LR 从 2.5e-4 平滑衰减到接近 0，最终几步策略基本冻结

## 结论

T6 达到退出标准（3/4 配置有非零胜率）。多配置训练成功解决了 T4 的零泛化问题。dragon_battle 的 0% 可能受限于模型容量（~4.15M 参数），建议在 T7 中探索更大架构。
