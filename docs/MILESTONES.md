# 里程碑 — 战斗 AI 复刻

> **T 系列（训练实战）进行中 — T1✅ T2✅ T3✅ T4✅ T5✅ 完成，准备 T6 稳定性优化。**
>
> - 规则层 M1–M7e：~99% 保真度，63 兵种，38 法术，298 测试
> - AI 决策层 A1–A4：~97% 决策行为覆盖（126 条审计），356 测试
> - 深度学习 R1–R7：CNN+PPO 自我博弈训练管线，205 测试
> - **T1 Baseline 评估框架**：eval_benchmark.py + 15 测试，PR #23
> - **T2 模型/训练改进**：GroupNorm + LR decay + Grad accum + TensorBoard，13 测试，PR #25
> - **T3 自博弈对手池**：OpponentPool + 50/50 池采样 + 磁盘持久化，16 测试，PR #27
> - **T4 训练战役**：200k 步，best.pt 胜率 88%（example.json），训练报告已写
> - **T5 多配置训练**：✅ 每 rollout 随机选配置，多配置 eval，平均 best.pt
> - **T6 稳定性优化**：📋 下一步
> - **T6 稳定性优化**：📋
> - **T7 模型升级**：📋（实验性）
> - **总计 622 测试，CI 守护**
>
> 详细规则对照见 [`docs/rules-audit.md`](rules-audit.md)（318 项）。
> AI 行为审计见 [`docs/ai-audit.md`](ai-audit.md)（126 条）。

---

## ✅ 已完成

### 规则层 M1–M7e

~99% 保真度，298 测试，63 种兵种，38 种法术，完整攻城系统，7 种能力钩子，
英雄技能 + 主属性，士气/运气 d24/d12。规则正式冻结（6 种复杂法术留作独立扩展）。

### AI 决策层 A1–A4 — 经典 AI 复刻

逐函数比对 fheroes2 C++ 源码（`ai_battle.cpp` + `ai_battle_spell.cpp`），126 条审计项逐条对齐。
覆盖率从 87% 提升至 97%，68 个新测试。11 项范围外全部被规则层缺失（6 种法术/投降）阻塞，
无遗漏。3600 局实战验证全通过。

### 深度学习 R1–R7 — 训练管线

从零到可运行的 PPO 自我博弈训练管线。参数共享（player-relative 编码），
CleanRL 风格实现，三阶段课程奖励调度。

| 里程碑 | 文件 | 说明 | 测试 |
|--------|------|------|------|
| R1 | `ai/base.py`, `ai/factory.py` | AIPlayer 抽象基类 + 工厂注册 | — |
| R2 | `ai/observation.py` | 33 通道 grid + 20 维全局向量，player-relative | 38 |
| R3 | `ai/action_space.py` | 13566 维扁平离散 + 合法性 mask | 53 |
| R4 | `ai/env.py`, `ai/self_play.py` | Gymnasium BattleEnv + 自博弈 runner | 32 |
| R5 | `ai/deep/model.py` | 4×64 ResBlock CNN + Policy/Value 双头（~4.15M 参数） | 21 |
| R6 | `ai/deep/trainer.py` | TrajectoryBuffer + GAE + PPOTrainer | 30 |
| R7 | `ai/deep/player.py`, `ai/deep/pipeline.py`, `scripts/train.py` | DeepAI + 训练管线 CLI | 31 |

**关键设计决策**（详见源码注释）：
- Player-relative 编码 + 参数共享（AlphaStar 方式）
- 三步合一动作空间（Cast/Move/Attack 统一在 13566 维）
- CleanRL 风格 PPO-Clip（lr=2.5e-4, γ=0.99, λ=0.95, ε=0.2）
- 课程奖励：dense+sparse → 线性衰减 → 纯稀疏
- BattleNet 推理：贪心 argmax（eval）或随机采样（训练）

---

## 📋 T 系列 — 训练实战与优化

> **目标**：让 DeepAI 通过训练在 vs ClassicAI 对战中胜出。
>
> **硬件**：RTX 3070（CUDA）
>
> **评估基准**（Benchmark Suite）— 4 个配置，覆盖从简单到困难：
>
> | 配置 | 阵容 | 特点 | 目标胜率 |
> |------|------|------|----------|
> | `example.json` | 镜像 Swordsman+Archer+Cavalry | 无英雄，最简单 | ≥50% |
> | `even_clash.json` | 非镜像 3v3 + 双英雄 | 中等复杂度 | ≥40% |
> | `mage_duel.json` | 镜像 + 双英雄 + 多法术 | 法术决策考验 | ≥30% |
> | `dragon_battle.json` | 高级兵种（龙/凤凰/骨龙） | 复杂局面 | ≥20% |
>
> 评估脚本 `scripts/eval_benchmark.py` 加载 checkpoint → 对每个配置打 100 局 → 输出胜率表。

### 依赖关系

```
T1(Baseline训练) ──→ T2(模型/训练改进) ──→ T3(对手池) ──→ T4(训练战役)
```

---

### T1 — Baseline 训练 + 评估框架 ✅

用当前模型和超参数跑一次完整训练，建立 baseline 指标。

**新增文件**：
- `scripts/eval_benchmark.py` — 基准评估脚本（加载 checkpoint → 多配置对战 → 输出胜率表）

**训练参数**：
```bash
python scripts/train.py \
  --total-steps 50000 \
  --rollout-steps 2048 \
  --config configs/example.json \
  --eval-interval 10000 \
  --eval-games 20 \
  --device cuda \
  --checkpoint-dir checkpoints/t1-baseline
```

**退出标准**：
- [x] `train.py` 在 CUDA 上正常运行，50k 步完成
- [x] 训练 loss 曲线记录完整（JSON lines 可解析）
- [x] `eval_benchmark.py` 能加载 checkpoint 并输出 4 个配置的胜率
- [x] Baseline 胜率记录到 `results/t1-baseline.json`
- [x] 所有 576 测试通过（原 561 + 新 15）

**实际结果**（PR #23, commit `914d974`）：
- 训练耗时 134 秒（~385 steps/s on RTX 3070）
- Entropy: 3.59 → 3.33（策略在收敛）
- 4 配置 benchmark 评估：全部 0%（预期 baseline，50k 步太短）
- 额外修复：`pipeline.py` 裸列表 JSON、`self_play.py` 法术终结战斗、`dragon_battle.json` 宽体越界

**预期**：初始随机网络 vs ClassicAI 胜率接近 0%。T1 目标不是赢，而是建立可复现的度量基线。

---

### T2 — 模型架构与训练改进 ✅

修复已知问题 + 加入高级训练技巧，为有效训练扫清障碍。

**修改文件**：
- `ai/deep/model.py` — BatchNorm → GroupNorm
- `ai/deep/trainer.py` — LR schedule + gradient accumulation
- `scripts/train.py` — TensorBoard 日志 + 新 CLI 参数
- 新增 `tests/test_model_v2.py` / `tests/test_trainer_v2.py`

**改进项**：

| 改进 | 说明 | 原因 |
|------|------|------|
| GroupNorm | 替换 `BatchNorm2d` 为 `GroupNorm(8, 64)` | BatchNorm 在 batch=1 时统计量不稳定，GroupNorm 不依赖 batch size |
| LR linear decay | 学习率从初始值线性衰减到 0 | PPO 标准做法，避免后期震荡 |
| Gradient accumulation | 累积 N 个 minibatch 再更新 | 等效增大 batch size（GPU 内存有限时有用） |
| TensorBoard | 写入 loss/entropy/eval/win_rate | 可视化训练过程，`tensorboard --logdir runs/` 查看 |

**退出标准**：
- [x] BattleNet 使用 GroupNorm，forward 输出形状和值域不变
- [x] LR schedule 在训练过程中可见衰减（日志可验证）
- [x] Gradient accumulation 等效 batch size 可配置（--grad-accum）
- [x] `tensorboard --logdir runs/` 可显示 loss/entropy/eval 曲线
- [x] 旧 checkpoint 不兼容（GroupNorm 替换后旧权重无法加载，预期行为）
- [x] 新旧测试全部通过（576 + 13 新增 = 589）

**实际结果**（PR #25, commit `50f6186`）：
- GroupNorm: 9 个 GN 层（1 stem + 2×4 ResBlock），零 BN，batch=1 稳定
- LR decay: `lr: 0.00025 → 0.0` 线性衰减可见
- Grad accum: 默认 1（关闭），`--grad-accum 4` 验证通过
- TensorBoard: loss/entropy/lr/eval 曲线写入 `runs/`
- 验证训练：20k 步稳定，无 NaN/Inf

**CLI 新参数**：
```bash
--lr-decay          # 启用线性 LR 衰减（默认关闭）
--grad-accum 4      # gradient accumulation 步数（默认 1）
--tensorboard       # 启用 TensorBoard 日志写入 runs/
```

---

### T3 — 自博弈对手池 ✅

保存历史 checkpoint 作为对手，防止策略坍塌到单一打法。

**新增文件**：
- `ai/deep/opponent_pool.py` — 对手池管理

**修改文件**：
- `ai/deep/trainer.py` — `collect_rollout` 支持 `opponent_model` 参数
- `scripts/train.py` — `--opponent-pool N` CLI + 50/50 自博弈/池采样

**实际实现**（PR #27, commit `85d15f0`）：
- `OpponentPool`：FIFO 淘汰 + 磁盘持久化 + `load_from_disk()` 恢复
- 对手方 transition 不存入 buffer，仅学习方参与 PPO 更新
- 日志 `pool_play: 0.0/1.0` 标识 rollout 来源
- 端到端验证：池空→自博弈，池满后 50% vs 对手

**退出标准**：
- [x] `OpponentPool` 类实现完整（add/sample/容量管理）
- [x] 训练循环集成对手池，日志显示对手来源
- [x] 对手池 checkpoint 持久化到磁盘（重启可恢复）
- [x] 新增测试覆盖池操作（16 个新测试）
- [x] 所有测试通过（605 passed）

---

### T4 — 训练战役 ✅

用全部改进跑一次完整训练，评估 vs ClassicAI 的最终胜率。

**训练参数**（基于 T1 baseline 调优后确定）：
```bash
python scripts/train.py \
  --total-steps 200000 \
  --rollout-steps 2048 \
  --config configs/example.json \
  --eval-interval 10000 \
  --eval-games 100 \
  --device cuda \
  --lr-decay \
  --grad-accum 4 \
  --tensorboard \
  --opponent-pool 5 \
  --checkpoint-dir checkpoints/t4-campaign
```

**实际结果**：
- 训练耗时 488 秒（~8 分钟，RTX 3070）
- Entropy: 3.63 → 2.63（策略在收敛）
- 对手池使用：53.5% 池采样 + 46.5% 自博弈
- best.pt 在 step 51200，example.json 胜率 **88%**
- 单配置训练，未见过的配置 0%（详见训练报告）

**退出标准**：
- [x] 200k 步训练完成，TensorBoard 曲线记录完整
- [x] 最佳 checkpoint 保存为 `checkpoints/t4-campaign/best.pt`
- [x] 训练结果写入 [`docs/t4-training-report.md`](t4-training-report.md)
- [x] README 更新训练结果展示
- [x] 所有测试通过（605 passed）
- ⚠️ Benchmark Suite：example 88% ✅（≥50%），其余 3 配置 0%（单一配置训练未泛化）

**结论**：训练管线完整可用，example.json 大幅超标。未泛化到其他配置——根因是单一配置训练，建议下一步实现多配置混合训练。

---

### T5 — 多配置混合训练 ✅

每局随机选一个配置训练，让 DeepAI 接触英雄/法术/高级兵种，解决零泛化问题。

**修改文件**：
- `scripts/train.py` — `--config` 接受多个文件，每 rollout 随机选一个；eval 遍历所有训练配置
- `ai/deep/trainer.py` — `collect_rollout` 接受可选 `env_config` 参数覆盖默认配置
- 新增 `tests/test_multi_config.py` — 17 个测试

**设计**：
```python
# train.py
p.add_argument("--config", nargs="+", default=None)

# 多配置加载
configs = [load_battle_config(c) for c in args.config]

# 每 rollout 随机选一个
selected_config = random.choice(configs)
info = trainer.train_step(..., env_config=selected_config)

# eval 遍历所有训练配置
for i, cfg in enumerate(configs):
    eval_info = eval_vs_classic(cfg, agent_fn, ...)
```

**退出标准**：
- [x] `--config` 接受多个文件路径
- [x] 每 rollout 随机选择一个配置
- [x] eval 报告每个配置的独立胜率
- [x] `best.pt` 按所有配置平均胜率选择
- [x] 新增测试覆盖多配置选择逻辑（17 个测试）
- [x] 所有测试通过（622 passed）

---

### T6 — 训练稳定性优化

改进超参数和训练策略，减少 eval 胜率震荡，提升最终性能。

**修改文件**：
- `scripts/train.py` — 新增 `--lr-schedule` 参数（linear / cosine）
- `ai/deep/trainer.py` — 支持 cosine annealing LR

**改进项**：

| 改进 | 说明 | 原因 |
|------|------|------|
| Cosine annealing LR | 学习率余弦退火，周期性重启 | 线性衰减末期 LR≈0 导致停滞，cosine 可跳出局部最优 |
| 对手池扩容 | 默认 5→10 或更多 | 更多对手多样性，减少循环主导策略 |
| 课程阶段延长 | phase1 10k→30k，phase2 30k→100k | 多配置训练更复杂，需要更多时间学基础 |
| 更长训练 | 500k+ 步 | T4 的 200k 步仍在震荡，需要更多步数收敛 |

**训练命令**（T5 完成后执行）：
```bash
python scripts/train.py \
  --total-steps 500000 \
  --rollout-steps 2048 \
  --config configs/example.json configs/even_clash.json \
           configs/mage_duel.json configs/dragon_battle.json \
  --eval-interval 10000 \
  --eval-games 100 \
  --device cuda \
  --lr-schedule cosine \
  --grad-accum 4 \
  --tensorboard \
  --opponent-pool 10 \
  --phase1-steps 30000 \
  --phase2-steps 100000 \
  --checkpoint-dir checkpoints/t6-stability
```

**退出标准**：
- [ ] Cosine annealing LR scheduler 选项添加
- [ ] 500k 步混合配置训练完成
- [ ] 4 配置 benchmark 与 T4 baseline 对比（期望 3/4 以上配置有非零胜率）
- [ ] 训练报告写入 `docs/t6-training-report.md`
- [ ] 所有测试通过

---

### T7 — 模型架构升级（实验性）

探索更好的网络架构是否带来性能提升。仅在 T5+T6 完成后、且有明确瓶颈证据时启动。

**修改文件**：
- `ai/deep/model.py` — 新增架构选项（attention / 更深更宽网络）

**候选架构**：

| 方案 | 说明 | 参数量估算 |
|------|------|-----------|
| Attention 增强型 | 在 ResBlock 后加 spatial attention 层 | ~5M |
| 更深网络 | 4→6 ResBlock | ~6M |
| 更宽网络 | 64→128 通道 | ~16M |

**退出标准**：
- [ ] 至少 1 种新架构可选
- [ ] 与 T6 baseline 的消融对比实验完成
- [ ] 最优架构记录在训练报告中
- [ ] 所有测试通过

---

### 可选扩展（未排期）

- [ ] 与原版 fheroes2 C++ AI 决策对照（oracle 一致率）
- [ ] 经典 AI arena 权重调优（threat / 姿态阈值参数搜索）
