# 里程碑 — 战斗 AI 复刻

> **T 系列训练实战 — T1✅ T2✅ T3✅ T4✅ T5✅ T6✅ T7✅，T8 架构升级训练失败(4.4%)，T9a✅ T9b✅(有改善但不够)，T9c MoE架构重构规划中。**
>
> - 规则层 M1–M7e：~99% 保真度，63 兵种，38 法术，298 测试
> - AI 决策层 A1–A4：~97% 决策行为覆盖（126 条审计），356 测试
> - 深度学习 R1–R7：CNN+PPO 自我博弈训练管线，205 测试
> - **T1 Baseline 评估框架**：eval_benchmark.py + 15 测试，PR #23
> - **T2 模型/训练改进**：GroupNorm + LR decay + Grad accum + TensorBoard，13 测试，PR #25
> - **T3 自博弈对手池**：OpponentPool + 50/50 池采样 + 磁盘持久化，16 测试，PR #27
> - **T4 训练战役**：200k 步，best.pt 胜率 88%（example.json），训练报告已写
> - **T5 多配置训练**：每 rollout 随机选配置，多配置 eval，平均 best.pt，PR #31
> - **T6 稳定性优化**：cosine LR + 500k 步 × 4 配置 × pool 10，3/4 benchmark 通过
> - **T7 配置多样化**：16 配置覆盖多维度，修复 dragon_battle 镜像，PR #33
> - **T8 模型架构升级**：13.1M 参数 + 兵种 Embedding，训练失败（4.4% 平均胜率）
> - **T9a 基线验证** ✅：架构有效（even_clash 100%），遗忘严重（40K步内全忘）
> - **T9b 经验回放** ✅：Replay buffer 减缓遗忘（后半段 +65%），但不够（even_clash 仍 100%→2.5%）
> - **T9c MoE架构重构**：Soft MoE + 三阶段训练，解决灾难性遗忘，规划中
> - **T9d MoE扩展配置训练**：4→16配置扩展 + expert数量调参，待T9c完成后启动
> - **总计 781 测试，CI 守护**
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
| R2 | `ai/observation.py` | 33→35 通道 grid + 20 维全局向量，player-relative | 38 |
| R3 | `ai/action_space.py` | 13566 维扁平离散 + 合法性 mask | 53 |
| R4 | `ai/env.py`, `ai/self_play.py` | Gymnasium BattleEnv + 自博弈 runner | 32 |
| R5 | `ai/deep/model.py` | 6×128 ResBlock CNN + Policy/Value 双头 + 兵种 Embedding（~13.1M 参数） | 21 |
| R6 | `ai/deep/trainer.py` | TrajectoryBuffer + GAE + PPOTrainer | 30 |
| R7 | `ai/deep/player.py`, `ai/deep/pipeline.py`, `scripts/train.py` | DeepAI + 训练管线 CLI | 31 |

**关键设计决策**（详见源码注释）：
- Player-relative 编码 + 参数共享（AlphaStar 方式）
- 三步合一动作空间（Cast/Move/Attack 统一在 13566 维）
- CleanRL 风格 PPO-Clip（lr=2.5e-4, γ=0.99, λ=0.95, ε=0.2）
- 课程奖励：dense+sparse → 线性衰减 → 纯稀疏
- BattleNet 推理：贪心 argmax（eval）或随机采样（训练）
- 兵种语义 Embedding（T8 新增）：Embedding(67, 16, padding_idx=0)

---

## 📋 T 系列 — 训练实战与优化

> **目标**：让 DeepAI 通过训练在 vs ClassicAI 对战中胜出。
>
> **硬件**：RTX 3070 Laptop（CUDA）/ AutoDL RTX 4090（云端训练）
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
                                                          │
T5(多配置训练) ──→ T6(稳定性优化) ──→ T7(配置多样化) ──→ T8(架构升级) ──→ T9(课程学习+经验回放)
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

**退出标准**：
- [x] `--config` 接受多个文件路径
- [x] 每 rollout 随机选择一个配置
- [x] eval 报告每个配置的独立胜率
- [x] `best.pt` 按所有配置平均胜率选择
- [x] 新增测试覆盖多配置选择逻辑（17 个测试）
- [x] 所有测试通过（622 passed）

---

### T6 — 训练稳定性优化 ✅

改进超参数和训练策略，减少 eval 胜率震荡，提升最终性能。

**修改文件**：
- `scripts/train.py` — `--lr-decay` 替换为 `--lr-schedule {none,linear,cosine}`
- 新增 `tests/test_lr_schedule.py` — 11 个测试

**训练结果**：
- 500k 步，4 配置混合训练，1745.9 秒（RTX 3070）
- Cosine LR: 2.5e-4 → 0.0 平滑衰减
- Entropy: 3.56 → 1.73
- best.pt (step 430,080): 平均胜率 **57%**
  - example 83% ✅ | even_clash 95% ✅ | mage_duel 51% ✅ | dragon_battle 0% ✗
- 训练报告：[`docs/t6-training-report.md`](t6-training-report.md)

**退出标准**：
- [x] Cosine annealing LR scheduler 选项添加
- [x] 500k 步混合配置训练完成
- [x] 4 配置 benchmark 与 T4 baseline 对比（3/4 配置有非零胜率 ✅）
- [x] 训练报告写入 `docs/t6-training-report.md`
- [x] 所有测试通过（633 passed）

---

### T7 — 配置多样化训练 ✅

> **根因分析**：T6 中 dragon_battle 0% 胜率并非模型容量不足，而是阵容严重不平衡
> （Team 0 总 HP 750 vs Team 1 总 HP 1190，相差 58.7%）。
> 同时，仅 4 个配置导致泛化维度不足。本里程碑通过 16 个平衡配置覆盖战斗的多维度。

16 个平衡训练配置，覆盖英雄/无英雄、飞行/步行、宽体/单格、远程/近战、法术/无法术、
镜像/非镜像、不同规模（1v1 到大规模混战）。

**实际结果**（PR #33）：
- 16 配置训练 1M 步
- 平均胜率 **6.6%**，仅 3 种配置有意义胜率（even_clash 52%, solo_duel 38%, dragon_battle 28%）
- 灾难性遗忘：850K 步 even_clash=52% → 1M 步降到 20%
- 根因：192 维 bottleneck 容量不足 + 无兵种信息 + 16 配置互相干扰

**退出标准**：
- [x] 16 个配置 JSON 文件创建完成（3 保留 + 1 修复 + 12 新增）
- [x] `dragon_battle.json` 修复为镜像平衡阵容
- [x] 镜像配置双方阵容完全一致；非镜像配置总 HP 差异 < 10%
- [x] `eval_benchmark.py` 支持自动发现 `configs/` 下所有 JSON
- [x] 每个配置均可正常运行完整战斗（100 局验证无崩溃）
- [x] 新增配置平衡性测试通过
- [x] 所有已有测试通过

---

### T8 — 模型架构升级 ❌ 训练失败

> **结论**：架构代码正确（781 测试通过），但训练策略无法应对多配置灾难性遗忘。

针对 T7 暴露的容量瓶颈和兵种信息缺失问题，升级模型架构。

**修改文件**（分支 `feat/t8-model-upgrade`，commit `278376a`）：
- `ai/deep/model.py` — 128ch × 6 ResBlock × 384 bottleneck + Embedding(67, 16)
- `ai/observation.py` — 33→35 通道（+2 兵种类型通道）+ 宽体越界修复
- 新增 13 个 embedding/type 测试

**架构变更**：

| 参数 | T7 (旧) | T8 (新) | 变化 |
|------|---------|---------|------|
| CNN 通道 | 64 | **128** | 2× |
| ResBlock | 4 | **6** | +2 |
| Bottleneck | 192 | **384** | 2× |
| 兵种 Embedding | 无 | **Embedding(67, 16)** | 新增 |
| 观测通道 | 33 | **35** | +2 |
| 总参数量 | 4.15M | **13.1M** | 3.2× |

**训练结果**（AutoDL RTX 4090, 500K 步）：

| 指标 | T8 实际 | T7 对比 | T6 基线 |
|------|---------|---------|---------|
| 平均胜率 | **4.4%** | 6.6% | 57% |
| 实际配置数 | **22**（脚本 bug） | 16 | 4 |
| 峰值平均胜率 | 18.5%（30K 步） | — | — |

**灾难性遗忘证据**：

| 配置 | 历史峰值 | 峰值步数 | 最终胜率 |
|------|---------|---------|---------|
| even_clash | 100% | 51K | **0%** |
| ranged_fest | 100% | 10K | 7.5% |
| flyer_swarm | 90% | 41K | **0%** |
| dragon_battle | 62.5% | 92K | **0%** |
| solo_duel | 47.5% | 164K | 47.5%（唯一稳定） |

**失败根因**：
1. **脚本 bug**：`autodl_run.sh` 用 `ls *.json | grep -v validation_results` 匹配了 22 个配置（含 example、ability_showcase 等），多于预期的 16 个
2. **根因不是容量**：参数 3 倍后反而更差（4.4% vs T7 的 6.6%），问题在于纯在线 PPO 的学习范式
3. **灾难性遗忘**：模型顺序学习不同配置，学新的就忘旧的，无保留机制
4. **Cosine LR 过早衰减**：250K 步时 LR 已降至 50%，后半段几乎不学习

**退出标准**（部分达成）：
- [x] 新架构可选（128ch/6ResBlock/384bottleneck/Embedding）
- [x] 所有测试通过（781 passed，+13 新增）
- [x] 架构代码变更量可控（7 文件，+342/-61 行）
- [ ] ~~与 T7 baseline 消融对比~~（训练失败，无法对比）
- [ ] ~~最优架构记录在训练报告中~~（架构 OK，训练策略不行）

**T8 训练数据**：`checkpoints/t8-upgrade/`, `logs/t8-upgrade/`, `runs/t8-upgrade/`

---

### T9 — 对抗灾难性遗忘

> **核心问题**：T7/T8 反复出现的灾难性遗忘证明，单纯加参数量或经验回放都不够。
> 需要从架构层面引入 MoE 机制，让不同 expert 专精不同配置，从根本上避免参数覆盖。

分 3 个子阶段增量推进，每个阶段有独立退出标准。

#### T9a — 基线验证 ✅

**目标**：确认 T8 架构在少量配置下有效（排除架构问题）

**训练参数**（与 T6 完全一致，苹果对苹果）：
- 4 配置：example + even_clash + mage_duel + dragon_battle
- 500K 步，cosine LR，opponent pool 10，eval 40 局/次
- 实际跑到 246K 步（结论已清楚，提前停止）

**结果**：

| 配置 | 峰值胜率 | 峰值步数 | 最终胜率(245K) |
|------|---------|---------|---------------|
| even_clash | **100%** | 10K | 0% 💀 |
| mage_duel | **60%** | 246K | 60% ✅ |
| example | **37.5%** | 133K | 0% 💀 |
| dragon_battle | **17.5%** | 225K | — |

**训练速度**：53 steps/s（T6 约 287 steps/s，T8 大模型慢 5.4×）

**灾难性遗忘时间线**：
- Step 195K：even_clash=95% + mage_duel=45%（avg **35%**）← 本轮最佳
- Step 205K：avg **1.2%**（40K 步内全部遗忘）💀
- Step 225K：dragon_battle=17.5%（新配置起来了，其余仍 0%）
- Step 235K：avg **0.6%**（几乎全灭）

**结论**：
- ✅ **架构有效**：模型能学会单配置到很高胜率（even_clash 100%, mage_duel 60%）
- ✅ **排除架构问题**：13.1M 参数量足够，不需要更大模型
- ❌ **灾难性遗忘严重**：4 配置下就已出现，与 T7/T8 的 16/22 配置问题一致
- ❌ **大模型反而更容易遗忘**：T6（4.15M）4 配置 57% vs T9a（13.1M）4 配置波动 0-35%

**退出标准**（结论性达成）：
- [x] 4 配置训练完成（246K 步，提前停止）
- [x] 架构有效性确认（even_clash 峰值 100%）
- [ ] ~~4 配置平均胜率 ≥ 50%~~（因遗忘未达到，但目的已达成）
- [ ] ~~所有测试通过~~（无代码改动）

**关键发现**：遗忘是纯在线 PPO 的根本问题，不是配置数或模型大小的错。需要引入经验回放（T9b）来保留已学策略。

**数据**：`checkpoints/t9a-baseline/`, `logs/t9a-baseline/`

---

#### T9b — 经验回放缓冲区 ✅ 有改善但不够

**目标**：引入防遗忘机制，让模型保留已学配置的策略

**新增文件**：
- `ai/deep/replay_buffer.py` — 环形经验回放缓冲区（~55 行）

**修改文件**：
- `ai/deep/trainer.py` — `update()` 集成回放数据混合（~45 行改动）
- `scripts/train.py` — `--replay-buffer N` CLI 参数（~15 行改动）

**新增测试**：
- `tests/test_replay_buffer.py` — 14 个测试（ReplayBuffer + PPOTrainer 集成）

**设计**：
```python
# ReplayBuffer: 环形缓冲区，存完整 rollout，FIFO 淘汰
class ReplayBuffer:
    def __init__(self, capacity=10): ...
    def add(self, rollout: Dict[str, Tensor]): ...
    def sample(self) -> Dict[str, Tensor]: ...

# PPOTrainer.update() 集成
# 1. 对旧观测 forward pass → fresh values（给 GAE 用）
# 2. 保留旧 log_probs 作为 PPO ratio 的 old 项
# 3. 拼接新旧数据做 PPO update
```

**训练结果**（4 配置, 200K 步, replay buffer=10）：

| Step | avg | even_clash | example | dragon_battle | mage_duel |
|------|-----|-----------|---------|---------------|-----------|
| 10K | 38.8% | 100% | 10% | 45% | 0% |
| 40K | 34.4% | 100% | 27.5% | 10% | 0% |
| 70K | 1.2% | 0% | 0% | 5% | 0% |
| 100K | 0.6% | 0% | 0% | 2.5% | 0% |
| 143K | 22.5% | 20% | 52.5% | 7.5% | 10% |
| 184K | 18.1% | 2.5% | 55% | 15% | 0% |
| **结尾** | **18.1%** | **2.5%** | **55%** | **17.5%** | **0%** |

**与 T9a（无 replay）对比**：

| 指标 | T9a | T9b | 变化 |
|------|-----|-----|------|
| 后半段(140K+) avg | ~10% | **~16.5%** | +65% ↑ |
| 结尾 avg | 0.6%（崩了） | **18.1%** | 大幅改善 |
| 多配置同时有胜率 | 1-2 个 | **3 个** | 更分散 |
| 训练速度 | 53 steps/s | 45 steps/s | -15% |

**退出标准**（部分达成）：
- [x] `ReplayBuffer` 类实现完整（add/sample/FIFO 淘汰/容量管理）
- [x] `PPOTrainer.update()` 支持混合新旧数据训练
- [x] 新增 replay buffer 测试通过（14 个测试）
- [x] 全量测试通过（795 passed）
- [ ] ~~8 配置 300K 步~~（改为 4 配置 200K 步快速验证）
- [ ] ~~平均胜率 ≥ 30%~~（实际结尾 18.1%，未达标）
- [ ] ~~无"学后忘"~~（even_clash 100% → 2.5%，仍然遗忘）

**结论**：Replay 确实**减缓了遗忘**（后半段 avg 16.5% vs T9a 10%），但 10 个 rollout 的缓冲区不够大，even_clash 仍从 100% 跌到 2.5%。需要配合课程学习（T9c）或增大 replay 容量。

**数据**：`checkpoints/t9b-replay/`, `logs/t9b-replay/`

---

#### T9c — MoE 架构重构

> **方向转变**：MoE 深度调研（5 篇论文）后决定采用 Soft MoE + 三阶段训练。
> 详细调研报告见 [`docs/MOE_RESEARCH.md`](MOE_RESEARCH.md)。

**目标**：用 Mixture of Experts 架构让不同 expert 专精不同配置，从根本上解决灾难性遗忘

**核心论文**：
- Soft MoE for Deep RL (Google DeepMind, ICML 2024) — RL 中 +20% 提升
- M3DT (ICML 2025) — 三阶段训练框架（backbone → experts → router）

**架构设计**：

```
CNN Backbone (shared, 不变)
    ↓
Shared Linear(21120, 512) + ReLU    ← backbone
    ↓
Soft MoE Layer:
  ├── Router: Linear(512 → num_experts)
  ├── Expert 0-3: Linear(512, 256) + ReLU each
  └── Weighted combination → Linear(256, 512)
    ↓
Policy Head: Linear(512, 11)  ← action logits
Value Head:  Linear(512, 1)   ← state value
```

**三阶段训练**（适配自 M3DT）：

| 阶段 | 冻结 | 训练 | 配置 | 步数 |
|------|------|------|------|------|
| Stage 1: Backbone | 无 | 全部 | 1-2 简单配置 | 50K-100K |
| Stage 2: Experts | CNN + Shared Linear | 4 个 Expert | 按 expert 分组 | 每个 30K-50K |
| Stage 3: Router | 全部 Expert | Router only | 所有配置混合 | 20K-30K |

**参数量**：新增 ~657K 参数 (~2.5MB VRAM)，总模型约 13.8M

**修改文件**：
- `ai/deep/model.py` — 新增 `SoftMoELayer` nn.Module，修改 `BattleNet`
- `ai/deep/trainer.py` — 三阶段 freeze/unfreeze 逻辑
- `scripts/train.py` — `--train-stage {1,2,3}` CLI 参数
- 新增 `tests/test_moe.py` — MoE 层单元测试

**退出标准**：
- [ ] `SoftMoELayer` 实现完整（forward/backward 可微分）
- [ ] `BattleNet` 集成 MoE 层，单配置 forward pass 正确
- [ ] 三阶段训练 CLI 支持冻结/解冻指定层
- [ ] Stage 1: backbone 预训练，单配置胜率 ≥ 50%
- [ ] Stage 2: per-expert 训练，每个 expert 有独立专精
- [ ] Stage 3: router 训练，多配置平均胜率 ≥ 50%
- [ ] 灾难性遗忘率 < 20%（已学配置胜率下降不超过 20%）
- [ ] 新增 MoE 测试通过（10+ 个）
- [ ] 全量测试通过（800+）
- [ ] 训练报告写入 `docs/t9c-training-report.md`

---

#### T9d — MoE 扩展配置训练

> **前置条件**：T9c 三阶段训练完成，MoE 架构验证有效。
> 目标是将 MoE 从 4 配置扩展到 8-16 配置，并优化 expert 路由质量。

**目标**：在 T9c 基础上扩展到全部 16 配置，验证 MoE 的可扩展性

**核心思路**：
- T9c 证明 MoE 有效后，本阶段扩大配置覆盖面
- 可能需要增加 expert 数量（4→6 或 4→8）以覆盖更多配置类型
- 引入经验回放 + MoE 联合训练，双重防遗忘

**训练策略**：

| 阶段 | 配置数 | Expert 数 | 训练方式 | 步数 |
|------|--------|----------|---------|------|
| 扩展 1 | 4→8 | 4 (不变) | 加载 T9c checkpoint，继续三阶段 | 150K-200K |
| 扩展 2 | 8→16 | 6-8 (增加) | 重新三阶段训练，旧 expert 可选冻结 | 200K-300K |
| 联合优化 | 16 | 最终数量 | 全配置 + replay buffer 联合训练 | 100K-200K |

**修改文件**：
- `ai/deep/model.py` — `num_experts` 可配置化
- `scripts/train.py` — `--load-checkpoint` + `--train-stage` 联合支持
- `ai/deep/trainer.py` — replay buffer 与 MoE 集成（replay 数据也过 MoE 路由）

**退出标准**：
- [ ] `num_experts` 可通过 CLI 参数配置（默认 4）
- [ ] 从 T9c checkpoint 加载并继续训练
- [ ] 8 配置扩展训练完成，平均胜率 ≥ 45%
- [ ] 16 配置扩展训练完成，平均胜率 ≥ 40%
- [ ] 灾难性遗忘率 < 15%（已学配置胜率下降不超过 15%）
- [ ] Replay + MoE 联合训练验证有效
- [ ] 训练报告写入 `docs/t9d-training-report.md`
- [ ] 全量测试通过（810+）

---

### T10+ — 可选扩展（未排期）

- [ ] Expert 分组策略优化（自动 vs 手动分组）
- [ ] Expert 数量调参（2/4/8 对比）
- [ ] MOORE 正交化约束（如果 expert mode collapse）
- [ ] CP-MoE 持续学习机制（如果仍有遗忘）
- [ ] 自博弈对手策略多样化（不同风格对手）
- [ ] 更长训练（1M 步）
- [ ] 与原版 fheroes2 C++ AI 决策对照（oracle 一致率）
- [ ] 挑战人类玩家对战
