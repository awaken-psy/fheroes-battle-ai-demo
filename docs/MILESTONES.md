# 里程碑 — 战斗 AI 复刻

> **当前阶段：规则层 + 经典 AI 决策层全部完成，进入 R 系列深度学习训练脚手架。**
>
> - 规则层 M1–M7e：~99% 保真度，63 兵种，38 法术，298 测试
> - AI 决策层 A1–A4：~97% 决策行为覆盖（126 条审计，104✅ / 11⚠️ / 11❌范围外），356 测试
> - AI 架构层 R1：可插拔骨架完成
>
> 详细规则对照见 [`docs/rules-audit.md`](rules-audit.md)（318 项）。
> AI 行为审计见 [`docs/ai-audit.md`](ai-audit.md)（126 条）。

---

## ✅ 已完成

### 规则层 M1–M7e

~99% 保真度，298 测试，63 种兵种，38 种法术，完整攻城系统，7 种能力钩子，
英雄技能 + 主属性，士气/运气 d24/d12。规则正式冻结（6 种复杂法术留作独立扩展）。

### AI 架构层 R1 — 可插拔骨架

`ai/base.py` AIPlayer 抽象基类 · `ai/classic/` ClassicAI · `ai/factory.py` 工厂 + 注册表

### AI 决策层 A1–A4 — 经典 AI 复刻

逐函数比对 fheroes2 C++ 源码（`ai_battle.cpp` + `ai_battle_spell.cpp`），126 条审计项逐条对齐。
覆盖率从 87% 提升至 97%，68 个新测试。11 项范围外全部被规则层缺失（6 种法术/投降）阻塞，
无遗漏。3600 局实战验证全通过。

---

## 📋 下一步 — R 系列深度学习训练脚手架

> **目标**：从零训练一个深度学习 AI（DeepAI），通过自我博弈学会战斗策略，
> 最终在与 ClassicAI 的对战中胜出。
>
> **技术路线**：
> - 观测：Hex Grid 多通道 CNN（11×9 棋盘 ≈ 20-30 通道）
> - 动作：扁平离散 ~10000 + 合法性掩码
> - 决策粒度：三步合一（施法 + 行动一次输出）
> - 训练：PPO + 自我博弈（先打最新自己）
> - 奖励：课程式混合（稠密引导 → 纯稀疏胜负）
> - 框架：PyTorch
> - 评估：定期 vs ClassicAI 对战测胜率

### 依赖关系

```
R2 ──┐
     ├──→ R4 ──┐
R3 ──┤         ├──→ R6 ──→ R7
     └──→ R5 ──┘
```

---

### R2 — 观测编码 `ai/observation.py`

将 `BattleState` 编码为神经网络的输入张量。

**输入**：`BattleState`（hex grid、单位、英雄、效果、攻城状态）

**输出**：
- `grid_tensor`：形状 `(C, 9, 11)`，C ≈ 20-30 通道，每通道是一层 11×9 的 hex 格信息
- `global_vector`：形状 `(G,)`，G ≈ 15-20 维，全局信息（英雄法力、回合数等）

**Grid 通道设计**（初步）：

| 通道组 | 通道 | 内容 |
|--------|------|------|
| team 0 存在 | 0 | 当前单位所在格 (0/1) |
| team 0 HP | 1 | 单位 HP 比例 (0~1) |
| team 0 数量 | 2 | count / 初始 count (0~1) |
| team 0 射手 | 3 | is_archer (0/1) |
| team 0 飞行 | 4 | is_flying (0/1) |
| team 0 宽体 | 5 | is_wide tail 格 (0/1) |
| team 0 攻击力 | 6 | effective_attack 归一化 |
| team 0 防御力 | 7 | effective_defense 归一化 |
| team 1 ×7 | 8-14 | 同上，敌方 |
| 已行动标记 | 15-16 | 双方已行动单位格 |
| 状态效果 | 17-22 | Haste/Slow/Blind/Paralyze/Bless/Curse 等效果层 |
| 攻城 | 23-26 | 城墙 HP / 护城河 / 箭塔 / 可通行 |

**全局向量**：
英雄法力 / 英雄 power / 回合数 / 攻城方 team / 法术列表 one-hot 等

**退出标准**：
- [ ] 给定任意 `BattleState`，输出 `(grid_tensor, global_vector)` 形状正确
- [ ] 通道内容与战场状态一致（单位位置、HP、状态效果）
- [ ] 镜像对称：交换 team 0/1 后张量对应翻转
- [ ] 归一化：所有值在 [0, 1] 或 [-1, 1] 范围内
- [ ] 单元测试覆盖基本场景 + 攻城场景

---

### R3 — 动作空间 `ai/action_space.py`

定义扁平离散动作空间，编号所有可能的行动，生成合法性掩码。

**动作编码方案**：

```
索引 0           → Wait（等待）
索引 1           → Defend（防御）
索引 2-100       → Move 到 hex[0..98]（移动到指定格）
索引 101-9900    → Attack(position, target) = position×99 + target（从某格攻击某目标）
索引 9901+       → Cast(spell_id, target_hex)（施法：法术编号 × 目标格）
```

总计约 ~10000 个动作编号。每步通过 **legality mask**（0/1 数组）过滤非法动作。

**关键功能**：
- `action_to_index(action) → int`：Action 对象 → 编号
- `index_to_action(index, battle) → Action`：编号 → Action 对象
- `legal_mask(battle, unit) → np.array`：当前合法动作掩码
- `enumerate_legal(battle, unit) → List[int]`：所有合法动作编号

**退出标准**：
- [ ] 所有合法动作可正确枚举（移动、攻击、施法、等待/防御）
- [ ] index → Action → index 往返一致
- [ ] 合法性掩码与引擎 `validate_action` 一致
- [ ] 施法动作覆盖全部 38 种法术 × 合法目标
- [ ] 宽体单位攻击位置合法性正确
- [ ] 单元测试覆盖基本场景 + 边界情况

---

### R4 — 环境封装 `ai/env.py`

Gym 风格的战斗环境，提供 `reset/step` 接口，整合观测、动作、奖励。

**核心接口**：

```python
class BattleEnv:
    def reset(preset=None, config=None) -> Observation
    def step(action_index: int) -> (Observation, reward: float, done: bool, info: dict)
```

**决策粒度**：三步合一。每个单位回合调用一次 `step()`，
施法和行动统一在同一个动作空间中（施法是动作编号的一部分）。
如果当前单位对应英雄需先施法再行动，由一步决策完成。

**自我博弈模式**：

```python
class SelfPlayRunner:
    def run_game(agent, opponent_agent) -> trajectory
```

**课程奖励函数**：

```
阶段 1（前 N_selfplay 局）：稠密 + 稀疏
  - damage_dealt / enemy_total_hp   → +δ
  - damage_taken / own_total_hp     → -δ
  - unit_killed                     → +0.1
  - own_unit_killed                 → -0.1
  - 终局胜/负                       → ±1

阶段 2（过渡）：中间奖励权重线性衰减
  - weight = max(0, 1 - progress)

阶段 3（后期）：纯稀疏，只有 ±1
```

**退出标准**：
- [ ] `reset()` 返回合法观测，`step()` 返回合法转移
- [ ] 完整 episode 从开始到结束正常运行
- [ ] 奖励在三个阶段的行为符合设计
- [ ] 自我博弈 runner 能跑完一局并收集 trajectory
- [ ] 与 ClassicAI 对战的 `eval_game()` 可用
- [ ] 单元测试覆盖 episode 生命周期 + 奖励计算

---

### R5 — 神经网络 `ai/deep/model.py`

CNN 骨干 + Policy/Value 双头的 PyTorch 模型。

**网络结构**：

```
输入: grid_tensor (C, 9, 11) + global_vector (G,)
          │
    ┌─────┴─────┐
    │  CNN 骨干  │  4-6 个残差卷积块 (Conv2d → BatchNorm → ReLU → Conv2d + skip)
    │  (共享)    │
    └─────┬─────┘
          │
    ┌─────┴──────┐
    │             │
┌───┴───┐   ┌────┴────┐
│Policy │   │  Value   │
│  Head │   │   Head   │
└───┬───┘   └────┬────┘
    │             │
动作概率     胜率预测
(ACTIONS,)   标量 [-1, 1]
```

- **CNN 骨干**：4-6 层残差块，处理 hex grid 空间结构
- **全局融合**：global_vector 在骨干后拼接
- **Policy Head**：输出 ACTION_DIM 维 logits，乘以 legality mask 后 softmax
- **Value Head**：输出 1 维值（tanh → [-1, 1]）

**退出标准**：
- [ ] forward pass 输出 (policy_logits, value) 形状正确
- [ ] mask 后非法动作概率为零
- [ ] value 输出在 [-1, 1] 范围内
- [ ] 参数量合理（~1-5M），单次 forward < 5ms
- [ ] 可序列化保存/加载
- [ ] 单元测试覆盖形状、mask、边界情况

---

### R6 — PPO 训练器 `ai/deep/trainer.py`

近端策略优化（PPO）训练循环，从自我博弈数据中学习。

**核心组件**：

- **Trajectory Buffer**：存储 (obs, action, reward, value, log_prob, mask) 序列
- **GAE 优势估计**：λ=0.95, γ=0.99，计算 each step 的 advantage
- **PPO Clip 更新**：ε=0.2，多 epoch mini-batch 更新
- **Value Loss**：MSE 回归价值函数
- **Entropy Bonus**：鼓励探索

**训练流程**：
1. 自我博弈收集 N 局 trajectory
2. 计算 GAE advantage
3. PPO 更新（多 epoch × mini-batch）
4. 记录 loss / entropy / value 估计

**退出标准**：
- [ ] 自我博弈数据收集完整，trajectory 格式正确
- [ ] GAE 计算验证（手工构造 case）
- [ ] PPO update 单步跑通，loss 有限不发散
- [ ] 连续训练 100 局，loss 呈下降趋势
- [ ] 梯度裁剪正常，无 NaN/Inf
- [ ] 单元测试覆盖 buffer、GAE、PPO step

---

### R7 — 训练管线 `scripts/train.py`

完整的训练脚本，串联自我博弈、训练、评估、日志。

**核心功能**：

```bash
# 启动训练
python scripts/train.py --episodes 100000 --eval-every 1000 --eval-games 100

# 从 checkpoint 恢复
python scripts/train.py --resume checkpoint.pt
```

**训练循环**：
```
for iteration in range(total):
    1. 自我博弈 N 局（收集 trajectory）
    2. PPO 更新（若干 epoch）
    3. 每 eval_interval 局：
       - vs ClassicAI 打 eval_games 局
       - 记录胜率
       - 保存 checkpoint（模型 + 优化器 + 统计）
    4. 日志输出（console + 可选 tensorboard/wandb）
```

**评估报告**：
- 每 N 局 vs ClassicAI 胜率曲线
- 自我博弈平均回合数
- Policy entropy（策略多样性）
- Value loss / Policy loss

**退出标准**：
- [ ] `python scripts/train.py` 能完整跑通
- [ ] 评估 vs ClassicAI 自动进行并输出胜率
- [ ] Checkpoint 保存和恢复正常
- [ ] 日志清晰可读
- [ ] DeepAI 注册进工厂：`create_ai("deep")` 可用
- [ ] DeepAI 与 ClassicAI 同台对战（arena 兼容）
