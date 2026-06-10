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
> - 观测：Hex Grid 多通道 CNN（11×9 棋盘，33 通道，player-relative 编码）
> - 动作：扁平离散 ~10000 + 合法性掩码
> - 决策粒度：三步合一（施法 + 行动一次输出）
> - 训练：PPO（CleanRL 实现）+ 自我博弈（先打最新自己）
> - 奖励：课程式混合（稠密引导 → 纯稀疏胜负）
> - 框架：PyTorch + Gymnasium + CleanRL
> - 评估：定期 vs ClassicAI 对战测胜率

### 依赖关系

```
R2 ──┐
     ├──→ R4 ──┐
R3 ──┤         ├──→ R6 ──→ R7
     └──→ R5 ──┘
```

### 外部依赖

```
torch >= 2.0
gymnasium >= 0.29
cleanrl  # PPO 单文件实现，参考/内联
numpy
```

---

### R2 — 观测编码 `ai/observation.py`

将 `BattleState` 编码为神经网络的输入张量。

**输入**：`BattleState` + `current_unit`（当前行动单位）

**输出**：
- `grid_tensor`：形状 `(33, 9, 11)` — 33 通道 hex grid 特征图
- `global_vector`：形状 `(20,)` — 全局标量信息

**Player-Relative 编码**：始终以当前行动方的视角编码。
team 0 行动时"我方"= team 0，team 1 行动时"我方"= team 1。
网络参数共享，数据效率翻倍（AlphaStar 方式）。

**Grid 通道设计**（33 通道）：

| 通道 | 内容 | 值域 |
|------|------|------|
| **我方单位 (0-9)** | | |
| 0 | 我方单位存在（head + tail 格） | 0/1 |
| 1 | 我方 HP 比例 (`_total_hp / _max_total_hp`) | 0~1 |
| 2 | 我方数量比例 (`count / initial_count`) | 0~1 |
| 3 | 我方攻击力 (`effective_attack / 30`) | 0~1 |
| 4 | 我方防御力 (`effective_defense / 30`) | 0~1 |
| 5 | 我方速度 (`speed / 10`) | 0~1 |
| 6 | 我方射手标记 | 0/1 |
| 7 | 我方飞行标记 | 0/1 |
| 8 | 我方宽体尾格（仅 tail 格标 1） | 0/1 |
| 9 | 我方已行动标记 | 0/1 |
| **敌方单位 (10-19)** | | |
| 10-19 | 同上，敌方 | |
| **状态效果 (20-29)** | | |
| 20 | Haste (+speed) | 0/1 |
| 21 | Slow (-speed) | 0/1 |
| 22 | Bless (×1.2 dmg) | 0/1 |
| 23 | Curse (×0.8 dmg) | 0/1 |
| 24 | Blind/Paralyze (skip_turn) | 0/1 |
| 25 | Bloodlust (+3 atk) | 0/1 |
| 26 | Stone Skin / Steel Skin (+def) | 0/1 |
| 27 | Shield (ranged ×0.5) | 0/1 |
| 28 | Anti-Magic | 0/1 |
| 29 | Disrupting Ray 层数 (`stacks / 5`) | 0~1 |
| **攻城 (30-32)** | | |
| 30 | 城墙 HP (`0 / 0.5 / 1`) | 0~1 |
| 31 | 护城河格 | 0/1 |
| 32 | 箭塔存在 | 0/1 |

> 注：效果层和攻城层不做 player-relative（效果属于单位本身，攻城是地形）。
> 宽体单位的属性（HP/数量/攻击等）只在 head 格标值，tail 格只在"存在"通道标 1。

**全局向量**（20 维）：

| 维度 | 内容 | 值域 |
|------|------|------|
| 0 | 回合数 / MAX_ROUNDS | 0~1 |
| 1 | 攻击方 team (0/1) | 0/1 |
| 2 | 我方存活单位数 / 7 | 0~1 |
| 3 | 敌方存活单位数 / 7 | 0~1 |
| 4 | 我方总 HP 比例 | 0~1 |
| 5 | 敌方总 HP 比例 | 0~1 |
| 6 | 我方英雄法力 / max_sp | 0~1 |
| 7 | 敌方英雄法力 / max_sp | 0~1 |
| 8 | 我方英雄 power / 15 | 0~1 |
| 9 | 敌方英雄 power / 15 | 0~1 |
| 10 | 我方英雄 attack / 15 | 0~1 |
| 11 | 敌方英雄 attack / 15 | 0~1 |
| 12 | 我方英雄 defense / 15 | 0~1 |
| 13 | 敌方英雄 defense / 15 | 0~1 |
| 14 | 是否攻城战 | 0/1 |
| 15 | 存活箭塔数 / 3 | 0~1 |
| 16 | 完好城墙数 / 4 | 0~1 |
| 17 | 我方士气 / 3 | -1~1 |
| 18 | 我方运气 / 3 | -1~1 |
| 19 | 当前行动单位在存活队列中的索引 / 14 | 0~1 |

**关键实现细节**：
- `BattleState` 需记录 `_initial_counts: dict` 以计算数量比例
- 无英雄时法力/power 等通道填 0
- 无攻城时攻城通道全填 0
- 效果通道在单位的 occupied_cells 上都标 1

**退出标准**：
- [ ] 给定任意 `BattleState + unit`，输出 `(33, 9, 11)` + `(20,)` 形状正确
- [ ] Player-relative：同一战局 team 0 和 team 1 行动时，张量正确翻转
- [ ] 通道内容与战场状态一致（单位位置、HP、效果）
- [ ] 归一化：所有值在 [-1, 1] 或 [0, 1] 范围内
- [ ] 宽体单位 head/tail 通道分离正确
- [ ] 单元测试覆盖基本场景 + 攻城场景 + player-relative 翻转

---

### R3 — 动作空间 `ai/action_space.py` ✅

定义扁平离散动作空间，编号所有可能的行动，生成合法性掩码。

**动作编码方案**：

```
索引 0           → Wait（等待）
索引 1           → Defend（防御）
索引 2-100       → Move 到 hex[0..98]（移动到指定格）
索引 101-9901    → Attack(position, target) = position×99 + target（从某格攻击某目标）
索引 9902-13564  → Cast(spell[0..36], hex[0..98]) — 37 法术 × 99 目标格（排除 Teleport）
索引 13565       → Retreat（撤退）
```

总计 **13 566** 个动作编号。每步通过 **legality mask**（float32 数组）过滤非法动作。

**设计决策**：
- Wait / Defend 均映射到 `SkipAction`（引擎无 Wait/Defend 语义）
- 远程攻击：position = 射手当前格，仅 `ranged=True` 合法
- Teleport 排除（需双 hex，稀有战术，编码代价过高）
- Mass / 全军法术：所有 99 个 hex 均标合法（执行时忽略 hex）
- Ring AOE：hex = 中心格，全部合法
- Cell 索引：row-major `row × 11 + col` (0-98)

**关键 API**：
- `action_to_index(action, battle, unit) → int`：Action → 编号
- `index_to_action(index, battle, unit) → Action`：编号 → Action
- `legal_mask(battle, unit) → np.ndarray(float32)`：合法性掩码
- `enumerate_legal(battle, unit) → List[int]`：所有合法动作编号

**退出标准**：
- [x] 所有合法动作可正确枚举（移动、攻击、施法、等待/防御）
- [x] index → Action → index 往返一致
- [x] 合法性掩码与引擎路径/攻击/施法逻辑一致
- [x] 施法动作覆盖 37 法术（排除 Teleport）× 合法目标
- [x] 宽体单位攻击位置合法性正确
- [x] 单元测试覆盖基本场景 + 边界情况（53 tests）

---

### R4 — 环境封装 `ai/env.py`

基于 **Gymnasium** 标准接口的战斗环境，整合观测、动作、奖励。

**核心接口**（继承 `gymnasium.Env`）：

```python
import gymnasium
from gymnasium import spaces

class BattleEnv(gymnasium.Env):
    obs_space = spaces.Dict({
        "grid":   spaces.Box(0, 1, shape=(33, 9, 11), dtype=np.float32),
        "global": spaces.Box(-1, 1, shape=(20,), dtype=np.float32),
        "mask":   spaces.Box(0, 1, shape=(ACTION_DIM,), dtype=np.float32),
    })

    def reset(self, *, seed=None, options=None) -> (obs, info)
    def step(self, action_index: int) -> (obs, reward, terminated, truncated, info)
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
- [ ] `reset()` 返回合法观测（符合 gymnasium 空间定义）
- [ ] `step()` 返回合法 `(obs, reward, terminated, truncated, info)` 五元组
- [ ] 完整 episode 从开始到结束正常运行
- [ ] 奖励在三个阶段的行为符合设计
- [ ] 自我博弈 runner 能跑完一局并收集 trajectory
- [ ] 与 ClassicAI 对战的 `eval_game()` 可用
- [ ] 兼容 `gymnasium.make()` 注册
- [ ] 单元测试覆盖 episode 生命周期 + 奖励计算

---

### R5 — 神经网络 `ai/deep/model.py`

CNN 骨干 + Policy/Value 双头的 PyTorch 模型。

**网络结构**：

```
输入: grid_tensor (33, 9, 11) + global_vector (20,)
          │
    ┌─────┴─────┐
    │  CNN 骨干  │  4-6 个残差卷积块 (Conv2d → BatchNorm → ReLU → Conv2d + skip)
    │  (共享)    │
    └─────┬─────┘
          │ + global_vector 拼接
    ┌─────┴──────┐
    │             │
┌───┴───┐   ┌────┴────┐
│Policy │   │  Value   │
│  Head │   │   Head   │
└───┬───┘   └────┬────┘
    │             │
动作概率     胜率预测
(ACTION_DIM,)   标量 [-1, 1]
```

- **CNN 骨干**：4-6 层残差块，处理 hex grid 空间结构
- **全局融合**：global_vector 在骨干后与 CNN 特征展平拼接
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

基于 **CleanRL** PPO 实现的训练循环。

**策略**：参考 CleanRL 的 `ppo_atari.py` 或 `ppo_continuous_action.py` 单文件实现，
根据我们的 `BattleEnv`（Dict obs + mask action）适配，而非从零实现 PPO 算法。
CleanRL 代码简洁可读（~300 行），方便理解和调试。

**核心组件**：

- **Trajectory Buffer**：存储 (obs, action, reward, value, log_prob, mask) 序列
- **GAE 优势估计**：λ=0.95, γ=0.99
- **PPO Clip 更新**：ε=0.2，多 epoch mini-batch 更新
- **Value Loss**：MSE 回归价值函数
- **Entropy Bonus**：鼓励探索
- **Action Masking**：在 policy head 直接 mask 非法动作

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
