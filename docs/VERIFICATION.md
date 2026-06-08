# 验证清单 — M1–M5

> 逐条复核已完成里程碑的功能是否真正生效。每项给出**验证方式**与**预期结果**。
> 里程碑目标与退出标准见 [MILESTONES.md](MILESTONES.md)。

## 环境准备

```bash
uv sync --group dev          # 安装含 pytest 的开发依赖
```

## 全局检查（一次跑完）

| # | 验证项 | 命令 | 预期 |
|---|---|---|---|
| G1 | 全部单测通过 | `uv run pytest` | **82 passed** |
| G2 | engine / ai 零 pygame 依赖 | `grep -rn "import pygame" ai/ engine/` | 无任何输出 |
| G3 | 无显示器可导入 engine+ai | `uv run pytest tests/test_planner.py::test_engine_and_ai_import_without_pygame` | 子进程 returncode 0 |
| G4 | GUI 可无头跑帧 | `SDL_VIDEODRIVER=dummy uv run main.py`（手动 Ctrl-C）或见 README | 不报错 |
| G5 | CI 绿 | GitHub Actions `Tests` 工作流（push/PR 触发） | pytest + arena smoke 通过 |

测试文件共 12 个、82 项，全部不依赖显示器。

---

## M1 `v0.3` — 验证闭环

| # | 验证项 | 验证方式 | 预期 |
|---|---|---|---|
| M1.1 | hex_grid 解耦 pygame | `tests/test_planner.py::test_engine_and_ai_import_without_pygame`；像素/绘图在 `ui/hex_renderer.py` | 通过；engine 仅纯几何 |
| M1.2 | 伤害拆分 expected/roll | `tests/test_combat.py`（`test_expected_damage_is_deterministic`、`test_roll_damage_*`、`test_calc_damage_is_roll_alias`） | expected 确定、roll 随机、`calc_damage is roll_damage` |
| M1.3 | planner 决策可复现 | `tests/test_planner.py`（5 个快照断言） | 护弓/追击/被贴脸/防御落点动作稳定 |
| M1.4 | arena 批量对战 | `uv run python scripts/arena.py --preset Balanced --games 500 --mirror --seed 0` | 输出胜率 + Wilson 95% CI + 撤退率 |
| M1.5 | 镜像公平（无重大站边偏差） | 同上 | Team0 胜率落 **40–60%**（Balanced ~40–42%） |

> 闭环价值：此环节当初暴露并修掉了 `should_defend` 方向硬编码 bug（team1 永不防守 → 镜像 9.8%）。

---

## M2 `v0.4` — 保真度修正

| # | 验证项 | 验证方式 | 预期 |
|---|---|---|---|
| M2.1 | 战力公式对齐量纲 | `tests/test_combat.py`（`test_base_strength_*`、`test_monster_strength_*`、`test_stack_strength_scales_with_count`） | `(1+0.1atk+0.05def)×base×count`，base=`sqrt(dmg·hp)·special` |
| M2.2 | threat 伤害+距离衰减 | `tests/test_scoring.py` | 射手/飞行 distMod=1；近战超出 speed+1 按 `1.5·dist/speed` 折价 |
| M2.3 | 反僵局撤退 | `tests/test_battle_state.py::test_stalemate_forces_attacker_to_retreat` | 50 回合无死亡 → 进攻方判负 |
| M2.4 | 交替出手 turn order | `tests/test_battle_state.py`（`test_equal_speed_units_alternate_between_teams` 等） | 同速 A,B,A,B；快者优先；first_team 决定平局 |
| M2.5 | 量纲换代后镜像仍公平 | `arena --preset "Archer Defense"/"Flyer Threat" --mirror` | 均落 40–60% |

> 闭环价值：M2 的集火暴露 turn-order「整队先动」不保真（镜像 35/70/79%），改为交替后回到 40–60%。

---

## M3 `v0.5` — 法术系统

| # | 验证项 | 验证方式 | 预期 |
|---|---|---|---|
| M3.1 | 6 法术伤害/效果 | `tests/test_spells.py`（`test_spell_damage_scales_with_power`、Haste/Slow 改速、Bless/Curse 改 damage_factor） | 伤害=base×power；速度/伤害修正生效 |
| M3.2 | buff/debuff 到期失效 | `tests/test_spells.py::test_effects_expire_after_duration` | 持续 power 回合后自动移除 |
| M3.3 | 英雄每回合≤1 法 | `tests/test_spells.py::test_hero_can_cast_once_per_round` | 施法后本回合不可再施，回合重置后恢复 |
| M3.4 | 施法 AI 阈值/折价/ratio | `tests/test_spell_ai.py` | 弱军/占优时不浪费法力；伤害法术选高价值目标；Slow 不打纯射手 |
| M3.5 | 法术对战影响显著 | `arena --preset Balanced --games 400 --seed 0 --hero0` | 带法术方胜率显著（≈100% vs 无法术） |
| M3.6 | 镜像（双方同英雄）仍公平 | `arena --preset Balanced --mirror --hero0 --hero1` | 40–60% |

---

## M4 `v0.6` — 撤退 + 完整回合

| # | 验证项 | 验证方式 | 预期 |
|---|---|---|---|
| M4.1 | 续战阈值 | `tests/test_retreat.py`（`test_hopeless_army_retreats`、`test_retreat_threshold_tracks_difficulty`） | `myStr×ratio < enemyStr` 才撤退；难度系数 Easy/Normal/Hard/Imp |
| M4.2 | 撤退结束战斗 + 告别法术 | `tests/test_retreat.py`（`test_retreat_action_ends_battle_with_loser`、`test_check_retreat_returns_farewell_and_retreat`） | 撤退方判负；告别选伤害法术打敌方 |
| M4.3 | arena 可观测撤退率 | `arena --config configs/example.json --games 200 --hero0 --hero1 --difficulty Impossible` | 撤退率 > 0（约 ~18%） |
| M4.4 | 狂暴 berserkTurn | `tests/test_berserk.py` | 攻击/移向最近单位（不分敌我）；射手射最近 |
| M4.5 | 士气/运气（引擎层） | `tests/test_morale_luck.py` | 运气好×2/坏×0.5；士气好额外/坏跳过；边界正确 |
| M4.6 | **planner 不评估士气运气** | `tests/test_morale_luck.py::test_ai_decision_unaffected_by_morale_and_luck` | 设值前后 planner 决策完全一致 |

---

## M5 `v1.0` — 特殊能力 + 谨慎走位

| # | 验证项 | 验证方式 | 预期 |
|---|---|---|---|
| M5.1 | 无限反击 | `tests/test_abilities.py`（`test_unlimited_retaliation_strikes_every_attacker` vs `test_normal_unit_retaliates_only_once_per_round`） | Griffin 每次被近战都反击；普通单位仅首次 |
| M5.2 | 吸血 hp_drain | `tests/test_abilities.py::test_hp_drain_heals_attacker`、`test_heal_never_resurrects` | 命中回血；不复活（只补残血） |
| M5.3 | 死亡凝视 death_gaze | `tests/test_abilities.py::test_death_gaze_kills_extra` | 命中额外斩杀 |
| M5.4 | 自愈 self_heal | `tests/test_abilities.py::test_self_heal_regenerates_at_round_start` | 回合开始回血 |
| M5.5 | base_strength/threat 能力倍率 | `tests/test_abilities.py`（`test_base_strength_includes_ability_terms`、`test_threat_scaled_by_attacker_abilities`） | 无限反击×1.25、凝视 threat×2 等 |
| M5.6 | 谨慎走位优化落点 | `tests/test_cautious.py` | cautious 时落点选威胁最低且前进（慢敌场景威胁 141→0）；非 cautious 走满 |
| M5.7 | 新兵种可战斗（GUI） | 无头跑含 Vampire/Troll/Medusa 的一局 | 正常分出胜负，不报错 |

---

## 一句话结论

`uv run pytest` 全绿（82）+ 三预设镜像 40–60% + arena 各项指标符合上表，即 M1–M5 验证通过。
当前覆盖原版战斗 AI 约 ~90%；宽体单位/攻城等留待 M5b/M6（见 MILESTONES.md）。
