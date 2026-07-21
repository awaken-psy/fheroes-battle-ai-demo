# Next Session

> 记忆保存路径：C:\Users\chang.wei\cw\fheroes-battle-ai-demo\.codely-memory\

Updated: 2026-07-21 14:22

## 上次做了什么

完成了 R 系列管线重构（修复 replay bug + 动作空间缩小 72% + 双阶段施法 + selected unit 通道）和 training-v2 混合对手策略实现。跑了 200K 步纯 sparse 训练，结果不理想（avg ~14% vs T6 baseline 57%），分析发现纯 sparse 在长回合配置上 credit assignment 太难、value function 无法学习、vs ClassicAI 全负。

## 从这里继续

下一步是回退到 dense+sparse 课程重新训练，验证管线改进是否有效：

```bash
uv run python scripts/train.py \
  --config configs/even_clash.json configs/example.json \
          configs/dragon_battle.json configs/mage_duel.json \
  --total-steps 200000 --rollout-steps 2048 \
  --eval-interval 20000 --eval-games 100 \
  --device cuda --lr-schedule cosine \
  --classic-ratio 0.15 --replay-buffer 20 --update-epochs 4 \
  --phase1-steps 10000 --phase2-steps 30000 \
  --checkpoint-dir checkpoints/training-v2-r1
```

目标：4 配置 avg ≥ 57%（匹配 T6），验证 R-refactor 管线改进 + 混合对手是否有效。

## 未完成

- training-v2 分支的 200K 步训练已完成但结果不理想，需要调整参数重跑
- training-v2 分支尚未合并到 main
- CODELY.md 在 .gitignore 中（被排除），不会进入 git

## 注意事项

- 当前在 `feat/training-v2` 分支上，main 已合并了 R-refactor 的所有改动
- PyTorch CUDA 版需要 pyproject.toml 中的 `[[tool.uv.index]]` 配置，不能只用 `uv pip install`
- 测试总数：846（删除了 test_router_supervised.py 的 10 个测试后为 846）
- analyze_train.py 是临时分析脚本，可删除
- .codely-memory/ 和 .codely-cli/ 都在 .gitignore 中

## 相关文件

- `ai/deep/trainer.py` — collect_rollout（混合对手逻辑）、update（replay buffer 修复）
- `scripts/train.py` — CLI 参数默认值、混合对手分配逻辑
- `ai/action_space.py` — 紧凑动作空间编码
- `ai/env.py` — 双阶段行动（cast phase + unit phase）
- `ai/observation.py` — selected unit 通道
- `ai/deep/model.py` — forward 返回 bottleneck 三元组
- `docs/MILESTONES.md` — 完整里程碑记录
