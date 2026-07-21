#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  T9g 完整复现脚本 — 从零训练到 avg 68.1%
#
#  三阶段：
#    Stage 1: 非 MoE backbone 训练（200K 步）→ 替代 T9a+T9b
#    Stage 2: MoE 热启动训练（300K 步）→ 对应 T9e
#    Stage 3: diversity loss + 配置绑定（400K 步）→ 对应 T9g
#
#  总计 ~900K 步，RTX 4090 约 1.5-2 小时
#
#  用法：
#    bash scripts/reproduce_t9g.sh
#
#  评估：
#    uv run python scripts/eval_benchmark.py \
#        checkpoints/t9g-phase1/best.pt --games 100 --device cuda
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

CONFIGS="configs/even_clash.json configs/example.json configs/dragon_battle.json configs/mage_duel.json"
DEVICE="${DEVICE:-cuda}"
COMMON_ARGS="--device $DEVICE --lr-schedule cosine --config $CONFIGS --eval-games 40"

echo "═══ Stage 1/3: 非 MoE backbone 训练（200K 步）═══"
uv run python scripts/train.py \
    $COMMON_ARGS \
    --total-steps 200000 \
    --rollout-steps 2048 \
    --eval-interval 20000 \
    --opponent-pool 5 \
    --checkpoint-dir checkpoints/backbone

echo ""
echo "═══ Stage 2/3: MoE 热启动训练（300K 步）═══"
uv run python scripts/train.py \
    $COMMON_ARGS \
    --use-moe --num-experts 4 --routing-topk 2 --moe-hidden-dim 384 \
    --load-backbone checkpoints/backbone/best.pt \
    --total-steps 300000 \
    --rollout-steps 2048 \
    --eval-interval 20000 \
    --opponent-pool 5 \
    --checkpoint-dir checkpoints/t9e-hotstart

echo ""
echo "═══ Stage 3/3: T9g diversity loss + 配置绑定（400K 步）═══"
uv run python scripts/train.py \
    $COMMON_ARGS \
    --use-moe --num-experts 4 --routing-topk 2 --moe-hidden-dim 384 \
    --load-backbone checkpoints/t9e-hotstart/best.pt \
    --train-stage 2 \
    --diversity-loss-weight 0.5 \
    --total-steps 400000 \
    --rollout-steps 2048 \
    --eval-interval 20480 \
    --checkpoint-dir checkpoints/t9g-phase1

echo ""
echo "═══ 训练完成，开始评估 ═══"
uv run python scripts/eval_benchmark.py \
    checkpoints/t9g-phase1/best.pt \
    --games 100 --device $DEVICE
