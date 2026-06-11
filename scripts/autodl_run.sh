#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  fheroes-battle-ai-demo — AutoDL 一键部署 + 训练脚本
#
#  使用方法：
#    1. 在 AutoDL 租一张 RTX 4090（PyTorch 2.x 镜像）
#    2. 把 fheroes-battle-ai-demo.tar.gz 上传到 /root/autodl-tmp/
#    3. SSH 进实例后执行：
#       cd /root/autodl-tmp
#       tar xzf fheroes-battle-ai-demo.tar.gz
#       cd fheroes-battle-ai-demo
#       bash scripts/autodl_run.sh
#
#  参数（可选，通过环境变量覆盖）：
#    TOTAL_STEPS    总训练步数       默认 500000
#    NUM_CONFIGS    配置数量 4/16    默认 16
#    EVAL_INTERVAL  评估间隔         默认 10000
#    EVAL_GAMES     评估局数         默认 40
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── 可调参数 ──────────────────────────────────────────────────
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_CONFIGS="${NUM_CONFIGS:-16}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EVAL_GAMES="${EVAL_GAMES:-40}"

# 持久化目录（tar.gz 解压在这里）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="${WORK_DIR}/checkpoints"
LOG_DIR="${WORK_DIR}/logs"

# ── 颜色 ─────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
stage() { echo -e "\n${CYAN}═══ $* ═══${NC}\n"; }

# ── Step 1: 环境检查 ─────────────────────────────────────────
stage "Step 1/4: 环境检查"

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    info "GPU: ${GPU_NAME} (${GPU_MEM})"
else
    warn "未检测到 GPU，将使用 CPU（极慢）"
fi

# Python
PYTHON=$(command -v python3 || command -v python)
PY_VER=$(${PYTHON} --version 2>&1)
info "Python: ${PY_VER}"

# CUDA
if ${PYTHON} -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    CUDA_VER=$(${PYTHON} -c "import torch; print(torch.version.cuda)")
    info "CUDA: ${CUDA_VER} ✓"
    DEVICE="cuda"
else
    warn "PyTorch 未检测到 CUDA，将使用 CPU"
    DEVICE="cpu"
fi

# ── Step 2: 安装依赖 ─────────────────────────────────────────
stage "Step 2/4: 安装依赖"

cd "${WORK_DIR}"

info "安装项目依赖..."
${PYTHON} -m pip install -e . --quiet

info "验证安装..."
${PYTHON} -c "
import torch
from ai.deep.model import BattleNet
m = BattleNet()
print(f'  BattleNet: {m.count_parameters()/1e6:.1f}M params ✓')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ✓')
"

# ── Step 3: 准备训练配置 ─────────────────────────────────────
stage "Step 3/4: 准备训练配置"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# 选择训练配置
if [ "${NUM_CONFIGS}" = "4" ]; then
    CONFIGS="configs/even_clash.json configs/solo_duel.json configs/dragon_battle.json configs/melee_brawl.json"
    info "模式: 4 配置快速验证"
else
    CONFIGS=$(ls configs/*.json | grep -v validation_results | tr '\n' ' ')
    info "模式: 全量配置训练"
fi

TRAIN_CMD="${PYTHON} scripts/train.py \
    --device ${DEVICE} \
    --total-steps ${TOTAL_STEPS} \
    --config ${CONFIGS} \
    --lr-schedule cosine \
    --opponent-pool 10 \
    --tensorboard \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-games ${EVAL_GAMES} \
    --checkpoint-dir ${CKPT_DIR}"

info "训练参数:"
echo "  总步数: ${TOTAL_STEPS}"
echo "  配置数: ${NUM_CONFIGS}"
echo "  评估间隔: ${EVAL_INTERVAL}"
echo "  评估局数: ${EVAL_GAMES}"
echo "  设备: ${DEVICE}"

# ── Step 4: 启动训练 ─────────────────────────────────────────
stage "Step 4/4: 启动训练"

info "预计时间: ~1-2 小时 (RTX 4090, 500K 步, 16 配置)"
info "模型保存至: ${CKPT_DIR}"
info "日志保存至: ${LOG_DIR}"
echo ""

# 同时输出到终端和日志文件
${TRAIN_CMD} 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# ── 训练完成 ─────────────────────────────────────────────────
stage "训练完成!"

info "结果文件:"
echo ""
echo "  最佳模型:    ${CKPT_DIR}/best.pt"
echo "  最终模型:    ${CKPT_DIR}/final.pt"
echo "  TensorBoard: ${WORK_DIR}/runs/"
echo "  训练日志:    ${LOG_DIR}/"
echo ""
info "下载方式: AutoDL 网页端 → JupyterLab → 找到上述文件下载"
