#!/usr/bin/env python3
"""可视化所有训练历史，对比 T4/T5/T6/T7 的训练效果。

用法：
    python scripts/plot_history.py          # 生成 PNG 图片
    python scripts/plot_history.py --show   # 弹出窗口显示
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing import event_accumulator

# ── 训练运行元数据 ──────────────────────────────────────────────
# 文件名 → (标签, 颜色)
RUNS = {
    "events.out.tfevents.1781114203.lingo.826840.0": ("T4 单配置 200K", "#2196F3"),
    "events.out.tfevents.1781115974.lingo.844074.0": ("T5 多配置 200K", "#4CAF50"),
    "events.out.tfevents.1781119622.lingo.874802.0": ("T6 稳定性 500K", "#FF9800"),
    "events.out.tfevents.1781123563.lingo.904573.0": ("T7 多样化 1M", "#F44336"),
}

RUNS_DIR = "runs"
OUT_DIR = "plots"


def load_scalar(event_path, tag):
    """从 TensorBoard event 文件加载某个 tag 的 (step, value) 数据。"""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def smooth(values, weight=0.6):
    """指数移动平均平滑。"""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(result[-1] * weight + v * (1 - weight))
    return result


def plot_comparison(runs_dir, out_dir, show=False):
    """生成对比图。"""
    os.makedirs(out_dir, exist_ok=True)

    # 加载所有数据
    data = {}
    for filename, (label, color) in RUNS.items():
        path = os.path.join(runs_dir, filename)
        if not os.path.exists(path):
            print(f"  跳过 {label}: 文件不存在")
            continue
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        data[label] = {"color": color, "tags": tags}
        for tag in tags:
            steps, values = load_scalar(path, tag)
            data[label][tag] = (steps, values)
        print(f"  ✓ {label}: {len(tags)} tags")

    # ── 图 1: 胜率对比 ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("fheroes2 Battle AI — 训练历史对比 (T4→T7)", fontsize=16, fontweight="bold")

    # 1a: 平均胜率
    ax = axes[0, 0]
    for label, d in data.items():
        color = d["color"]
        for tag in ["eval/win_rate", "eval/best_win_rate"]:
            if tag in d["tags"]:
                steps, values = d[tag]
                style = "-" if tag == "eval/win_rate" else "--"
                ax.plot([s / 1000 for s in steps], values, style,
                        color=color, label=label if tag == "eval/win_rate" else None,
                        alpha=0.8, linewidth=1.5)
    ax.set_xlabel("训练步数 (K)")
    ax.set_ylabel("胜率")
    ax.set_title("平均胜率 (实线=当前, 虚线=最佳)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    # 1b: 每步平均奖励
    ax = axes[0, 1]
    for label, d in data.items():
        if "train/mean_reward" in d:
            steps, values = d["train/mean_reward"]
            ax.plot([s / 1000 for s in steps], smooth(values, 0.7),
                    color=d["color"], label=label, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("训练步数 (K)")
    ax.set_ylabel("平均奖励 (EMA)")
    ax.set_title("训练奖励趋势")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1c: 策略损失
    ax = axes[1, 0]
    for label, d in data.items():
        if "train/policy_loss" in d:
            steps, values = d["train/policy_loss"]
            ax.plot([s / 1000 for s in steps], smooth(values, 0.7),
                    color=d["color"], label=label, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("训练步数 (K)")
    ax.set_ylabel("Policy Loss (EMA)")
    ax.set_title("策略损失")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 1d: 熵 (探索度)
    ax = axes[1, 1]
    for label, d in data.items():
        if "train/entropy" in d:
            steps, values = d["train/entropy"]
            ax.plot([s / 1000 for s in steps], smooth(values, 0.7),
                    color=d["color"], label=label, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("训练步数 (K)")
    ax.set_ylabel("Entropy (EMA)")
    ax.set_title("探索度 (越高 = 越多尝试)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\n✓ 保存: {path}")

    # ── 图 2: 各配置胜率 (T6 & T7) ────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("各配置单独胜率对比", fontsize=16, fontweight="bold")

    for ax, target_label in [(ax1, "T6 稳定性 500K"), (ax2, "T7 多样化 1M")]:
        d = data.get(target_label, {})
        config_tags = [t for t in d.get("tags", []) if t.startswith("eval/") and t.endswith("/win_rate")]
        if not config_tags:
            ax.text(0.5, 0.5, f"无数据: {target_label}", ha="center", va="center")
            continue

        for tag in sorted(config_tags):
            config_name = tag.replace("eval/", "").replace("/win_rate", "")
            steps, values = d[tag]
            ax.plot([s / 1000 for s in steps], values, label=config_name, alpha=0.8, linewidth=1.5)

        ax.set_xlabel("训练步数 (K)")
        ax.set_ylabel("胜率")
        ax.set_title(target_label)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path2 = os.path.join(out_dir, "per_config_winrate.png")
    fig2.savefig(path2, dpi=150)
    print(f"✓ 保存: {path2}")

    # ── 图 3: 最终胜率柱状图 ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    final_results = {}

    for label, d in data.items():
        for tag in ["eval/win_rate", "eval/best_win_rate"]:
            if tag in d and d[tag][0]:
                steps, values = d[tag]
                key = f"{label}\n({'当前' if 'best' not in tag else '最佳'})"
                final_results[key] = values[-1]

    if final_results:
        colors_map = {
            "T4": "#2196F3", "T5": "#4CAF50", "T6": "#FF9800", "T7": "#F44336"
        }
        bar_colors = []
        for k in final_results:
            for prefix, c in colors_map.items():
                if prefix in k:
                    bar_colors.append(c)
                    break
            else:
                bar_colors.append("#999")

        bars = ax3.bar(final_results.keys(), final_results.values(), color=bar_colors, alpha=0.85)
        for bar, val in zip(bars, final_results.values()):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.1%}", ha="center", va="bottom", fontweight="bold")

        ax3.set_ylabel("胜率")
        ax3.set_title("各训练阶段最终胜率对比", fontsize=14, fontweight="bold")
        ax3.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax3.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path3 = os.path.join(out_dir, "final_winrate_bar.png")
        fig3.savefig(path3, dpi=150)
        print(f"✓ 保存: {path3}")

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"\n共生成 3 张图片在 {out_dir}/ 目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练历史")
    parser.add_argument("--show", action="store_true", help="弹出窗口显示")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    plot_comparison(args.runs_dir, args.out_dir, args.show)
