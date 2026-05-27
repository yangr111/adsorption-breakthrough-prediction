"""
plot_training_metrics.py — 将训练过程中的四个指标分别画成四张图

默认读取当前目录下 results/*_history.json。
每个指标单独出图：
    train_loss.png/pdf
    val_loss.png/pdf
    train_R2.png/pdf
    val_R2.png/pdf

用法：
    python plot_training_metrics.py --results_dir results
    python plot_training_metrics.py --pattern 'results/time_*_history.json'
    python plot_training_metrics.py --pattern 'results/*_history.json' --out_dir training_figures
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    from common_names import display_name
except Exception:
    def display_name(x):
        return str(x)


METRIC_KEYS = {
    "train_loss": ["train_loss", "Train_loss", "train_Loss"],
    "val_loss":   ["val_loss", "Val_loss", "valid_loss", "validation_loss"],
    "train_R2":   ["train_R2", "train_r2", "Train_R2"],
    "val_R2":     ["val_R2", "val_r2", "Val_R2", "valid_R2", "validation_R2"],
}


def get_item(history, candidates):
    for k in candidates:
        if k in history and isinstance(history[k], list) and len(history[k]) > 0:
            return history[k]
    return None


def make_label(path):
    name = os.path.basename(path).replace("_history.json", "")
    name = name.replace("time_", "")
    parts = name.split("_")
    # 尽量把模型名改成统一显示格式
    if parts:
        parts[0] = display_name(parts[0])
    return "_".join(parts)


def style_axis(ax, ylabel):
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_metric(files, metric_name, out_dir):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.cm.tab10.colors
    has_curve = False

    for i, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fp:
            h = json.load(fp)
        y = get_item(h, METRIC_KEYS[metric_name])
        if y is None:
            continue
        ax.plot(y, lw=1.8, color=colors[i % len(colors)], label=make_label(f))
        has_curve = True

    if not has_curve:
        plt.close(fig)
        print(f"Skip {metric_name}: no data found.")
        return

    title_map = {
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
        "train_R2": "Training R²",
        "val_R2": "Validation R²",
    }
    ylabel_map = {
        "train_loss": "Loss",
        "val_loss": "Loss",
        "train_R2": "R²",
        "val_R2": "R²",
    }

    ax.set_title(title_map.get(metric_name, metric_name), fontsize=13, fontweight="bold")
    style_axis(ax, ylabel_map.get(metric_name, metric_name))
    ax.legend(frameon=True, prop={"weight": "bold", "size": 8})
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{metric_name}.png")
    out_pdf = os.path.join(out_dir, f"{metric_name}.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--pattern", type=str, default=None,
                        help="例如 results/time_*_history.json；如果不填则使用 results_dir/*_history.json")
    parser.add_argument("--out_dir", type=str, default="training_figures")
    args = parser.parse_args()

    pattern = args.pattern or os.path.join(args.results_dir, "*_history.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No history files found by pattern: {pattern}")
        return

    print(f"Found {len(files)} history files.")
    for metric in ["train_loss", "val_loss", "train_R2", "val_R2"]:
        plot_metric(files, metric, args.out_dir)


if __name__ == "__main__":
    main()
