"""
evaluate.py — 汇总所有实验结果，打印对比表，并绘制训练曲线

用法：
    python evaluate.py                    # 汇总 results/ 下所有结果
    python evaluate.py --plot             # 额外绘制训练 loss 曲线图
"""

import json
import os
import argparse
import glob
import pandas as pd


# ============================================================
#  读取所有 metrics JSON
# ============================================================
def load_all_metrics(results_dir="results") -> pd.DataFrame:
    pattern = os.path.join(results_dir, "*_metrics.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No metrics files found in '{results_dir}/'. Run train.py first.")
        return pd.DataFrame()

    rows = []
    for f in files:
        with open(f) as fp:
            rows.append(json.load(fp))

    df = pd.DataFrame(rows)
    # 排序：先按 model，再按 features
    df = df.sort_values(["model", "features"]).reset_index(drop=True)
    return df


# ============================================================
#  美化打印
# ============================================================
def print_table(df: pd.DataFrame):
    cols = ["experiment", "model", "features", "params",
            "R2_all", "R2_C1", "R2_C2", "MAE", "RMSE"]
    # 只保留存在的列
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()

    # 格式化数值
    for c in ["R2_all", "R2_C1", "R2_C2"]:
        if c in sub:
            sub[c] = sub[c].apply(lambda x: f"{x:.4f}")
    for c in ["MAE", "RMSE"]:
        if c in sub:
            sub[c] = sub[c].apply(lambda x: f"{x:.6f}")

    print("\n" + "="*80)
    print("  EXPERIMENT COMPARISON TABLE")
    print("="*80)
    print(sub.to_string(index=False))
    print("="*80 + "\n")

    # 找最优实验
    if "R2_all" in df.columns:
        best_idx = df["R2_all"].astype(float).idxmax()
        best = df.loc[best_idx]
        print(f"  Best R²(all): {float(best['R2_all']):.4f}  →  {best['experiment']}")
    if "RMSE" in df.columns:
        best_idx = df["RMSE"].astype(float).idxmin()
        best = df.loc[best_idx]
        print(f"  Best RMSE   : {float(best['RMSE']):.6f}  →  {best['experiment']}")
    print()


# ============================================================
#  绘制训练 loss 曲线（可选）
# ============================================================
def plot_loss_curves(results_dir="results"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    pattern = os.path.join(results_dir, "*_history.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print("No history files found.")
        return

    # 字体设置：Linux 下不用强制 Arial，避免 findfont 报错
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_train, ax_val = axes

    colors = plt.cm.tab10.colors

    for i, f in enumerate(files):
        tag = os.path.basename(f).replace("_history.json", "")

        with open(f, "r", encoding="utf-8") as fp:
            h = json.load(fp)

        c = colors[i % len(colors)]

        if "train_loss" in h:
            ax_train.plot(
                h["train_loss"],
                label=tag,
                color=c,
                linewidth=1.8
            )

        if "val_loss" in h:
            ax_val.plot(
                h["val_loss"],
                label=tag,
                color=c,
                linestyle="--",
                linewidth=1.8
            )

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")

        ax.tick_params(axis="both", labelsize=10, width=1.2)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.legend(
            frameon=True,
            prop={"weight": "bold", "size": 8}
        )

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    plt.tight_layout()

    out_png = os.path.join(results_dir, "conc_loss_curves.png")
    out_pdf = os.path.join(results_dir, "conc_loss_curves.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")

    print(f"Loss curves saved to: {out_png}")
    print(f"Loss curves saved to: {out_pdf}")

    plt.show()


# ============================================================
#  绘制 R² 对比柱状图（可选）
# ============================================================
def plot_r2_bar(df: pd.DataFrame, results_dir="results"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if df.empty or "R2_all" not in df.columns:
        return

    experiments = df["experiment"].tolist()
    r2_all = df["R2_all"].astype(float).tolist()
    r2_c1  = df["R2_C1"].astype(float).tolist()
    r2_c2  = df["R2_C2"].astype(float).tolist()

    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, r2_all, width, label="R²(all)")
    ax.bar(x,         r2_c1,  width, label="R²(C1)")
    ax.bar(x + width, r2_c2,  width, label="R²(C2)")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("R² Comparison Across Experiments")
    ax.legend(prop={"weight": "bold"})
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="red", linestyle=":", alpha=0.5, label="R²=0.9")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "r2_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"R² comparison chart saved to: {out}")
    plt.show()

# ============================================================
#  绘制训练 R² 曲线（可选）
# ============================================================
def plot_training_curves(results_dir="results"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    pattern = os.path.join(results_dir, "*_history.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print("No history files found.")
        return

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10.colors
    has_curve = False

    for i, f in enumerate(files):
        tag = os.path.basename(f).replace("_history.json", "")

        with open(f, "r", encoding="utf-8") as fp:
            h = json.load(fp)

        c = colors[i % len(colors)]

        # 如果 history 里有 train_r2，就画训练 R²
        if "train_r2" in h and len(h["train_r2"]) > 0:
            ax.plot(
                h["train_r2"],
                label=f"{tag} Train R²",
                color=c,
                linewidth=1.8,
                linestyle="-"
            )
            has_curve = True

        # 如果 history 里有 val_R2，就画验证 R²
        if "val_R2" in h and len(h["val_R2"]) > 0:
            ax.plot(
                h["val_R2"],
                label=f"{tag} Val R²",
                color=c,
                linewidth=1.8,
                linestyle="--"
            )
            has_curve = True

    if not has_curve:
        print("No train_r2 or val_R2 found in history files.")
        plt.close()
        return

    ax.set_title("Training Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("R²", fontsize=12, fontweight="bold")

    ax.tick_params(axis="both", labelsize=10, width=1.2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(
        frameon=True,
        prop={"weight": "bold", "size": 8}
    )

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    out_png = os.path.join(results_dir, "training_r2_curves.png")
    out_pdf = os.path.join(results_dir, "training_r2_curves.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")

    print(f"Training curves saved to: {out_png}")
    print(f"Training curves saved to: {out_pdf}")

    plt.show()

# ============================================================
#  CSV 导出
# ============================================================
def save_csv(df: pd.DataFrame, results_dir="results"):
    out = os.path.join(results_dir, "summary.csv")
    df.to_csv(out, index=False)
    print(f"Summary CSV saved to: {out}")


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plot", action="store_true",
                        help="Plot loss curves and R² bar chart")
    args = parser.parse_args()

    df = load_all_metrics(args.results_dir)

    if df.empty:
        exit(0)

    print_table(df)
    save_csv(df, args.results_dir)

    if args.plot:
        plot_loss_curves(args.results_dir)
        plot_training_curves(args.results_dir)
        plot_r2_bar(df, args.results_dir)
