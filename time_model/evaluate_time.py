"""
evaluate_time.py — 汇总穿透时间预测实验结果

支持三个目标（total / c1 / c2），按 target 分组展示和出图。

用法：
    python evaluate_time.py                        # 汇总所有目标，不出图
    python evaluate_time.py --plot                 # 额外出图
    python evaluate_time.py --target c1 --plot     # 只看 c1 目标
"""

import json
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ML 模型集合（用于配色区分）
ML_MODELS = {"svr", "rf", "xgb", "lgb"}
ALL_TARGETS = ["total", "c1", "c2"]
TARGET_LABELS = {
    "total": "穿透总时间 (total)",
    "c1":    "C1 穿出时间 (c1)",
    "c2":    "C2 穿出时间 (c2)",
}


# ============================================================
#  读取结果
# ============================================================
def load_all_metrics(results_dir: str = "results") -> pd.DataFrame:
    """
    读取所有 time_*_metrics.json，自动识别 target 字段。
    若 JSON 里没有 target 字段（旧版结果），从文件名中解析。
    """
    pattern = os.path.join(results_dir, "time_*_metrics.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No time metrics found in '{results_dir}/'. Run train_time.py first.")
        return pd.DataFrame()

    rows = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        # 兼容旧版（没有 target 字段）：从文件名末尾解析
        if "target" not in d:
            basename = os.path.basename(f)   # time_resnet_feat24_c1_metrics.json
            for t in ALL_TARGETS:
                if f"_{t}_metrics" in basename:
                    d["target"] = t
                    break
            else:
                d["target"] = "total"
        rows.append(d)

    df = pd.DataFrame(rows)
    df["_is_ml"] = df["model"].isin(ML_MODELS).astype(int)
    df = (df.sort_values(["target", "_is_ml", "model", "features"])
            .drop(columns="_is_ml")
            .reset_index(drop=True))
    return df


def filter_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[df["target"] == target].reset_index(drop=True)


# ============================================================
#  打印表格
# ============================================================
def print_table(df: pd.DataFrame, target: str):
    sub = filter_target(df, target)
    if sub.empty:
        print(f"  No results for target={target}")
        return

    cols = ["experiment", "model", "features", "params", "R2", "MAE", "MAPE(%)", "RMSE"]
    cols = [c for c in cols if c in sub.columns]
    disp = sub[cols].copy()
    for c in ["R2"]:
        if c in disp: disp[c] = disp[c].apply(lambda x: f"{float(x):.4f}")
    for c in ["MAE", "RMSE"]:
        if c in disp: disp[c] = disp[c].apply(lambda x: f"{float(x):.4f}")
    for c in ["MAPE(%)"]:
        if c in disp: disp[c] = disp[c].apply(lambda x: f"{float(x):.2f}")

    label = TARGET_LABELS.get(target, target)
    print(f"\n{'='*80}")
    print(f"  {label}")
    print("="*80)
    print(disp.to_string(index=False))
    print("="*80)

    for metric, asc, fmt in [("R2", False, ".4f"), ("MAPE(%)", True, ".2f"), ("RMSE", True, ".4f")]:
        if metric in sub.columns:
            idx  = sub[metric].astype(float).idxmin() if asc else sub[metric].astype(float).idxmax()
            best = sub.loc[idx]
            verb = "min" if asc else "max"
            print(f"  Best {metric:8s} ({verb}): {float(best[metric]):{fmt}}  → {best['experiment']}")
    print()


# ============================================================
#  出图
# ============================================================
def plot_results(df: pd.DataFrame, results_dir: str = "results",
                 targets: list = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    targets = targets or [t for t in ALL_TARGETS if t in df["target"].unique()]
    os.makedirs(results_dir, exist_ok=True)

    DL_COLOR = "#4C72B0"
    ML_COLOR = "#DD8452"

    # ── 图1：每个 target 各一张 R² + MAPE 双列图 ──────────────
    for target in targets:
        sub = filter_target(df, target)
        if sub.empty:
            continue

        label   = TARGET_LABELS.get(target, target)
        exps    = sub["experiment"].tolist()
        r2_vals = sub["R2"].astype(float).tolist()
        mp_vals = sub["MAPE(%)"].astype(float).tolist()
        colors  = [ML_COLOR if m in ML_MODELS else DL_COLOR
                   for m in sub["model"].tolist()]
        x = np.arange(len(exps))

        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(exps) * 0.9), 4.5))

        axes[0].bar(x, r2_vals, color=colors, zorder=3, edgecolor="white", linewidth=0.4)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(exps, rotation=35, ha="right", fontsize=7)
        axes[0].set_ylabel("R²")
        axes[0].set_title(f"R²  —  {label}")
        axes[0].set_ylim(0, min(1.05, max(r2_vals) + 0.08))
        axes[0].axhline(0.95, color="red", ls=":", lw=0.8, alpha=0.6)
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(x, mp_vals, color=colors, zorder=3, edgecolor="white", linewidth=0.4)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(exps, rotation=35, ha="right", fontsize=7)
        axes[1].set_ylabel("MAPE (%)")
        axes[1].set_title(f"MAPE (%)  —  {label}")
        axes[1].grid(True, axis="y", alpha=0.3)

        handles = [mpatches.Patch(color=DL_COLOR, label="Deep Learning"),
                   mpatches.Patch(color=ML_COLOR, label="Machine Learning")]
        fig.legend(handles=handles, loc="upper right", fontsize=9)
        plt.tight_layout()

        out = os.path.join(results_dir, f"time_compare_{target}.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close()

    # ── 图2：三个目标的 R² 热力图，行=模型，列=目标×特征 ───────
    if len(targets) > 1:
        models   = sorted(df["model"].unique())
        features = sorted(df["features"].unique())

        # 列：(target, feat) 的组合
        col_keys  = [(t, f) for t in targets for f in features if t in df["target"].unique()]
        col_labels = [f"{t}\n{f}feat" for t, f in col_keys]

        mat = np.full((len(models), len(col_keys)), np.nan)
        for i, m in enumerate(models):
            for j, (t, f) in enumerate(col_keys):
                sub = df[(df["model"] == m) & (df["target"] == t) & (df["features"] == f)]
                if not sub.empty:
                    mat[i, j] = float(sub["R2"].values[0])

        fig, ax = plt.subplots(figsize=(len(col_keys) * 1.2 + 1.5,
                                         len(models) * 0.6 + 1.5))
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
        im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(col_keys)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.upper() for m in models])

        for i in range(len(models)):
            for j in range(len(col_keys)):
                v = mat[i, j]
                if not np.isnan(v):
                    t_norm = (v - vmin) / (vmax - vmin + 1e-8)
                    tc = "white" if t_norm > 0.6 else "#1a3a5c"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color=tc, fontweight="500")

        # 目标分隔线
        for k in range(1, len(targets)):
            ax.axvline(k * len(features) - 0.5, color="white", lw=2)

        plt.colorbar(im, ax=ax, shrink=0.7, label="R²")
        ax.set_title("R² heatmap — all targets × models × features", fontsize=11, pad=10)
        plt.tight_layout()
        out = os.path.join(results_dir, "time_heatmap_all_targets.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close()

    # ── 图3：DL 时间模型训练曲线（只画 total）────────────────────
    # 只画 total
    target = "total"

    history_files = sorted(glob.glob(os.path.join(results_dir, "time_*_history.json")))
    target_files = [f for f in history_files if f"_{target}_history" in os.path.basename(f)]

    if target_files:
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        # 模型名称美化
        name_map = {
            "lstm_feat24": "LSTM-24",
            "lstm_feat38": "LSTM-38",
            "lstm_feat50": "LSTM-50",
            "mlp_feat24": "MLP-24",
            "mlp_feat38": "MLP-38",
            "mlp_feat50": "MLP-50",
            "resnet_feat24": "ResNet-24",
            "resnet_feat38": "ResNet-38",
            "resnet_feat50": "ResNet-50",
            "transformer_feat24": "Transformer-24",
            "transformer_feat38": "Transformer-38",
            "transformer_feat50": "Transformer-50",
        }

        # 相对柔和的颜色
        colors = [
            "#4C78A8", "#72B7B2", "#54A24B",
            "#F58518", "#E45756", "#B279A2",
            "#9D755D", "#BAB0AC", "#8CD17D",
            "#499894", "#D37295", "#79706E",
        ]

        for i, f in enumerate(target_files):
            tag = (
                os.path.basename(f)
                .replace("time_", "")
                .replace(f"_{target}_history.json", "")
            )

            with open(f, "r", encoding="utf-8") as fp:
                h = json.load(fp)

            val_r2 = h.get("val_r2", [])

            if len(val_r2) == 0:
                continue

            label = name_map.get(tag, tag)

            ax.plot(
                val_r2,
                label=label,
                lw=1.6,
                color=colors[i % len(colors)],
                alpha=0.95
            )

        ax.set_title("Validation R² curves of time models", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation R²", fontsize=11, fontweight="bold")

        ax.tick_params(axis="both", labelsize=10, width=1.0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        ax.legend(
            frameon=True,
            loc="lower right",
            ncol=1,
            prop={"weight": "bold", "size": 8}
        )

        plt.tight_layout()

        out_png = os.path.join(results_dir, "time_total_val_r2_curve.png")
        out_pdf = os.path.join(results_dir, "time_total_val_r2_curve.pdf")

        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")

        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")

        plt.close()
    else:
        print("No total history files found.")


# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,   default="results")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--target",      type=str,   default=None,
                        choices=ALL_TARGETS,
                        help="只看某个目标；不指定则显示全部")
    args = parser.parse_args()

    df = load_all_metrics(args.results_dir)
    if df.empty:
        exit(0)

    targets_to_show = [args.target] if args.target else ALL_TARGETS

    # 按 target 分组打印表格
    for t in targets_to_show:
        if t in df["target"].unique():
            print_table(df, t)

    # 保存完整 CSV（所有目标合并）
    csv_path = os.path.join(args.results_dir, "time_summary_all.csv")
    df.to_csv(csv_path, index=False)
    print(f"Full summary CSV: {csv_path}")

    # 按 target 各保存一份 CSV
    for t in df["target"].unique():
        sub = filter_target(df, t)
        sub.to_csv(os.path.join(args.results_dir, f"time_summary_{t}.csv"), index=False)
        print(f"Target CSV: {args.results_dir}/time_summary_{t}.csv")


    plot_results(df, args.results_dir, targets=targets_to_show)
