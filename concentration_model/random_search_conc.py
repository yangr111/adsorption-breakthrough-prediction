"""
random_search_conc.py — 浓度曲线模型随机搜索（新版，保留每次 trial 的模型文件）

功能：
1. 对多种深度学习模型和特征集进行随机搜索。
2. 每次 trial 完成后，立刻把 checkpoint、scaler、history、metrics 复制到独立 artifact 目录。
3. 输出 all_trials.csv、best_configs.csv 和 pivot 表。
4. 指标默认按 R2 最大筛选。

推荐用法：
    python random_search_conc.py --trials 8 --epochs 500
"""

import os
import json
import time
import glob
import shutil
import random
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common_names import display_name, CONC_DL_MODELS

FEATURES_LIST = [24, 38, 50]

SEARCH_SPACE = {
    "resnet": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "num_blocks": [3, 5, 7],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
        "smooth_weight": [0.0, 0.005, 0.01],
    },
    "mlp": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "num_layers": [4, 6, 8],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
        "smooth_weight": [0.0, 0.005, 0.01],
    },
    "lstm": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "lstm_layers": [1, 2, 3],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
        "smooth_weight": [0.0, 0.005, 0.01],
    },
    "transformer": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1],
        "num_layers": [2, 4, 6],
        "nhead": [4, 8],
        "dim_feedforward": [256, 512, 1024],
        "lr": [5e-4, 1e-4],
        "batch_size": [256, 512],
        "smooth_weight": [0.0, 0.005, 0.01],
    },
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def post_process(model_name, cfg):
    cfg = dict(cfg)
    if model_name == "transformer":
        valid_heads = [h for h in [4, 8] if cfg["hidden"] % h == 0]
        if len(valid_heads) == 0:
            return None
        if cfg["nhead"] not in valid_heads:
            cfg["nhead"] = random.choice(valid_heads)
        if cfg["dim_feedforward"] < cfg["hidden"]:
            cfg["dim_feedforward"] = cfg["hidden"] * 2
    return cfg


def sample_config(model_name):
    space = SEARCH_SPACE[model_name]
    cfg = {k: random.choice(v) for k, v in space.items()}
    return post_process(model_name, cfg)


def build_cmd(model, features, cfg, seed, args):
    cmd = [
        "python", args.train_script,
        "--model", model,
        "--features", str(features),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
    ]
    for k, v in cfg.items():
        if v is not None:
            cmd.extend([f"--{k}", str(v)])
    return cmd


def expected_tag(model, features):
    return f"{model}_feat{features}"


def load_metrics(model, features):
    tag = expected_tag(model, features)
    path = f"results/{tag}_metrics.json"
    if not os.path.exists(path):
        print(f"Metrics not found: {path}")
        return None, None
    return load_json(path), path


def backup_artifacts(model, features, trial, cfg, metrics, metrics_path, cmd, elapsed, args):
    tag = expected_tag(model, features)
    trial_dir = os.path.join(args.search_dir, "artifacts", f"{model}_feat{features}_trial{trial + 1:03d}")
    ensure_dir(trial_dir)

    save_json(cfg, os.path.join(trial_dir, "config.json"))
    save_json(metrics, os.path.join(trial_dir, "metrics.json"))
    save_json({
        "task": "concentration",
        "model": model,
        "model_display": display_name(model),
        "features": features,
        "trial": trial + 1,
        "elapsed_sec": round(elapsed, 2),
        "command": cmd,
        "metrics_source": metrics_path,
    }, os.path.join(trial_dir, "run_info.json"))

    patterns = [
        f"results/{tag}_*.json",
        f"checkpoints/conc_{tag}*.pt",
        f"scalers/{tag}_*.pkl",
    ]
    copied = []
    for pat in patterns:
        for src in sorted(glob.glob(pat)):
            if os.path.isfile(src):
                dst = os.path.join(trial_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                copied.append(dst)
    print(f"[Backup] {len(copied)} files -> {trial_dir}")
    return trial_dir, copied


def run_trial(model, features, cfg, trial, args):
    seed = args.seed + trial
    cmd = build_cmd(model, features, cfg, seed, args)

    print("\n" + "=" * 90)
    print(f"[Trial] {display_name(model)} | feat={features} | trial={trial+1}/{args.trials}")
    print("CMD:", " ".join(cmd))
    print("=" * 90)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"Trial failed after {elapsed:.1f}s: {e}")
        return {
            "task": "concentration",
            "model": model,
            "model_display": display_name(model),
            "features": features,
            "trial": trial + 1,
            "status": "failed",
            "elapsed_sec": round(elapsed, 2),
            **cfg,
        }

    elapsed = time.time() - t0
    metrics, metrics_path = load_metrics(model, features)
    if metrics is None:
        return {
            "task": "concentration",
            "model": model,
            "model_display": display_name(model),
            "features": features,
            "trial": trial + 1,
            "status": "no_metrics",
            "elapsed_sec": round(elapsed, 2),
            **cfg,
        }

    artifact_dir, copied = backup_artifacts(model, features, trial, cfg, metrics, metrics_path, cmd, elapsed, args)
    rec = {
        "task": "concentration",
        "model": model,
        "model_display": display_name(model),
        "features": features,
        "trial": trial + 1,
        "seed": seed,
        "status": "success",
        "elapsed_sec": round(elapsed, 2),
        "artifact_dir": artifact_dir,
        "n_copied_files": len(copied),
        **cfg,
        **metrics,
    }
    return rec


def save_records(records, args):
    ensure_dir(args.search_dir)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.search_dir, "all_trials.csv"), index=False, encoding="utf-8-sig")


def summarize(records, args):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df[df.get("status") == "success"].copy()
    if df.empty:
        return pd.DataFrame()

    metric = args.metric
    direction = args.direction
    if metric not in df.columns:
        if "R2" in df.columns:
            metric = "R2"
            direction = "max"
        elif "MAE" in df.columns:
            metric = "MAE"
            direction = "min"
        else:
            raise ValueError("No usable metric column found.")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    ascending = direction == "min"

    best_rows = []
    for (m, f), sub in df.groupby(["model", "features"]):
        sort_cols = [metric]
        ascending_list = [ascending]
        if metric != "MAE" and "MAE" in sub.columns:
            sort_cols.append("MAE")
            ascending_list.append(True)
        sub = sub.sort_values(sort_cols, ascending=ascending_list)
        best_rows.append(sub.iloc[0])

    best_df = pd.DataFrame(best_rows).sort_values(["model", "features"])
    best_df.to_csv(os.path.join(args.search_dir, "best_configs.csv"), index=False, encoding="utf-8-sig")

    try:
        pivot = best_df.pivot(index="model_display", columns="features", values=metric)
        pivot.to_csv(os.path.join(args.search_dir, f"best_pivot_{metric}.csv"), encoding="utf-8-sig")
        print("\n===== Best pivot =====")
        print(pivot)
    except Exception:
        pass

    print("\n===== Best configs =====")
    show = ["model_display", "features", "trial", metric, "R2", "R2_C1", "R2_C2", "MAE", "RMSE", "artifact_dir",
            "hidden", "dropout", "lr", "batch_size", "smooth_weight",
            "num_blocks", "num_layers", "lstm_layers", "channels", "kernel_size", "num_levels", "nhead", "dim_feedforward"]
    show = [c for c in show if c in best_df.columns]
    print(best_df[show].to_string(index=False))
    return best_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", type=str, default="train_conc.py")
    parser.add_argument("--search_dir", type=str, default="search_conc_results")
    parser.add_argument("--models", nargs="+", default=CONC_DL_MODELS)
    parser.add_argument("--features", nargs="+", type=int, default=FEATURES_LIST)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="R2")
    parser.add_argument("--direction", type=str, default="max", choices=["max", "min"])
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_dir(args.search_dir)
    ensure_dir(os.path.join(args.search_dir, "artifacts"))
    save_json(vars(args), os.path.join(args.search_dir, "search_args.json"))

    records = []
    for model in args.models:
        if model not in SEARCH_SPACE:
            print(f"Skip unsupported model: {model}")
            continue
        for features in args.features:
            print("\n" + "#" * 100)
            print(f"Search Start | model={display_name(model)} | features={features}")
            print("#" * 100)
            for trial in range(args.trials):
                cfg = sample_config(model)
                if cfg is None:
                    continue
                rec = run_trial(model, features, cfg, trial, args)
                records.append(rec)
                save_records(records, args)

    summarize(records, args)
    print(f"\nAll concentration-search results saved in: {args.search_dir}")


if __name__ == "__main__":
    main()
