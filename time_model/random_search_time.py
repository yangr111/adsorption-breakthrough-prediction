"""
random_search_time.py — 时间模型随机搜索（新版，保留每次 trial 的模型文件）

功能：
1. 对 total/c1/c2 三个时间目标进行随机搜索。
2. 每次 trial 完成后，立刻把 checkpoint、scaler、history、metrics 复制到独立 artifact 目录。
3. 输出 all_trials_*.csv、best_configs_*.csv、summary_all_targets.csv。
4. 指标默认按 R2 最大筛选；如果想按 MAPE 最小，可使用 --metric 'MAPE(%)' --direction min。

推荐用法：
    python random_search_time.py --targets total --trials 8 --epochs 500
    python random_search_time.py --targets total c1 c2 --trials 8
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

from common_names import display_name, DL_MODELS, TIME_ML_MODELS

ALL_TARGETS = ["total", "c1", "c2"]
FEATURES_LIST = [24, 38, 50]

DL_SEARCH_SPACE = {
    "resnet": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "num_blocks": [3, 5, 7],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
    },
    "mlp": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "num_layers": [4, 6, 8],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
    },
    "lstm": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "lstm_layers": [1, 2, 3],
        "lr": [1e-3, 5e-4, 4e-4, 1e-4],
        "batch_size": [256, 512],
    },
    "transformer": {
        "hidden": [128, 256, 512],
        "dropout": [0.0, 0.05, 0.1],
        "num_layers": [2, 4, 6],
        "nhead": [4, 8],
        "dim_feedforward": [256, 512, 1024],
        "lr": [5e-4, 1e-4],
        "batch_size": [256, 512],
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


def sample_dl_config(model_name):
    space = DL_SEARCH_SPACE[model_name]
    cfg = {k: random.choice(v) for k, v in space.items()}
    return post_process(model_name, cfg)


def build_cmd(model, features, target, cfg, seed, args):
    cmd = [
        "python", args.train_script,
        "--model", model,
        "--features", str(features),
        "--target", target,
        "--threshold", str(args.threshold),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
    ]
    for k, v in cfg.items():
        if v is not None:
            cmd.extend([f"--{k}", str(v)])
    return cmd


def expected_tag(model, features, target):
    return f"{model}_feat{features}_{target}"


def load_metrics(model, features, target):
    tag = expected_tag(model, features, target)
    path = f"results/time_{tag}_metrics.json"
    if not os.path.exists(path):
        print(f"Metrics not found: {path}")
        return None, None
    return load_json(path), path


def backup_artifacts(model, features, target, trial, cfg, metrics, metrics_path, cmd, elapsed, args):
    tag = expected_tag(model, features, target)
    trial_dir = os.path.join(
        args.search_dir,
        "artifacts",
        target,
        f"{model}_feat{features}_trial{trial + 1:03d}"
    )
    ensure_dir(trial_dir)

    save_json(cfg, os.path.join(trial_dir, "config.json"))
    save_json(metrics, os.path.join(trial_dir, "metrics.json"))
    save_json({
        "task": "time",
        "model": model,
        "model_display": display_name(model),
        "features": features,
        "target": target,
        "trial": trial + 1,
        "elapsed_sec": round(elapsed, 2),
        "command": cmd,
        "metrics_source": metrics_path,
    }, os.path.join(trial_dir, "run_info.json"))

    patterns = [
        f"results/time_{tag}_*.json",
        f"checkpoints/time_{tag}*.pt",
        f"checkpoints/time_{tag}*.pkl",
        f"scalers/time_{tag}_*.pkl",
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


def run_trial(model, features, target, cfg, trial, args):
    seed = args.seed + trial
    cmd = build_cmd(model, features, target, cfg, seed, args)

    print("\n" + "=" * 90)
    print(f"[Trial] {display_name(model)} | feat={features} | target={target} | trial={trial+1}/{args.trials}")
    print("CMD:", " ".join(cmd))
    print("=" * 90)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"Trial failed after {elapsed:.1f}s: {e}")
        return {
            "task": "time",
            "model": model,
            "model_display": display_name(model),
            "features": features,
            "target": target,
            "trial": trial + 1,
            "status": "failed",
            "elapsed_sec": round(elapsed, 2),
            **cfg,
        }

    elapsed = time.time() - t0
    metrics, metrics_path = load_metrics(model, features, target)
    if metrics is None:
        return {
            "task": "time",
            "model": model,
            "model_display": display_name(model),
            "features": features,
            "target": target,
            "trial": trial + 1,
            "status": "no_metrics",
            "elapsed_sec": round(elapsed, 2),
            **cfg,
        }

    artifact_dir, copied = backup_artifacts(model, features, target, trial, cfg, metrics, metrics_path, cmd, elapsed, args)

    rec = {
        "task": "time",
        "model": model,
        "model_display": display_name(model),
        "features": features,
        "target": target,
        "trial": trial + 1,
        "seed": args.seed + trial,
        "status": "success",
        "elapsed_sec": round(elapsed, 2),
        "artifact_dir": artifact_dir,
        "n_copied_files": len(copied),
        **cfg,
        **metrics,
    }
    return rec


def save_records(records, target, args):
    ensure_dir(args.search_dir)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.search_dir, f"all_trials_{target}.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(args.search_dir, "all_trials.csv"), index=False, encoding="utf-8-sig")


def summarize(records, target, args):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df[(df.get("status") == "success") & (df.get("target") == target)].copy()
    if df.empty:
        return pd.DataFrame()

    metric = args.metric
    direction = args.direction
    if metric not in df.columns:
        if "R2" in df.columns:
            metric = "R2"
            direction = "max"
        elif "MAPE(%)" in df.columns:
            metric = "MAPE(%)"
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
    best_df.to_csv(os.path.join(args.search_dir, f"best_configs_{target}.csv"), index=False, encoding="utf-8-sig")

    try:
        pivot = best_df.pivot(index="model_display", columns="features", values=metric)
        pivot.to_csv(os.path.join(args.search_dir, f"best_pivot_{target}_{metric.replace('%','pct').replace('(','').replace(')','')}.csv"), encoding="utf-8-sig")
        print("\n===== Best pivot =====")
        print(pivot)
    except Exception:
        pass

    print(f"\n===== Best configs for target={target} =====")
    show = ["model_display", "features", "trial", metric, "R2", "MAE", "MAPE(%)", "RMSE", "artifact_dir",
            "hidden", "dropout", "lr", "batch_size", "num_blocks", "num_layers", "lstm_layers", "nhead", "dim_feedforward"]
    show = [c for c in show if c in best_df.columns]
    print(best_df[show].to_string(index=False))
    return best_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_script", type=str, default="train_time.py")
    parser.add_argument("--search_dir", type=str, default="search_time_results")
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    parser.add_argument("--models", nargs="+", default=DL_MODELS)
    parser.add_argument("--include_ml", action="store_true", help="额外运行 RF/XGBoost/LightGBM 时间基线。")
    parser.add_argument("--features", nargs="+", type=int, default=FEATURES_LIST)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.01)
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

    all_records = []
    all_best = []

    run_models = list(args.models)
    if args.include_ml:
        run_models += TIME_ML_MODELS

    for target in args.targets:
        target_records = []
        print("\n" + "#" * 100)
        print(f"TARGET = {target}")
        print("#" * 100)

        for model in run_models:
            is_ml = model in TIME_ML_MODELS
            for features in args.features:
                n_trials = 1 if is_ml else args.trials
                for trial in range(n_trials):
                    cfg = {} if is_ml else sample_dl_config(model)
                    if cfg is None:
                        continue
                    rec = run_trial(model, features, target, cfg, trial, args)
                    target_records.append(rec)
                    all_records.append(rec)
                    save_records(all_records, target, args)

        best_df = summarize(target_records, target, args)
        if not best_df.empty:
            all_best.append(best_df)

    if all_best:
        summary = pd.concat(all_best, ignore_index=True)
        summary.to_csv(os.path.join(args.search_dir, "summary_all_targets.csv"), index=False, encoding="utf-8-sig")

    print(f"\nAll time-search results saved in: {args.search_dir}")


if __name__ == "__main__":
    main()
