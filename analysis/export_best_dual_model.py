"""
export_best_dual_model.py — 从随机搜索结果中导出最优双模型

作用：
    读取 time/search_time_results 和 conc/search_conc_results 中的结果；
    选出表现最好的时间模型和浓度模型；
    将对应 .pt/.pkl、scaler、metrics、config 以及模型结构文件复制到 final_dual_model/。

默认选择：
    时间模型：target=total，按 R2 最大，且只选择 PyTorch 深度学习模型（有 .pt 文件）
    浓度模型：按 R2 最大

用法：
    python export_best_dual_model.py
    python export_best_dual_model.py --time_target total --metric R2 --direction max
    python export_best_dual_model.py --allow_time_ml
"""

import os
import glob
import json
import shutil
import argparse
from pathlib import Path

import pandas as pd


ML_TIME_MODELS = {"rf", "xgb", "lgb", "svr"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_path(base_dir, maybe_relative):
    p = Path(str(maybe_relative))
    if p.is_absolute():
        return p
    return Path(base_dir) / p


def pick_best(csv_path, metric="R2", direction="max", target=None, dl_only=False):
    df = pd.read_csv(csv_path)
    df = df[df.get("status") == "success"].copy() if "status" in df.columns else df.copy()
    if target is not None and "target" in df.columns:
        df = df[df["target"] == target].copy()
    if dl_only and "model" in df.columns:
        df = df[~df["model"].isin(ML_TIME_MODELS)].copy()
    if df.empty:
        raise RuntimeError(f"No usable rows found in {csv_path}")

    if metric not in df.columns:
        fallback = "R2" if "R2" in df.columns else None
        if fallback is None:
            raise RuntimeError(f"Metric {metric} not found in {csv_path}")
        metric = fallback
        direction = "max"

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        raise RuntimeError(f"Metric {metric} has no numeric values in {csv_path}")

    ascending = direction == "min"
    sort_cols = [metric]
    asc = [ascending]
    if metric != "MAE" and "MAE" in df.columns:
        sort_cols.append("MAE")
        asc.append(True)

    return df.sort_values(sort_cols, ascending=asc).iloc[0].to_dict(), metric


def copy_first(patterns, dst_path, required=True):
    for pat in patterns:
        matches = sorted(glob.glob(str(pat)))
        for src in matches:
            if os.path.isfile(src):
                shutil.copy2(src, dst_path)
                return str(src)
    if required:
        raise FileNotFoundError("No file matched: " + " | ".join(map(str, patterns)))
    return None


def copy_artifact_set(row, base_dir, out_dir, prefix, allow_pkl_model=False):
    artifact_dir = resolve_path(base_dir, row["artifact_dir"])
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact_dir not found: {artifact_dir}")

    ensure_dir(out_dir)

    # metrics / config / run_info
    for name in ["metrics.json", "config.json", "run_info.json"]:
        src = artifact_dir / name
        if src.exists():
            shutil.copy2(src, Path(out_dir) / f"{prefix}_{name}")

    # 模型文件：优先 best_r2.pt；如果允许 ML，则可复制 pkl 模型
    copied_model = None
    try:
        copied_model = copy_first(
            [artifact_dir / "*best_r2.pt", artifact_dir / "*best_R2.pt", artifact_dir / "*.pt"],
            Path(out_dir) / f"{prefix}_model_best.pt",
            required=not allow_pkl_model,
        )
    except FileNotFoundError:
        if not allow_pkl_model:
            raise

    if copied_model is None and allow_pkl_model:
        copied_model = copy_first(
            [artifact_dir / "time_*.pkl", artifact_dir / "*.pkl"],
            Path(out_dir) / f"{prefix}_model_best.pkl",
            required=True,
        )

    scaler_x = copy_first([artifact_dir / "*scaler_X.pkl"], Path(out_dir) / f"{prefix}_scaler_X.pkl")
    scaler_y = copy_first([artifact_dir / "*scaler_Y.pkl", artifact_dir / "*scaler_y.pkl"], Path(out_dir) / f"{prefix}_scaler_Y.pkl")

    return {
        "artifact_dir": str(artifact_dir),
        "model_file_source": copied_model,
        "scaler_X_source": scaler_x,
        "scaler_Y_source": scaler_y,
    }


def write_models_wrapper(out_dir):
    content = r'''# -*- coding: utf-8 -*-
"""
models.py — 最终双模型绘图用模型入口

本文件用于 final_dual_model 文件夹：
    - 时间模型结构来自 models_time.py
    - 浓度模型结构来自 models_conc.py

绘图脚本可以这样导入：
    from models import build_time_model, build_conc_model
"""

from models_time import (
    build_time_model,
    default_groups as default_time_groups,
    ResNetTime,
    MLPTime,
    LSTMTime,
    TransformerTime,
)

from models_conc import (
    build_model as build_conc_model,
    default_groups as default_conc_groups,
    ResNetCurve,
    MLPCurve,
    LSTMCurve,
    TransformerCurve,
)

MODEL_DISPLAY_NAMES = {
    "mlp": "MLP",
    "resnet": "ResNet",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "rf": "RF",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_key).lower(), str(model_key))
'''
    with open(Path(out_dir) / "models.py", "w", encoding="utf-8") as f:
        f.write(content)


def copy_model_sources(time_dir, conc_dir, out_dir):
    for src in [
        Path(time_dir) / "models_time.py",
        Path(conc_dir) / "models_conc.py",
        Path(time_dir) / "feature_build.py",
        Path(time_dir) / "new_feature_build.py",
    ]:
        if src.exists():
            shutil.copy2(src, Path(out_dir) / src.name)
    write_models_wrapper(out_dir)


def write_summary_md(out_dir, time_row, conc_row, time_metric, conc_metric, copied):
    text = f"""# 最终双模型文件说明

该文件夹由 `export_best_dual_model.py` 自动生成，用于后续绘制双模型曲线对比图。

## 1. 时间模型

- 模型：{time_row.get('model_display', time_row.get('model'))}
- 特征数：{time_row.get('features')}
- 目标：{time_row.get('target')}
- 选择指标：{time_metric}
- 测试集 R²：{time_row.get('R2')}
- MAE：{time_row.get('MAE')}
- RMSE：{time_row.get('RMSE')}
- 原始 artifact：{copied['time']['artifact_dir']}

## 2. 浓度模型

- 模型：{conc_row.get('model_display', conc_row.get('model'))}
- 特征数：{conc_row.get('features')}
- 选择指标：{conc_metric}
- 测试集 R²：{conc_row.get('R2')}
- C1 R²：{conc_row.get('R2_C1')}
- C2 R²：{conc_row.get('R2_C2')}
- MAE：{conc_row.get('MAE')}
- RMSE：{conc_row.get('RMSE')}
- 原始 artifact：{copied['conc']['artifact_dir']}

## 3. 主要文件

- `time_model_best.pt`：时间模型权重，若选择了机器学习时间模型，则可能为 `time_model_best.pkl`
- `time_scaler_X.pkl`：时间模型输入标准化器
- `time_scaler_Y.pkl`：时间模型输出标准化器，注意时间模型是在 log1p 空间训练
- `conc_model_best.pt`：浓度模型权重
- `conc_scaler_X.pkl`：浓度模型输入标准化器
- `conc_scaler_Y.pkl`：浓度模型输出标准化器
- `models.py`：最终双模型统一入口
- `models_time.py` / `models_conc.py`：原始模型结构定义
- `time_metrics.json` / `conc_metrics.json`：最终模型对应指标

## 4. 后续画图建议

绘制曲线对比图时，建议使用：

1. 时间模型预测 `Tmax`；
2. 用 `np.linspace(0, Tmax_pred, 100)` 构造预测时间轴；
3. 浓度模型预测 200 维输出，前 100 维为 C1，后 100 维为 C2；
4. 真实曲线使用 `curve_clean.csv` 中每个样本自己的真实时间轴和真实浓度；
5. 预测曲线使用预测时间轴和预测浓度。
"""
    with open(Path(out_dir) / "README_final_dual_model.md", "w", encoding="utf-8") as f:
        f.write(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_dir", type=str, default="time_model")
    parser.add_argument("--conc_dir", type=str, default="concentration_model")
    parser.add_argument("--time_search_csv", type=str, default="search_time_results/all_trials.csv")
    parser.add_argument("--conc_search_csv", type=str, default="search_conc_results/all_trials.csv")
    parser.add_argument("--out_dir", type=str, default="final_dual_model")
    parser.add_argument("--time_target", type=str, default="total")
    parser.add_argument("--metric", type=str, default="R2")
    parser.add_argument("--direction", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--allow_time_ml", action="store_true", help="允许最终时间模型选择 RF/XGBoost/LightGBM 等 pkl 模型。默认只选 .pt 深度学习模型。")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    time_csv = Path(args.time_dir) / args.time_search_csv
    conc_csv = Path(args.conc_dir) / args.conc_search_csv

    time_row, time_metric = pick_best(
        time_csv,
        metric=args.metric,
        direction=args.direction,
        target=args.time_target,
        dl_only=not args.allow_time_ml,
    )
    conc_row, conc_metric = pick_best(
        conc_csv,
        metric=args.metric,
        direction=args.direction,
        target=None,
        dl_only=False,
    )

    copied = {
        "time": copy_artifact_set(time_row, args.time_dir, args.out_dir, "time", allow_pkl_model=args.allow_time_ml),
        "conc": copy_artifact_set(conc_row, args.conc_dir, args.out_dir, "conc", allow_pkl_model=False),
    }

    copy_model_sources(args.time_dir, args.conc_dir, args.out_dir)
    save_json({"time": time_row, "conc": conc_row, "copied": copied}, Path(args.out_dir) / "selected_models.json")
    write_summary_md(args.out_dir, time_row, conc_row, time_metric, conc_metric, copied)

    print("\nExport finished.")
    print(f"Final dual-model folder: {args.out_dir}")
    print("Selected time model:", time_row.get("model_display", time_row.get("model")), time_row.get("features"), time_row.get("target"), time_row.get("R2"))
    print("Selected conc model:", conc_row.get("model_display", conc_row.get("model")), conc_row.get("features"), conc_row.get("R2"))


if __name__ == "__main__":
    main()
