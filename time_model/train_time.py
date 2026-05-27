"""
train_time.py — 穿透时间预测训练脚本（新版）

主要改动：
1. 训练过程统一保存 train_loss、val_loss、train_R2、val_R2，便于后续每个指标单独作图。
2. PyTorch 时间模型同时保存 best_val_loss 和 best_val_R2 两个 checkpoint，最终测试默认使用 best_val_R2。
3. metrics.json 中保存模型名称、超参数、checkpoint/scaler 路径，便于随机搜索后自动归档。
4. 模型显示名称统一使用 common_names.py。

数据文件要求：
    curve_dataset_clean.csv
    curve_clean.csv

用法示例：
    python train_time.py --model lstm --features 50 --target total
"""

import argparse
import random
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from models_time import build_time_model, default_groups

try:
    from common_names import display_name
except Exception:
    def display_name(x):
        return str(x)


# ============================================================
#  随机种子
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
#  特征加载
# ============================================================
def load_features(feature_dim: int) -> np.ndarray:
    if feature_dim == 24:
        X = pd.read_csv("curve_dataset_clean.csv", header=None).values
    elif feature_dim == 38:
        import feature_build
        X_raw = pd.read_csv("curve_dataset_clean.csv", header=None).values
        X = np.array(feature_build.build_features(X_raw))
    elif feature_dim == 50:
        import new_feature_build
        X_raw = pd.read_csv("curve_dataset_clean.csv", header=None).values
        X = np.array(new_feature_build.build_features(X_raw))
    else:
        raise ValueError(f"Unsupported feature_dim: {feature_dim}")
    assert X.shape[1] == feature_dim, f"Expected {feature_dim}, got {X.shape[1]}"
    return X


# ============================================================
#  指标与变换
# ============================================================
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def to_log(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(y, 0.0, None))


def from_log(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


def inverse_time_from_scaled(y_scaled: np.ndarray, scaler_Y: StandardScaler) -> np.ndarray:
    """scaled log1p 空间 → 原始时间尺度。"""
    y_log = scaler_Y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    return np.clip(from_log(y_log), 0.0, None)


# ============================================================
#  目标提取
# ============================================================
def extract_target(Y_all: np.ndarray, target: str, threshold: float) -> np.ndarray:
    time_axis = Y_all[:, 0:100]
    c1_curves = Y_all[:, 100:200]
    c2_curves = Y_all[:, 200:300]

    if target == "total":
        return time_axis[:, -1].astype(float)

    if target not in ("c1", "c2"):
        raise ValueError(f"Unknown target: {target}")

    curves = c1_curves if target == "c1" else c2_curves
    breakthrough_times = np.zeros(len(curves), dtype=float)
    no_bt = 0

    for i in range(len(curves)):
        t = time_axis[i]
        c = curves[i]
        over = np.where(c > threshold)[0]
        if len(over) == 0:
            breakthrough_times[i] = t[-1]
            no_bt += 1
            continue
        idx = int(over[0])
        if idx == 0:
            breakthrough_times[i] = t[0]
        else:
            t0, t1 = t[idx - 1], t[idx]
            c0, c1 = c[idx - 1], c[idx]
            if abs(c1 - c0) > 1e-10:
                breakthrough_times[i] = t0 + (threshold - c0) * (t1 - t0) / (c1 - c0)
            else:
                breakthrough_times[i] = t0

    if no_bt > 0:
        print(f"Warning: {no_bt} samples never exceeded threshold={threshold} for {target}; using total time.")

    return breakthrough_times


# ============================================================
#  PyTorch 训练
# ============================================================
def train_torch(model, X_train, Y_train, X_val, Y_val, scaler_Y, args, tag, device):
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)

    Y_train_orig_for_r2 = inverse_time_from_scaled(Y_train, scaler_Y)
    Y_val_orig_for_r2 = inverse_time_from_scaled(Y_val, scaler_Y)

    best_val_loss = 1e9
    best_val_r2 = -1e9
    best_val_mape = 1e9
    patience_loss = 0
    patience_r2 = 0

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_loss = f"checkpoints/time_{tag}_best_loss.pt"
    ckpt_r2 = f"checkpoints/time_{tag}_best_r2.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_R2": [],
        "val_R2": [],
        "val_MAPE": [],
    }

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            # train R2：在训练集上重新前向，保证和验证 R2 定义一致
            train_pred_s = model(X_train_t).detach().cpu().numpy()
            train_pred_orig = inverse_time_from_scaled(train_pred_s, scaler_Y)
            train_r2 = r2_score(Y_train_orig_for_r2, train_pred_orig)

            val_pred_s = model(X_val_t)
            val_loss = loss_fn(val_pred_s, Y_val_t).item()
            val_pred_orig = inverse_time_from_scaled(val_pred_s.detach().cpu().numpy(), scaler_Y)
            val_r2 = r2_score(Y_val_orig_for_r2, val_pred_orig)
            val_mape = mape(Y_val_orig_for_r2, val_pred_orig)

        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_R2"].append(float(train_r2))
        history["val_R2"].append(float(val_r2))
        history["val_MAPE"].append(float(val_mape))

        if epoch % 20 == 0:
            print(
                f"[{tag}] Ep {epoch:4d} | train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} train_R2={train_r2:.4f} "
                f"val_R2={val_r2:.4f} val_MAPE={val_mape:.2f}%"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_loss = 0
            torch.save(model.state_dict(), ckpt_loss)
        else:
            patience_loss += 1

        if val_r2 > best_val_r2 + 1e-5:
            best_val_r2 = val_r2
            best_val_mape = val_mape
            patience_r2 = 0
            torch.save(model.state_dict(), ckpt_r2)
        else:
            patience_r2 += 1

        if patience_loss >= args.patience and patience_r2 >= args.patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"best_val_loss={best_val_loss:.5f} best_val_R2={best_val_r2:.4f}"
            )
            break

    model.load_state_dict(torch.load(ckpt_r2, map_location=device))
    best_info = {
        "best_val_loss": float(best_val_loss),
        "best_val_R2": float(best_val_r2),
        "best_val_MAPE": float(best_val_mape),
        "checkpoint_best_loss": ckpt_loss,
        "checkpoint_best_r2": ckpt_r2,
    }
    return model, history, best_info


# ============================================================
#  主训练函数
# ============================================================
def train(args):
    set_seed(args.seed)

    tag = f"{args.model}_feat{args.features}_{args.target}"
    print(f"\n{'=' * 70}")
    print(f"  Time Experiment : {tag}")
    print(f"  Model={display_name(args.model)}  Features={args.features}  Target={args.target}")
    print(f"{'=' * 70}\n")

    X = load_features(args.features)
    Y_all = pd.read_csv("curve_clean.csv", header=None).values
    Y_time = extract_target(Y_all, args.target, args.threshold)

    print(
        f"Time target — min={Y_time.min():.2f} max={Y_time.max():.2f} "
        f"median={np.median(Y_time):.2f} mean={Y_time.mean():.2f} "
        f"std={Y_time.std():.2f} n>1000={np.sum(Y_time > 1000)}"
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_time, test_size=0.2, random_state=args.seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.15, random_state=args.seed
    )
    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(to_log(Y_train).reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(to_log(Y_train).reshape(-1, 1)).ravel()
    Y_val_s = scaler_Y.transform(to_log(Y_val).reshape(-1, 1)).ravel()

    os.makedirs("scalers", exist_ok=True)
    scaler_x_path = f"scalers/time_{tag}_scaler_X.pkl"
    scaler_y_path = f"scalers/time_{tag}_scaler_Y.pkl"
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_Y, scaler_y_path)

    groups = default_groups(args.features)
    model = build_time_model(
        args.model,
        in_dim=args.features,
        groups=groups,
        hidden=args.hidden,
        dropout=args.dropout,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        lstm_layers=args.lstm_layers,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_levels=args.num_levels,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        random_state=args.seed,
    )

    is_sklearn = getattr(model, "is_sklearn", False)
    history = {}
    best_info = {
        "best_val_loss": None,
        "best_val_R2": None,
        "best_val_MAPE": None,
        "checkpoint_best_loss": None,
        "checkpoint_best_r2": None,
    }

    if is_sklearn:
        print(f"Fitting sklearn model: {model} ...")
        model.fit(X_train_s, Y_train_s)

        train_pred_orig = inverse_time_from_scaled(model.predict(X_train_s), scaler_Y)
        val_pred_orig = inverse_time_from_scaled(model.predict(X_val_s), scaler_Y)
        train_r2 = r2_score(Y_train, train_pred_orig)
        val_r2 = r2_score(Y_val, val_pred_orig)
        val_mape = mape(Y_val, val_pred_orig)

        os.makedirs("checkpoints", exist_ok=True)
        sklearn_path = f"checkpoints/time_{tag}.pkl"
        joblib.dump(model.model, sklearn_path)
        total_params = "N/A (sklearn)"
        best_info.update({
            "best_val_R2": float(val_r2),
            "best_val_MAPE": float(val_mape),
            "checkpoint_sklearn": sklearn_path,
        })
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_R2": [float(train_r2)],
            "val_R2": [float(val_r2)],
            "val_MAPE": [float(val_mape)],
        }
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,}")
        print(f"Groups    : {[(n, s) for n, s in zip(groups.names, groups.slices)]}")

        model, history, best_info = train_torch(
            model, X_train_s, Y_train_s, X_val_s, Y_val_s,
            scaler_Y, args, tag, device,
        )

    # ── Test 评估 ─────────────────────────────────────────────
    print(f"\n--- Test Evaluation: {tag} ---")
    if is_sklearn:
        pred_scaled = model.predict(X_test_s)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

    Y_pred = inverse_time_from_scaled(pred_scaled, scaler_Y)
    Y_true = Y_test

    metrics = {
        "experiment": tag,
        "task": "time",
        "model": args.model,
        "model_display": display_name(args.model),
        "features": args.features,
        "target": args.target,
        "threshold": args.threshold,
        "params": total_params,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "num_blocks": args.num_blocks,
        "num_layers": args.num_layers,
        "lstm_layers": args.lstm_layers,
        "channels": args.channels,
        "kernel_size": args.kernel_size,
        "num_levels": args.num_levels,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "R2": float(r2_score(Y_true, Y_pred)),
        "MAE": float(mean_absolute_error(Y_true, Y_pred)),
        "MAPE(%)": float(mape(Y_true, Y_pred)),
        "RMSE": float(mean_squared_error(Y_true, Y_pred) ** 0.5),
        "best_val_R2": best_info.get("best_val_R2"),
        "best_val_MAPE": best_info.get("best_val_MAPE"),
        "best_val_loss": best_info.get("best_val_loss"),
        "checkpoint_best_r2": best_info.get("checkpoint_best_r2"),
        "checkpoint_best_loss": best_info.get("checkpoint_best_loss"),
        "checkpoint_sklearn": best_info.get("checkpoint_sklearn"),
        "scaler_X": scaler_x_path,
        "scaler_Y": scaler_y_path,
    }

    print(f"\n===== Results: {tag} =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/time_{tag}_metrics.json"
    history_path = f"results/time_{tag}_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics: {metrics_path}")
    print(f"History: {history_path}")
    return metrics


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train breakthrough time prediction model")

    parser.add_argument("--model", type=str, default="resnet",
                        choices=["resnet", "mlp", "lstm", "transformer", "rf", "xgb", "lgb"])
    parser.add_argument("--features", type=int, default=24, choices=[24, 38, 50])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target", type=str, default="total", choices=["total", "c1", "c2"])
    parser.add_argument("--threshold", type=float, default=0.01)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_blocks", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lstm_layers", type=int, default=3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_levels", type=int, default=5)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=None)

    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)

    args = parser.parse_args()
    train(args)
