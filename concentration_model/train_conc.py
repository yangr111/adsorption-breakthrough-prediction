"""
train_conc.py — 浓度曲线预测训练脚本（新版）

主要改动：
1. 训练过程统一保存 train_loss、val_loss、train_R2、val_R2，便于后续每个指标单独作图。
2. 保存 best_val_loss 和 best_val_R2 两个 checkpoint，最终测试默认使用 best_val_R2。
3. metrics.json 中保存模型名称、超参数、checkpoint/scaler 路径，便于随机搜索后自动归档。
4. 模型显示名称统一使用 common_names.py。

数据文件要求：
    curve_dataset_clean.csv
    curve_clean.csv

用法示例：
    python train_conc.py --model resnet --features 50
"""

import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os

from models_conc import build_model, WeightedSmoothL1Loss, smoothness_loss, default_groups

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
        print("Loading 24-dim features from curve_dataset_clean.csv ...")
        X = pd.read_csv("curve_dataset_clean.csv", header=None).values
    elif feature_dim == 38:
        print("Building 38-dim features via feature_build.build_features() ...")
        import feature_build
        X_raw = pd.read_csv("curve_dataset_clean.csv", header=None).values
        X = np.array(feature_build.build_features(X_raw))
    elif feature_dim == 50:
        print("Building 50-dim features via new_feature_build.build_features() ...")
        import new_feature_build
        X_raw = pd.read_csv("curve_dataset_clean.csv", header=None).values
        X = np.array(new_feature_build.build_features(X_raw))
    else:
        raise ValueError(f"Unsupported feature_dim: {feature_dim}.")

    assert X.shape[1] == feature_dim, f"Expected {feature_dim}, got {X.shape[1]}."
    return X


# ============================================================
#  批量预测，避免全量前向时显存过大
# ============================================================
def predict_in_batches(model, X_array, device, batch_size=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_array), batch_size):
            xb = torch.FloatTensor(X_array[i:i + batch_size]).to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.vstack(preds)


# ============================================================
#  主训练函数
# ============================================================
def train(args):
    set_seed(args.seed)

    tag = f"{args.model}_feat{args.features}"
    print(f"\n{'=' * 70}")
    print(f"  Concentration Experiment : {tag}")
    print(f"  Model={display_name(args.model)}  Features={args.features}")
    print(f"{'=' * 70}\n")

    # ── 1. Data ──────────────────────────────────────────────
    X = load_features(args.features)
    Y_all = pd.read_csv("curve_clean.csv", header=None).values
    Y_conc = Y_all[:, 100:300]

    # ── 2. Split + scaler ────────────────────────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_conc, test_size=0.2, random_state=args.seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.15, random_state=args.seed
    )
    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)

    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    Y_train_s = scaler_Y.transform(Y_train)
    Y_val_s = scaler_Y.transform(Y_val)

    os.makedirs("scalers", exist_ok=True)
    scaler_x_path = f"scalers/{tag}_scaler_X.pkl"
    scaler_y_path = f"scalers/{tag}_scaler_Y.pkl"
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_Y, scaler_y_path)

    # ── 3. DataLoader ─────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(Y_train_s)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # ── 4. Model ──────────────────────────────────────────────
    groups = default_groups(args.features)
    model = build_model(
        args.model,
        in_dim=args.features,
        groups=groups,
        hidden=args.hidden,
        out_dim=200,
        dropout=args.dropout,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        lstm_layers=args.lstm_layers,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_levels=args.num_levels,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Feature groups  : {[(g, s) for g, s in zip(groups.names, groups.slices)]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    data_loss_fn = WeightedSmoothL1Loss(beta=0.1, w_front=2.0, front_points=30)

    # ── 5. Training loop ──────────────────────────────────────
    best_val_loss = 1e9
    best_val_r2 = -1e9
    patience_loss = 0
    patience_r2 = 0

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_loss = f"checkpoints/conc_{tag}_best_loss.pt"
    ckpt_r2 = f"checkpoints/conc_{tag}_best_r2.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_R2": [],
        "val_R2": [],
    }

    X_val_t = torch.FloatTensor(X_val_s).to(device)
    Y_val_t = torch.FloatTensor(Y_val_s).to(device)

    # 反标准化后的真实值，用于 R2
    Y_train_true_for_r2 = Y_train
    Y_val_true_for_r2 = Y_val

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = data_loss_fn(pred, yb) + args.smooth_weight * smoothness_loss(pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_pred_s = model(X_val_t)
            val_data = data_loss_fn(val_pred_s, Y_val_t)
            val_smooth = smoothness_loss(val_pred_s)
            val_loss = (val_data + args.smooth_weight * val_smooth).item()

        # R2 统一在原始浓度尺度上计算
        train_pred_s = predict_in_batches(model, X_train_s, device, batch_size=args.r2_batch_size)
        val_pred_np_s = val_pred_s.detach().cpu().numpy()

        train_pred = scaler_Y.inverse_transform(train_pred_s)
        val_pred = scaler_Y.inverse_transform(val_pred_np_s)
        train_pred = np.clip(train_pred, 0.0, None)
        val_pred = np.clip(val_pred, 0.0, None)

        train_r2 = r2_score(Y_train_true_for_r2, train_pred, multioutput="variance_weighted")
        val_r2 = r2_score(Y_val_true_for_r2, val_pred, multioutput="variance_weighted")

        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_R2"].append(float(train_r2))
        history["val_R2"].append(float(val_r2))

        if epoch % 20 == 0:
            print(
                f"[{tag}] Ep {epoch:4d} | train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} train_R2={train_r2:.4f} val_R2={val_r2:.4f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_loss = 0
            torch.save(model.state_dict(), ckpt_loss)
        else:
            patience_loss += 1

        if val_r2 > best_val_r2 + 1e-5:
            best_val_r2 = val_r2
            patience_r2 = 0
            torch.save(model.state_dict(), ckpt_r2)
        else:
            patience_r2 += 1

        if patience_loss >= args.patience and patience_r2 >= args.patience:
            print("\nEarly stopping at epoch {}.".format(epoch))
            print(f"  Best val_loss = {best_val_loss:.6f}")
            print(f"  Best val_R2   = {best_val_r2:.4f}")
            break

    # ── 6. Test evaluation ────────────────────────────────────
    print("\n--- Evaluating best-R2 checkpoint ---")
    model.load_state_dict(torch.load(ckpt_r2, map_location=device))
    Y_pred_s = predict_in_batches(model, X_test_s, device, batch_size=args.r2_batch_size)
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)
    Y_true = Y_test
    Y_pred = np.clip(Y_pred, 0.0, None)

    C1_true, C2_true = Y_true[:, :100], Y_true[:, 100:]
    C1_pred, C2_pred = Y_pred[:, :100], Y_pred[:, 100:]

    metrics = {
        "experiment": tag,
        "task": "concentration",
        "model": args.model,
        "model_display": display_name(args.model),
        "features": args.features,
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
        "smooth_weight": args.smooth_weight,
        "seed": args.seed,
        "R2": float(r2_score(Y_true, Y_pred, multioutput="variance_weighted")),
        "R2_C1": float(r2_score(C1_true, C1_pred, multioutput="variance_weighted")),
        "R2_C2": float(r2_score(C2_true, C2_pred, multioutput="variance_weighted")),
        "MAE": float(mean_absolute_error(Y_true, Y_pred)),
        "RMSE": float(mean_squared_error(Y_true, Y_pred) ** 0.5),
        "best_val_loss": float(best_val_loss),
        "best_val_R2": float(best_val_r2),
        "checkpoint_best_r2": ckpt_r2,
        "checkpoint_best_loss": ckpt_loss,
        "scaler_X": scaler_x_path,
        "scaler_Y": scaler_y_path,
    }

    print(f"\n===== Results: {tag} =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/{tag}_metrics.json"
    history_path = f"results/{tag}_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nCheckpoint (R2)  : {ckpt_r2}")
    print(f"Checkpoint (loss): {ckpt_loss}")
    print(f"Metrics          : {metrics_path}")
    print(f"History          : {history_path}")
    return metrics


# ============================================================
#  CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train breakthrough-curve concentration model")

    parser.add_argument("--model", type=str, default="resnet",
                        choices=["resnet", "mlp", "lstm", "cnn", "tcn", "transformer"])
    parser.add_argument("--features", type=int, default=24, choices=[24, 38, 50])
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--r2_batch_size", type=int, default=4096,
                        help="Batch size used when computing train/val R2 curves.")

    args = parser.parse_args()
    train(args)
