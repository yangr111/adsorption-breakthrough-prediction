"""
models_time.py — 穿透时间预测模型

模型列表：
  resnet      ResNet MLP
  mlp         普通多层全连接
  lstm        分组感知双向 LSTM
  transformer 分组 token Transformer Encoder
  svr         Support Vector Regression
  rf          Random Forest
  xgb         XGBoost
  lgb         LightGBM
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FeatureGroups:
    """
    描述输入特征的物理分组。

    示例（24维）：
        groups = FeatureGroups(
            names  = ["adsorbent", "gas_a", "gas_b"],
            slices = [slice(0,6), slice(6,15), slice(15,24)]
        )
    """
    names:  List[str]
    slices: List[slice]

    @property
    def num_groups(self) -> int:
        return len(self.slices)

    @property
    def group_sizes(self) -> List[int]:
        return [s.stop - s.start for s in self.slices]

    def split(self, x: torch.Tensor) -> List[torch.Tensor]:
        """把 (B, in_dim) 按分组切成 list of (B, group_size)"""
        return [x[:, s] for s in self.slices]


def default_groups(in_dim: int) -> FeatureGroups:
    """
    根据特征总维度返回默认分组。
    24维：吸附剂6 | 气体A9 | 气体B9
    38维：吸附剂6 | 气体A16 | 气体B16
    50维：吸附剂6 | 气体A22 | 气体B22
    （如果你的实际分法不同，在调用 build_model 时直接传 groups=FeatureGroups(...)）
    """
    if in_dim == 24:
        return FeatureGroups(
            names  = ["adsorbent", "gas_a", "gas_b"],
            slices = [slice(0, 6), slice(6, 15), slice(15, 24)]
        )
    elif in_dim == 38:
        return FeatureGroups(
            names  = ["adsorbent", "gas_a", "gas_b"],
            slices = [slice(0, 6), slice(6, 22), slice(22, 38)]
        )
    elif in_dim == 50:
        return FeatureGroups(
            names  = ["adsorbent", "gas_a", "gas_b"],
            slices = [slice(0, 6), slice(6, 28), slice(28, 50)]
        )
    else:
        # 无法推断时，退化为一组
        return FeatureGroups(
            names  = ["all"],
            slices = [slice(0, in_dim)]
        )


# ============================================================
#  通用标量回归头
# ============================================================
def _scalar_head(hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1)
    )


# ============================================================
#  ResNet
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h

class ResNetTime(nn.Module):
    def __init__(self, in_dim, hidden=256, num_blocks=5, dropout=0.05, groups=None):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(num_blocks)])
        self.head   = _scalar_head(hidden, dropout)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


# ============================================================
#  MLP
# ============================================================
class MLPTime(nn.Module):
    def __init__(self, in_dim, hidden=256, num_layers=6, dropout=0.05, groups=None):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.net  = nn.Sequential(*layers)
        self.head = _scalar_head(hidden, dropout)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


# ============================================================
#  LSTM（分组感知）
# ============================================================
class LSTMTime(nn.Module):
    def __init__(self, in_dim, hidden=256, lstm_layers=3, dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)
        self.group_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(s.stop - s.start, hidden), nn.GELU())
            for s in self.groups.slices
        ])
        self.lstm = nn.LSTM(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.head = _scalar_head(hidden * 2, dropout)

    def forward(self, x):
        tokens = [embed(x[:, s]).unsqueeze(1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq    = torch.cat(tokens, dim=1)
        out, _ = self.lstm(seq)
        pooled = self.drop(self.norm(out.mean(dim=1)))
        return self.head(pooled).squeeze(-1)


# ============================================================
#  Transformer（分组感知）
# ============================================================
class TransformerTime(nn.Module):
    def __init__(self, in_dim, hidden=256, nhead=8, num_layers=4,
                 dim_feedforward=None, dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)
        for h in [nhead, 8, 4, 2, 1]:
            if hidden % h == 0:
                nhead = h
                break
        dim_feedforward = dim_feedforward or hidden * 4

        self.group_embeds = nn.ModuleList([
            nn.Linear(s.stop - s.start, hidden)
            for s in self.groups.slices
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.groups.num_groups, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(hidden)
        self.head    = _scalar_head(hidden, dropout)

    def forward(self, x):
        tokens = [embed(x[:, s]).unsqueeze(1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq = torch.cat(tokens, dim=1) + self.pos_embed
        out = self.norm(self.encoder(seq).mean(dim=1))
        return self.head(out).squeeze(-1)


# ============================================================
#  ML 基线包装器（RF / XGB / LGB）
# ============================================================
class SklearnWrapper:
    """
    统一接口包装器。is_sklearn=True 是判断标志。
    XGB/LGB 也通过此类统一管理。
    """
    is_sklearn = True

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type

        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators     = kwargs.get("n_estimators", 300),
                max_depth        = kwargs.get("max_depth", None),
                min_samples_leaf = kwargs.get("min_samples_leaf", 2),
                max_features     = kwargs.get("max_features", "sqrt"),
                n_jobs           = -1,
                random_state     = kwargs.get("random_state", 42),
            )

        elif model_type == "xgb":
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators     = kwargs.get("n_estimators", 500),
                max_depth        = kwargs.get("max_depth", 6),
                learning_rate    = kwargs.get("learning_rate", 0.05),
                subsample        = kwargs.get("subsample", 0.8),
                colsample_bytree = kwargs.get("colsample_bytree", 0.8),
                min_child_weight = kwargs.get("min_child_weight", 1),
                reg_alpha        = kwargs.get("reg_alpha", 0.0),
                reg_lambda       = kwargs.get("reg_lambda", 1.0),
                n_jobs           = -1,
                random_state     = kwargs.get("random_state", 42),
                verbosity        = 0,
            )

        elif model_type == "lgb":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators     = kwargs.get("n_estimators", 500),
                max_depth        = kwargs.get("max_depth", -1),
                num_leaves       = kwargs.get("num_leaves", 63),
                learning_rate    = kwargs.get("learning_rate", 0.05),
                subsample        = kwargs.get("subsample", 0.8),
                colsample_bytree = kwargs.get("colsample_bytree", 0.8),
                min_child_samples= kwargs.get("min_child_samples", 20),
                reg_alpha        = kwargs.get("reg_alpha", 0.0),
                reg_lambda       = kwargs.get("reg_lambda", 1.0),
                n_jobs           = -1,
                random_state     = kwargs.get("random_state", 42),
                verbosity        = -1,
            )

        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __repr__(self):
        return f"SklearnWrapper({self.model_type})"


# ============================================================
#  工厂函数
# ============================================================
_TORCH_CLS = {
    "resnet":      ResNetTime,
    "mlp":         MLPTime,
    "lstm":        LSTMTime,
    "transformer": TransformerTime,
}

_TORCH_KWARGS = {
    "resnet":      {"num_blocks"},
    "mlp":         {"num_layers"},
    "lstm":        {"lstm_layers"},
    "transformer": {"nhead", "num_layers", "dim_feedforward"},
}

_SKLEARN_TYPES = {"rf", "xgb", "lgb"}


def build_time_model(model_type: str, in_dim: int,
                     groups: FeatureGroups = None, **kwargs):
    """
    Args:
        model_type : resnet/mlp/lstm/transformer/svr/rf/xgb/lgb
        in_dim     : 输入特征总维度
        groups     : FeatureGroups；None 时按 in_dim 推断
        **kwargs   : 超参数（PyTorch 模型自动过滤无关键；
                     sklearn/xgb/lgb 把相关 kwargs 透传给构造函数）
    """
    if model_type in _SKLEARN_TYPES:
        # 只把 sklearn 认识的参数传进去，过滤掉 torch 专有键
        sklearn_ignore = {"hidden", "dropout", "num_blocks", "num_layers",
                          "lstm_layers", "nhead", "dim_feedforward",
                          "channels", "kernel_size", "num_levels"}
        sk_kwargs = {k: v for k, v in kwargs.items() if k not in sklearn_ignore}
        return SklearnWrapper(model_type, **sk_kwargs)

    if model_type not in _TORCH_CLS:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available: {list(_TORCH_CLS.keys()) + list(_SKLEARN_TYPES)}"
        )
    allowed  = _TORCH_KWARGS[model_type] | {"hidden", "dropout"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return _TORCH_CLS[model_type](
        in_dim=in_dim,
        groups=groups or default_groups(in_dim),
        **filtered
    )
