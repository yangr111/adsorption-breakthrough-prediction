"""
models_conc.py — 共用模型定义

输入特征结构（24维为例）：
  [0:6]   吸附剂参数（adsorbent）
  [6:15]  气体A参数（gas_a）
  [15:24] 气体B参数（gas_b）

38维 / 50维 时，分组边界通过 FeatureGroups 传入。
输出：200维穿透曲线，前100维=C1，后100维=C2

模型列表：
  resnet      ResNet MLP（残差全连接）
  mlp         普通多层全连接
  lstm        分组编码后拼接，再送双向 LSTM
  cnn         分组感知的 1-D 卷积残差网络
  tcn         分组感知的 TCN（膨胀因果卷积）
  transformer 分组 token + 语义位置编码的 Transformer Encoder
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
#  特征分组描述
# ============================================================
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


class ResNetCurve(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=200, num_blocks=5,
                 dropout=0.05, groups=None):
        super().__init__()
        self.input_layer  = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.blocks        = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(num_blocks)])
        self.output_layer  = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)


# ============================================================
#  MLP
# ============================================================
class MLPCurve(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=200, num_layers=6,
                 dropout=0.05, groups=None):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
#  LSTM（分组感知版）
# ============================================================
class LSTMCurve(nn.Module):
    """
    物理感知策略：
      对每个特征分组（吸附剂/气体A/气体B）分别用独立的线性层映射到 embed_dim，
      得到3个 token，组成长度=3 的"物理意义序列"，再送双向 LSTM。

    这样 LSTM 的时序关系对应的是：
        step0=吸附剂 → step1=气体A → step2=气体B
    在物理上是"床层属性 → 组分A → 组分B"的信息流，有实际意义。
    """
    def __init__(self, in_dim, hidden=512, out_dim=200, lstm_layers=3,
                 dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)

        # 每个分组独立的嵌入层
        self.group_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(s.stop - s.start, hidden),
                nn.GELU()
            )
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
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # 每个分组嵌入 → (B, 1, hidden)，拼成序列 (B, num_groups, hidden)
        tokens = [embed(x[:, s]).unsqueeze(1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq = torch.cat(tokens, dim=1)      # (B, num_groups, hidden)

        out, _ = self.lstm(seq)             # (B, num_groups, hidden*2)
        pooled = out.mean(dim=1)            # mean pool
        pooled = self.drop(self.norm(pooled))
        return self.head(pooled)


# ============================================================
#  CNN（分组感知版）
# ============================================================
class ConvResBlock(nn.Module):
    """1-D 卷积残差块"""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.05):
        super().__init__()
        self.pad   = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=self.pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def _trim(self, x):
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        h = self.act(self.bn1(self._trim(self.conv1(x))))
        h = self.drop(h)
        h = self.bn2(self._trim(self.conv2(h)))
        return self.act(x + h)


class CNNCurve(nn.Module):
    """
    物理感知策略：
      每个分组先独立投影到 embed_dim，然后把3个 embed 向量拼成
      长度=3 的序列（channel=embed_dim）送给 1-D 卷积。
      这样卷积核在"吸附剂 | 气体A | 气体B"这个有意义的维度上滑动。
    """
    def __init__(self, in_dim, hidden=256, out_dim=200,
                 channels=128, num_blocks=4, kernel_size=3,
                 dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)

        self.group_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(s.stop - s.start, channels), nn.GELU())
            for s in self.groups.slices
        ])

        self.blocks = nn.Sequential(
            *[ConvResBlock(channels, kernel_size=min(kernel_size, self.groups.num_groups),
                           dilation=2**i, dropout=dropout)
              for i in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        tokens = [embed(x[:, s]).unsqueeze(-1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq = torch.cat(tokens, dim=-1)   # (B, channels, num_groups)

        seq = self.blocks(seq)            # (B, channels, num_groups)
        seq = seq.mean(dim=-1)            # (B, channels)
        return self.head(seq)


# ============================================================
#  TCN（分组感知版）
# ============================================================
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.05):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pad  = pad
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _trim(self, x):
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        h = self.act(self._trim(self.conv1(x)))
        h = self.drop(h)
        h = self._trim(self.conv2(h))
        res = self.skip(x) if self.skip is not None else x
        return self.act(h + res)


class TCNCurve(nn.Module):
    """
    物理感知策略与 CNNCurve 相同：
      分组嵌入 → 长度=num_groups 的序列 → 膨胀因果卷积
    """
    def __init__(self, in_dim, hidden=256, out_dim=200,
                 channels=128, num_levels=5, kernel_size=3,
                 dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)

        self.group_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(s.stop - s.start, channels), nn.GELU())
            for s in self.groups.slices
        ])

        blocks = []
        for i in range(num_levels):
            in_ch = channels  # 所有层统一 channels（第一层也是 channels，因为已经嵌入过了）
            blocks.append(TCNBlock(in_ch, channels, kernel_size,
                                   dilation=2**i, dropout=dropout))
        self.tcn  = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        tokens = [embed(x[:, s]).unsqueeze(-1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq = torch.cat(tokens, dim=-1)   # (B, channels, num_groups)
        seq = self.tcn(seq)               # (B, channels, num_groups)
        seq = seq.mean(dim=-1)            # (B, channels)
        return self.head(seq)


# ============================================================
#  Transformer（分组感知版）
# ============================================================
class TransformerCurve(nn.Module):
    """
    物理感知策略：
      每个分组作为一个 token，用组名做"语义位置编码"。
      token 顺序：[adsorbent(0), gas_a(1), gas_b(2)]
      位置编码是可学习的，长度固定为 num_groups，不是 in_dim。

    好处：注意力直接在"吸附剂 ↔ 气体A ↔ 气体B"三者之间计算，
    而不是在 24/38/50 个独立标量之间。
    """
    def __init__(self, in_dim, hidden=256, out_dim=200,
                 nhead=8, num_layers=4, dim_feedforward=None,
                 dropout=0.05, groups=None):
        super().__init__()
        self.groups = groups or default_groups(in_dim)

        # 保证 hidden % nhead == 0
        for h in [nhead, 8, 4, 2, 1]:
            if hidden % h == 0:
                nhead = h
                break
        dim_feedforward = dim_feedforward or hidden * 4

        # 每个分组独立嵌入
        self.group_embeds = nn.ModuleList([
            nn.Linear(s.stop - s.start, hidden)
            for s in self.groups.slices
        ])
        # 可学习语义位置编码（长度=num_groups，对应分组数量）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.groups.num_groups, hidden)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = hidden,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(hidden)
        self.head    = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # 每个分组 → (B, 1, hidden)，拼成 (B, num_groups, hidden)
        tokens = [embed(x[:, s]).unsqueeze(1)
                  for embed, s in zip(self.group_embeds, self.groups.slices)]
        seq = torch.cat(tokens, dim=1)         # (B, num_groups, hidden)
        seq = seq + self.pos_embed             # 加语义位置编码
        seq = self.encoder(seq)                # (B, num_groups, hidden)
        out = self.norm(seq.mean(dim=1))       # mean pool → (B, hidden)
        return self.head(out)                  # (B, out_dim)


# ============================================================
#  Weighted SmoothL1 Loss（前沿加权）
# ============================================================
class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.1, w_front=2.0, front_points=30):
        super().__init__()
        self.beta         = beta
        self.w_front      = w_front
        self.front_points = front_points

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        w = torch.ones(pred.shape[1], device=pred.device)
        w[0:self.front_points]         = self.w_front   # C1 前沿
        w[100:100 + self.front_points] = self.w_front   # C2 前沿
        return (loss * w.unsqueeze(0)).mean()


def smoothness_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """
    对 C1（前100维）和 C2（后100维）分别计算平滑损失，
    避免在两条曲线的拼接点处做无意义的差分。
    """
    c1_diff = y_pred[:, 1:100]  - y_pred[:, :99]    # C1 内部差分
    c2_diff = y_pred[:, 101:200] - y_pred[:, 100:199] # C2 内部差分
    return (torch.mean(c1_diff ** 2) + torch.mean(c2_diff ** 2)) / 2.0


# ============================================================
#  工厂函数
# ============================================================
_MODEL_CLS = {
    "resnet":      ResNetCurve,
    "mlp":         MLPCurve,
    "lstm":        LSTMCurve,
    "cnn":         CNNCurve,
    "tcn":         TCNCurve,
    "transformer": TransformerCurve,
}

_MODEL_KWARGS = {
    "resnet":      {"num_blocks"},
    "mlp":         {"num_layers"},
    "lstm":        {"lstm_layers"},
    "cnn":         {"channels", "num_blocks", "kernel_size"},
    "tcn":         {"channels", "num_levels", "kernel_size"},
    "transformer": {"nhead", "num_layers", "dim_feedforward"},
}


def build_model(model_type: str, in_dim: int,
                groups: FeatureGroups = None, **kwargs) -> nn.Module:
    """
    工厂函数，自动过滤无关 kwargs。

    Args:
        model_type : resnet / mlp / lstm / cnn / tcn / transformer
        in_dim     : 输入特征总维度
        groups     : FeatureGroups，描述物理分组；为 None 时按 in_dim 推断默认分组
        **kwargs   : 超参数（无关键会被忽略）
    """
    if model_type not in _MODEL_CLS:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available: {list(_MODEL_CLS.keys())}"
        )
    allowed  = _MODEL_KWARGS[model_type] | {"hidden", "out_dim", "dropout"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return _MODEL_CLS[model_type](
        in_dim=in_dim,
        groups=groups or default_groups(in_dim),
        **filtered
    )
