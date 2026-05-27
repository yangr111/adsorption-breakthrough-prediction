# -*- coding: utf-8 -*-
"""统一模型名称，供训练、搜索和画图使用。"""

MODEL_DISPLAY_NAMES = {
    "mlp": "MLP",
    "resnet": "ResNet",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "svr": "SVR",
    "rf": "RF",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
}

DL_MODELS = ["mlp", "resnet", "lstm", "transformer"]
CONC_DL_MODELS = ["mlp", "resnet", "lstm", "transformer"]
TIME_ML_MODELS = ["rf", "xgb", "lgb"]


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_key).lower(), str(model_key))
