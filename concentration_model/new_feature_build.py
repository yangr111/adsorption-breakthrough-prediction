"""构建不同压力下的吸附量 q 特征（离散压力点 50特征 保留原有吸附等温线参数）"""

import numpy as np

def langmuir_freundlich(qmax, b, v, p):
    """Langmuir–Freundlich 等温线 q(p)"""
    return qmax * (b * (p ** v)) / (1.0 + b * (p ** v))


def extract_params_from_row(row):
    """
    从 curve_dataset.csv 的一行提取 24 个参数并拆分：
    - 吸附剂参数（6）
    - 组分 A 参数（9）
    - 组分 B 参数（9）
    """
    adsorbent = row[0:6]          # 6 个吸附剂参数

    compA = row[6:15]             # 9 个组分 A 参数
    compB = row[15:24]            # 9 个组分 B 参数

    # 气体的 3 个条件参数
    gasA_main = compA[0:3]
    gasB_main = compB[0:3]

    # Langmuir-Freundlich 参数（A）
    qA1, bA1, vA1 = compA[3:6]
    qA2, bA2, vA2 = compA[6:9]

    # Langmuir-Freundlich 参数（B）
    qB1, bB1, vB1 = compB[3:6]
    qB2, bB2, vB2 = compB[6:9]

    LF_A = [(qA1, bA1, vA1), (qA2, bA2, vA2)]
    LF_B = [(qB1, bB1, vB1), (qB2, bB2, vB2)]

    return adsorbent, gasA_main, gasB_main, LF_A, LF_B


def build_features(X_raw, p_list=None):
    """
    使用离散压力点计算等温线吸附量特征

    默认压力点（Pa）：
    10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5

    """

    if p_list is None:
        p_list = [10, 20, 50, 100, 200, 500,
                  1000, 2000, 5000, 10000, 20000, 50000, 100000]

    p_list = np.asarray(p_list, dtype=np.float64)
    num_p = len(p_list)

    N = X_raw.shape[0]
    all_features = np.zeros((N, 24 + 2 * num_p), dtype=np.float32)

    for i in range(N):
        row = X_raw[i].astype(np.float64)

        adsorbent, gasA_main, gasB_main, LF_A, LF_B = extract_params_from_row(row)

        # ---- 计算 A 的吸附量 qA(p) ----
        qA_vals = np.zeros(num_p, dtype=np.float64)
        for (qmax, b, v) in LF_A:
            qA_vals += langmuir_freundlich(qmax, b, v, p_list)

        # ---- 计算 B 的吸附量 qB(p) ----
        qB_vals = np.zeros(num_p, dtype=np.float64)
        for (qmax, b, v) in LF_B:
            qB_vals += langmuir_freundlich(qmax, b, v, p_list)

        LF_A_params = np.reshape(LF_A, (-1, ))
        LF_B_params = np.reshape(LF_B, (-1, ))

        feat = np.concatenate([
            adsorbent,           # 6
            gasA_main,           # 3
            LF_A_params,         # 6
            qA_vals,             # len(p_list)
            gasB_main,           # 3
            LF_B_params,         # 6
            qB_vals              # len(p_list)
        ], axis=0)

        all_features[i] = feat.astype(np.float32)

    return all_features
