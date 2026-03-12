# data_utils.py
# ============================================================
# 数据工具 — 支持留一法跨电池验证
# ============================================================

import numpy as np
import extract


def create_sequences(X, y, seq_len=3):
    """
    将原始数据打包成滑动窗口时间序列。
    序列 i 对应的目标 = y[i + seq_len - 1]
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)


def normalize_images(X_train, X_test):
    """
    逐通道 Z-score 标准化
    在训练集上计算每个通道的 mean 和 std，
    然后同时标准化训练集和测试集（避免数据泄漏）。
    """
    if X_train.ndim == 5:
        N, S, C, H, W = X_train.shape
        flat = X_train.reshape(-1, C, H, W)
    else:
        flat = X_train
        C = flat.shape[1]

    mean = flat.mean(axis=(0, 2, 3), keepdims=True)
    std  = flat.std(axis=(0, 2, 3), keepdims=True) + 1e-8

    if X_train.ndim == 5:
        mean5 = mean.reshape(1, 1, C, 1, 1)
        std5  = std.reshape(1, 1, C, 1, 1)
        X_train = (X_train - mean5) / std5
        X_test  = (X_test  - mean5) / std5
    else:
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

    return X_train.astype(np.float32), X_test.astype(np.float32)


def load_single_battery(folder, bat_id, sheets):
    """
    加载单个电池的数据，返回数据字典。
    """
    eis_raw, cap_raw = extract.extract(folder, [f'{bat_id}.xlsx'], sheets)

    if len(eis_raw) == 0 or len(cap_raw) == 0:
        raise ValueError(f"未提取到电池 {bat_id} 的数据！请检查路径和Sheet。")

    num_cycles = len(cap_raw)
    eis_cycles = np.zeros((num_cycles, 60, 3))
    for i in range(num_cycles):
        cycle_data = eis_raw[i*60 : (i+1)*60]
        eis_cycles[i, :, 0] = cycle_data[:, 1]
        eis_cycles[i, :, 1] = cycle_data[:, 2]
        eis_cycles[i, :, 2] = -cycle_data[:, 3]

    Re    = eis_cycles[:, :, 1]
    Im    = eis_cycles[:, :, 2]

    return {
        're':    Re,
        'im':    Im,
        'soh':   cap_raw,
    }


def load_and_split_sequence_data(folder, files, sheets, seq_len=5):
    """时序版：提取数据并返回数据字典（兼容原有接口）"""
    eis_raw, cap_raw = extract.extract(folder, files, sheets)

    if len(eis_raw) == 0 or len(cap_raw) == 0:
        raise ValueError("未提取到数据！请检查路径和Sheet。")

    num_cycles = len(cap_raw)
    eis_cycles = np.zeros((num_cycles, 60, 3))
    for i in range(num_cycles):
        cycle_data = eis_raw[i*60 : (i+1)*60]
        eis_cycles[i, :, 0] = cycle_data[:, 1]
        eis_cycles[i, :, 1] = cycle_data[:, 2]
        eis_cycles[i, :, 2] = -cycle_data[:, 3]

    Re    = eis_cycles[:, :, 1]
    Im    = eis_cycles[:, :, 2]
    Phase = np.arctan2(Im, Re)
    Mag   = np.sqrt(Re**2 + Im**2)

    return {
        'freq':  eis_cycles[:, :, 0],
        're':    Re,
        'im':    Im,
        'phase': Phase,
        'mag':   Mag,
        'soh':   cap_raw,
    }
