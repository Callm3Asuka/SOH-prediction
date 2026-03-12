# extract.py — 与其他文件夹保持一致
import pandas as pd
import numpy as np
import os

def capacity_extract(file_path, sheet_name='Capacity'):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='calamine')
        df.columns = pd.Index([str(c).strip() for c in df.columns])
        required_cols = ['cycle number', 'ox/red', 'Capacity/mA.h']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Columns mismatch in {file_path} sheet '{sheet_name}'.")
            return np.array([])
        data = df[required_cols].values
        capacity_list = []
        for i in range(data.shape[0] - 1):
            if (data[i+1, 1] - data[i, 1]) == 1:
                capacity_list.append(data[i, 2])
        return np.array(capacity_list)
    except Exception as e:
        print(f"Error extracting capacity: {e}")
        return np.array([])

def eis_extract(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='calamine')
        df.columns = pd.Index([str(c).strip() for c in df.columns])
        cols = ['cycle number', 'freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm']
        if not all(col in df.columns for col in cols):
            return np.array([])
        return df[cols].values
    except Exception:
        return np.array([])

def extract(folder, file_names, sheet_names):
    all_eis, all_capacity = [], []
    for fname in file_names:
        file_path = os.path.join(folder, fname)
        if not os.path.exists(file_path):
            continue
        cap_raw = capacity_extract(file_path, sheet_name='Capacity')
        if len(cap_raw) == 0:
            continue
        for sheet in sheet_names:
            eis_raw = eis_extract(file_path, sheet)
            if len(eis_raw) == 0:
                continue
            max_cycle_eis = int(np.max(eis_raw[:, 0])) if len(eis_raw) > 0 else 0
            num = min(len(cap_raw), max_cycle_eis)
            if num == 0:
                continue
            points_per_cycle = 60
            if eis_raw.shape[0] < num * points_per_cycle:
                num = eis_raw.shape[0] // points_per_cycle
            eis_subset = eis_raw[:num * points_per_cycle, :]
            soh_subset = cap_raw[:num] / np.max(cap_raw)
            all_eis.append(eis_subset)
            all_capacity.append(soh_subset)
    if not all_eis:
        return np.array([]), np.array([])
    final_eis = np.vstack(all_eis)
    final_capacity = np.concatenate(all_capacity)
    points_per_cycle = 60
    total_samples = len(final_eis) // points_per_cycle
    new_cycles = np.repeat(np.arange(1, total_samples + 1), points_per_cycle)
    final_eis[:, 0] = new_cycles
    return final_eis, final_capacity
