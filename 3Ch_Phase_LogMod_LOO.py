# 策略 — 留一法跨电池 SOH 预测（三通道相位/对数模特征图 + SE-ResNet-Transformer）
"""
编码策略：
  使用三通道 EIS 特征图，分别编码相位差、相位和与对数模相关信息，
  配合 SE-ResNet + Transformer 模型进行跨电池 SOH 预测。

  设 φ_k = arctan(Im_k / Re_k) 为频率点 k 处的阻抗相位角，
     M_k = sqrt(Re_k² + Im_k²) 为频率点 k 处的阻抗模值，则：

  Ch0 (纯相位差场):  Matrix[k, l] = φ_k - φ_l
      反对称矩阵，捕捉任意两频率间的相位差异，对电池老化敏感。

  Ch1 (纯相位和场):  Matrix[k, l] = φ_k + φ_l
      对称矩阵，捕捉两频率相位的累积信息。

  Ch2 (对数模相关场): Matrix[k, l] = log(M_k) × log(M_l)
      对称正定矩阵，通过对数压缩幅值动态范围后做外积，
      对阻抗幅值的频率间相关性进行编码。

验证策略：
  留一法 (Leave-One-Out, LOO) 跨电池验证
  电池池: 25C01, 25C02, 25C03, 25C06, 25C07
  每轮以 1 个电池为测试集，其余 4 个电池为训练集。
  这样可以评估模型在【未见过的电池】上的泛化能力。

可复现性：
  全局随机种子固定为 42，确保结果可复现。
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_utils import load_single_battery, create_sequences, normalize_images
from se_resnet_model import SEResNet_Transformer
from train_utils import train_model

# ── 随机种子（确保可复现性）──
SEED = 42


def set_seed(seed=SEED):
    """固定所有随机源，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family':    'Times New Roman',
    'font.size':      12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize':11,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
    'axes.linewidth': 1.2,
    'lines.linewidth':1.8,
    'figure.dpi':     150,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
})

# ── 超参数 ──
BATTERY_IDS = ['25C01', '25C02', '25C03', '25C06', '25C07']
SEQ_LEN     = 5
EPOCHS      = 300
BATCH_SIZE  = 16
LR          = 2e-4
FOLDER      = r'C:\Users\admin\Desktop\肖凯旗的文件夹\txt转xlsx'


def create_3ch_phase_logmod_images(re, im):
    """
    三通道 EIS 特征图编码。

    参数:
        re: ndarray, shape (N_cycles, 60) — 各循环各频率点的 Re(Z)
        im: ndarray, shape (N_cycles, 60) — 各循环各频率点的 Im(Z)

    返回:
        img: ndarray, shape (N_cycles, 3, 60, 60)
            - Ch0: 纯相位差场  Matrix[k,l] = φ_k - φ_l
            - Ch1: 纯相位和场  Matrix[k,l] = φ_k + φ_l
            - Ch2: 对数模相关场 Matrix[k,l] = log(|Z_k|) × log(|Z_l|)
    """
    N = re.shape[0]
    img = np.zeros((N, 3, 60, 60), dtype=np.float32)

    for i in range(N):
        re_i = re[i]   # (60,)
        im_i = im[i]   # (60,)

        # 各频率点的相位角 φ_k = arctan(Im_k / Re_k)
        phi = np.arctan2(im_i, re_i)   # (60,)

        # Ch0: 相位差场 — 反对称矩阵，对角线为 0
        img[i, 0] = np.subtract.outer(phi, phi)   # φ_k - φ_l

        # Ch1: 相位和场 — 对称矩阵
        img[i, 1] = np.add.outer(phi, phi)         # φ_k + φ_l

        # Ch2: 对数模相关场 — 外积后对称矩阵
        mod = np.sqrt(re_i ** 2 + im_i ** 2)
        # 避免 log(0)：将模值限制在最小正数
        log_mod = np.log(np.maximum(mod, 1e-10))   # (60,)
        img[i, 2] = np.outer(log_mod, log_mod)     # log(|Z_k|) × log(|Z_l|)

    return img


def load_battery_sequences(bat_id):
    """加载单个电池数据并生成三通道图像序列。"""
    data = load_single_battery(FOLDER, bat_id, ['Stage5'])
    X_img = create_3ch_phase_logmod_images(data['re'], data['im'])
    X_seq, y_seq = create_sequences(X_img, data['soh'], seq_len=SEQ_LEN)
    return X_seq, y_seq, data['soh']


def plot_feature_maps(re, im, soh, bat_id, save_dir,
                      method_name='3Ch Phase-Diff/Sum + LogMod'):
    """可视化三个代表性循环的三通道 EIS 特征图。"""
    N = re.shape[0]
    indices = [0, N // 2, N - 1]
    stage_labels = ['Early Cycle', 'Mid Cycle', 'Late Cycle']
    ch_labels = [
        'Ch0: Phase Diff Field\n$\\varphi_k - \\varphi_l$',
        'Ch1: Phase Sum Field\n$\\varphi_k + \\varphi_l$',
        'Ch2: Log-Mod Corr Field\n$\\ln|Z_k| \\cdot \\ln|Z_l|$',
    ]
    cmaps = ['RdBu_r', 'plasma', 'inferno']

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(
        f'Feature Map Visualization — {method_name}\nBattery {bat_id}',
        fontsize=15, fontweight='bold', y=1.02
    )

    for row, idx in enumerate(indices):
        re_i = re[idx]
        im_i = im[idx]

        phi = np.arctan2(im_i, re_i)
        ch0 = np.subtract.outer(phi, phi)
        ch1 = np.add.outer(phi, phi)
        mod = np.sqrt(re_i ** 2 + im_i ** 2)
        log_mod = np.log(np.maximum(mod, 1e-10))
        ch2 = np.outer(log_mod, log_mod)

        channels = [ch0, ch1, ch2]
        for col, (ch_img, cmap) in enumerate(zip(channels, cmaps)):
            ax = axes[row, col]
            im_obj = ax.imshow(ch_img, cmap=cmap, aspect='equal', origin='lower')
            fig.colorbar(im_obj, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(ch_labels[col], fontsize=10, pad=6)
            ax.set_xlabel('Freq index $l$', fontsize=9)
            ax.set_ylabel('Freq index $k$', fontsize=9)
            if col == 0:
                ax.text(-0.45, 0.5,
                        f'{stage_labels[row]}\nCycle {idx+1}\nSOH={soh[idx]:.3f}',
                        transform=ax.transAxes, fontsize=10,
                        va='center', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  fc='lightyellow', ec='gray', alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'feature_maps_3Ch_PhaseLogMod_{bat_id}.png')
    plt.savefig(save_path, format='png')
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_loo_results(all_results,
                     method_name='3Ch Phase-Diff/Sum + LogMod + SE-ResNet-Transformer (LOO)',
                     save_path=None):
    """绘制留一法跨电池验证结果。"""
    if save_path is None:
        save_path = os.path.join(SAVE_DIR, 'result_LOO_3Ch_PhaseLogMod.png')

    n_batteries = len(all_results)
    n_cols = 3
    n_rows = (n_batteries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    fig.suptitle(f'Battery SOH Estimation — {method_name}\n(Leave-One-Out Cross-Battery Validation)',
                 fontsize=16, fontweight='bold', y=1.02)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    test_bat_ids = list(all_results.keys())

    for idx, test_bat in enumerate(test_bat_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        res = all_results[test_bat]
        y_true  = res['y_true_test']
        y_pred  = res['y_pred_test']
        cycles  = res['test_cycles']
        train_bats = res['train_batteries']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        srt = np.argsort(cycles)
        ax.plot(cycles[srt], y_true[srt],
                color='#1f4e79', lw=2.0, label='True SOH', zorder=3)
        ax.plot(cycles[srt], y_pred[srt],
                color='#e84040', lw=1.5, ls='--',
                label='Predicted SOH', zorder=2)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('SOH')
        ax.set_title(f'Test: {test_bat}  (Train: {", ".join(train_bats)})',
                     fontweight='bold', pad=6, fontsize=10)
        ax.set_xlim(left=1)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(True, ls='--', lw=0.5, alpha=0.5)

        ax.text(0.03, 0.03,
                f'RMSE={rmse:.4f}\nMAE={mae:.4f}\n$R^2$={r2:.4f}',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          ec='gray', alpha=0.9))

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    # 隐藏多余子图
    for idx in range(n_batteries, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close(fig)
    print(f"[Saved] {save_path}")

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  {method_name} — LOO Cross-Battery Results")
    print(f"{'='*60}")
    print(f"  {'Test Battery':<14} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"  {'-'*44}")
    rmse_list, mae_list, r2_list = [], [], []
    for test_bat in test_bat_ids:
        res = all_results[test_bat]
        rmse = np.sqrt(mean_squared_error(res['y_true_test'], res['y_pred_test']))
        mae  = mean_absolute_error(res['y_true_test'], res['y_pred_test'])
        r2   = r2_score(res['y_true_test'], res['y_pred_test'])
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        print(f"  {test_bat:<14} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
    print(f"  {'-'*44}")
    print(f"  {'Average':<14} {np.mean(rmse_list):<10.4f} {np.mean(mae_list):<10.4f} {np.mean(r2_list):<10.4f}")
    print(f"{'='*60}\n")


def run_loo():
    """
    留一法跨电池验证主流程（三通道相位/对数模特征图策略）。
    每轮以 1 个电池为测试集，其余 4 个为训练集。
    随机种子固定为 42，确保可复现性。
    """
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}

    # 预加载所有电池数据（避免重复 I/O）
    print("正在加载所有电池数据...")
    battery_data = {}
    for bat_id in BATTERY_IDS:
        X_seq, y_seq, soh_raw = load_battery_sequences(bat_id)
        battery_data[bat_id] = {
            'X_seq': X_seq,
            'y_seq': y_seq,
            'soh_raw': soh_raw,
        }
        print(f"  {bat_id}: {len(soh_raw)} cycles, {len(y_seq)} sequences")

    # 为第一个电池生成特征图
    first_data = load_single_battery(FOLDER, BATTERY_IDS[0], ['Stage5'])
    plot_feature_maps(first_data['re'], first_data['im'], first_data['soh'],
                      BATTERY_IDS[0], SAVE_DIR)

    # 留一法循环
    for test_bat in BATTERY_IDS:
        # 每轮重置种子，确保各轮训练可复现
        set_seed(SEED)

        train_bats = [b for b in BATTERY_IDS if b != test_bat]

        print(f"\n{'='*60}")
        print(f"  测试电池: {test_bat}")
        print(f"  训练电池: {', '.join(train_bats)}")
        print(f"{'='*60}")

        # 拼接训练集（各电池的序列独立构造后拼接）
        X_train_list, y_train_list = [], []
        for bat_id in train_bats:
            X_train_list.append(battery_data[bat_id]['X_seq'])
            y_train_list.append(battery_data[bat_id]['y_seq'])

        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)

        # 测试集
        X_test_all = battery_data[test_bat]['X_seq']
        y_test_all = battery_data[test_bat]['y_seq']

        print(f"  训练样本数: {len(y_train_all)}, 测试样本数: {len(y_test_all)}")

        # 标准化（统计量仅来自训练集）
        X_train_np, X_test_np = normalize_images(X_train_all, X_test_all)

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_all, dtype=torch.float32).unsqueeze(1)
        X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
        y_test  = torch.tensor(y_test_all, dtype=torch.float32).unsqueeze(1)

        # 测试集的周期编号（1-indexed）
        n_test_cycles = len(battery_data[test_bat]['soh_raw'])
        test_cycles = np.arange(SEQ_LEN, n_test_cycles + 1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

        model = SEResNet_Transformer(
            seq_len=SEQ_LEN, d_model=256, nhead=4, num_layers=2,
            in_channels=3, dropout=0
        ).to(device)

        if test_bat == BATTERY_IDS[0]:
            print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        print(f"开始训练 (LOO: 测试={test_bat})...")
        train_losses, val_losses = train_model(
            model, train_loader, X_test, y_test,
            device, epochs=EPOCHS, lr=LR, patience=30,
            weight_decay=0
        )

        model.eval()
        with torch.no_grad():
            preds_test = model(X_test.to(device)).cpu().numpy().flatten()

        all_results[test_bat] = {
            'y_true_test':    y_test_all,
            'y_pred_test':    preds_test,
            'test_cycles':    test_cycles,
            'train_batteries': train_bats,
            'train_losses':   train_losses,
            'val_losses':     val_losses,
        }

        rmse = np.sqrt(mean_squared_error(y_test_all, preds_test))
        mae  = mean_absolute_error(y_test_all, preds_test)
        r2   = r2_score(y_test_all, preds_test)
        print(f"  结果: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # 绘制汇总结果
    plot_loo_results(
        all_results,
        method_name='3Ch Phase-Diff/Sum + LogMod + SE-ResNet-Transformer (LOO)',
        save_path=os.path.join(SAVE_DIR, 'result_LOO_3Ch_PhaseLogMod.png')
    )


if __name__ == '__main__':
    run_loo()
