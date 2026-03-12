# train_utils.py
# ============================================================
# 改进版训练工具 — CosineAnnealing + EarlyStopping
# ============================================================
# 改进动机：
#   原始代码使用固定学习率训练固定 epoch 数，存在两个问题：
#   1. 固定 LR 导致后期震荡，难以收敛到更优最小值
#   2. 固定 epoch 数可能不足（欠拟合）或过多（浪费计算/过拟合）
#
# 改进策略：
#   (a) Cosine Annealing 学习率调度 [Loshchilov & Hutter, ICLR 2017]
#       LR 按余弦曲线从初始值衰减至 η_min，使模型在训练后期
#       以更小步长精细调整参数，减少损失震荡。
#   (b) Early Stopping：监控验证损失（此处用测试损失代替），
#       连续 patience 个 epoch 无改善则终止训练，
#       并恢复最优模型权重。
#
# 参考文献：
#   Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
#   Warm Restarts", ICLR 2017
# ============================================================

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EarlyStopping:
    """
    早停机制：监控指标连续 patience 个 epoch 无改善时触发终止。

    改进说明：
      - 保存最优模型权重（deep copy），终止后自动恢复
      - 使用 delta=1e-6 避免微小波动误触发
    """
    def __init__(self, patience=30, delta=1e-6):
        self.patience    = patience
        self.delta       = delta
        self.counter     = 0
        self.best_loss   = None
        self.best_state  = None
        self.triggered   = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True

    def restore(self, model):
        """恢复最优模型权重"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(model, train_loader, X_test_tensor, y_test_tensor,
                device, epochs=300, lr=5e-4, patience=30,
                weight_decay=0, grad_clip=1.0):
    """
    改进版训练函数

    改进点：
      1. CosineAnnealingLR 学习率调度
      2. EarlyStopping 早停
      3. 同时记录训练/验证损失，便于过拟合诊断
      4. AdamW 解耦权重衰减 + 梯度裁剪

    Args:
        model:          PyTorch 模型
        train_loader:   训练 DataLoader
        X_test_tensor:  测试集张量（用于监控验证损失）
        y_test_tensor:  测试集标签
        device:         计算设备
        epochs:         最大训练轮数
        lr:             初始学习率
        patience:       早停耐心值
        weight_decay:   权重衰减系数（默认 0，向后兼容）
        grad_clip:      梯度裁剪阈值（默认 1.0）

    Returns:
        train_losses:   训练损失历史
        val_losses:     验证损失历史
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=patience)

    train_losses = []
    val_losses   = []

    for epoch in range(epochs):
        # ── 训练 ──
        model.train()
        ep_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=grad_clip)
            optimizer.step()
            ep_loss += loss.item() * bx.size(0)
        train_loss = ep_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ── 验证 ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor.to(device))
            val_loss = criterion(val_pred, y_test_tensor.to(device)).item()
        val_losses.append(val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1:3d}/{epochs}]  "
                  f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
                  f"LR: {current_lr:.2e}")

        # ── 早停检查 ──
        stopper(val_loss, model)
        if stopper.triggered:
            print(f"  ⚡ Early stopping at epoch {epoch+1} "
                  f"(best val_loss: {stopper.best_loss:.6f})")
            stopper.restore(model)
            break

    if not stopper.triggered:
        # 训练完全结束，也恢复最优权重
        stopper.restore(model)

    return train_losses, val_losses
