# se_resnet_model.py
# ============================================================
# 改进方案：SE-ResNet 轻量级模型 用于 EIS 图像回归
# ============================================================
# 改进动机：
#   原始 LiteCNN 是朴素的 3 层堆叠卷积，存在两个学术层面的不足：
#   1. 无残差连接 —— 梯度在深层可能出现退化，且丢失浅层纹理特征；
#   2. 所有通道被同等对待 —— 但 EIS 三通道（Re/Im/Phase）信息量
#      差异显著，应有选择性地增强关键通道。
#
# 改进策略：
#   (a) 引入 Residual Block（残差块），每块含 2 层 3×3 Conv + BN + ReLU，
#       配合恒等跳跃连接，保证梯度通畅。
#   (b) 嵌入 Squeeze-and-Excitation (SE) 模块 [Hu et al., CVPR 2018]，
#       通过全局平均池化 → FC → Sigmoid 学习通道注意力权重，
#       自适应放大重要通道（如低频 Re 变化）、抑制噪声通道。
#   (c) 保持轻量化：仅 3 个 SE-ResBlock（32→64→128），
#       总参数约 ~90K，远小于 VGG-16 的 ~15M，适合小样本训练。
#
# 架构示意：
#   Input(C_in, 60, 60)
#   → SE-ResBlock(32)  → MaxPool → 30×30
#   → SE-ResBlock(64)  → MaxPool → 15×15
#   → SE-ResBlock(128) → AdaptiveAvgPool(4,4) → 2048-dim
#
# SE-ResBlock 结构：
#   x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+) SE(·) → ReLU → out
#       ↘ ─────── identity / 1×1 Conv ──────────↗
#
# 参考文献：
#   [1] He et al. "Deep Residual Learning for Image Recognition", CVPR 2016
#   [2] Hu et al. "Squeeze-and-Excitation Networks", CVPR 2018
# ============================================================

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块

    核心思想：
      1. Squeeze: 全局平均池化 → (B, C, 1, 1)，获取通道级全局描述
      2. Excitation: FC(C→C//r) → ReLU → FC(C//r→C) → Sigmoid
         学习通道间的非线性依赖关系
      3. 用得到的注意力权重逐通道缩放原特征图

    对 EIS 图像的意义：
      不同通道编码了 Re(Z)、Im(Z)、Phase/Freq 等不同物理量，
      SE 模块能自动学习哪些通道在当前老化阶段更具判别力。
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 8)  # 防止过小
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        # Squeeze: (B, C, H, W) → (B, C)
        w = self.squeeze(x).view(B, C)
        # Excitation: (B, C) → (B, C)
        w = self.excitation(w).view(B, C, 1, 1)
        # Scale: 逐通道缩放
        return x * w


class SEResBlock(nn.Module):
    """
    SE-残差块 = 2×(Conv3×3 + BN) + 残差连接 + SE注意力

    当 in_channels != out_channels 时，使用 1×1 Conv 对齐维度。
    """
    def __init__(self, in_channels, out_channels, se_reduction=4):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock(out_channels, reduction=se_reduction)

        # 残差跳跃连接（维度不匹配时用 1×1 Conv 对齐）
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)           # 通道注意力加权
        out = self.relu(out + identity)  # 残差连接
        return out


class SEResNet(nn.Module):
    """
    轻量级 SE-ResNet 特征提取器

    架构：3 个 SE-ResBlock (32→64→128)
    输入：(B, C_in, 60, 60)  — C_in 可为 3/6/9 通道
    输出：(B, 128, 4, 4) = 2048-dim 展平后
    """
    def __init__(self, in_channels=3):
        super(SEResNet, self).__init__()
        self.layer1 = nn.Sequential(
            SEResBlock(in_channels, 32),
            nn.MaxPool2d(2, 2),                  # 60→30
        )
        self.layer2 = nn.Sequential(
            SEResBlock(32, 64),
            nn.MaxPool2d(2, 2),                  # 30→15
        )
        self.layer3 = nn.Sequential(
            SEResBlock(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),         # →4×4
        )
        self.out_dim = 128 * 4 * 4  # 2048

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class SEResNet_Regressor(nn.Module):
    """
    SE-ResNet 直接回归器（用于非时序策略，如 CNN 文件夹的等价物）

    架构：SEResNet(2048) → FC(256) → FC(64) → FC(1)
    """
    def __init__(self, in_channels=3):
        super(SEResNet_Regressor, self).__init__()
        self.backbone = SEResNet(in_channels=in_channels)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x)


class SEResNet_Transformer(nn.Module):
    """
    SE-ResNet + Transformer 时序预测模型

    架构：
      SEResNet(2048-dim) → Linear(256) → PositionalEncoding
      → TransformerEncoder(d_model=256, nhead=4, num_layers=2)
      → 取最后一个时间步 → FC(128→64→1)

    改进亮点：
      1. SE-ResNet 替代 LiteCNN：残差连接+通道注意力提升空间特征质量
      2. Transformer 替代 GRU/LSTM：
         - Self-Attention 可直接建模任意两帧间的依赖关系，
           无需像 RNN 逐步传播，避免长程信息衰减
         - 计算可并行化，训练更高效
         - 对于短序列（seq_len=3~5），Transformer 计算开销可接受
      3. Positional Encoding：为序列注入位置信息

    输入：(Batch, Seq_Len, C_in, 60, 60)
    输出：(Batch, 1)
    """
    def __init__(self, seq_len=3, d_model=256, nhead=4, num_layers=2,
                 in_channels=3, dropout=0.0):
        super(SEResNet_Transformer, self).__init__()

        # 1. SE-ResNet 空间特征提取
        self.cnn = SEResNet(in_channels=in_channels)
        cnn_out = self.cnn.out_dim  # 2048

        # 2. 特征投影
        self.feat_proj = nn.Sequential(
            nn.Linear(cnn_out, d_model),
            nn.ReLU(inplace=True),
        )

        # 3. 可学习位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,     # 512
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 5. 回归头（支持可选 Dropout 正则化）
        fc_layers = [nn.Linear(d_model, 128), nn.ReLU(inplace=True)]
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers += [nn.Linear(128, 64), nn.ReLU(inplace=True)]
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(64, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        B, S, C, H, W = x.size()

        # CNN 逐帧提取
        x = x.view(B * S, C, H, W)
        x = self.cnn(x)               # (B*S, 128, 4, 4)
        x = x.view(B * S, -1)         # (B*S, 2048)
        x = self.feat_proj(x)         # (B*S, 256)
        x = x.view(B, S, -1)          # (B, S, 256)

        # 加入位置编码
        x = x + self.pos_embedding[:, :S, :]

        # Transformer 时序建模
        x = self.transformer(x)        # (B, S, 256)

        # 取最后时间步
        x = x[:, -1, :]               # (B, 256)

        return self.fc(x)


def get_se_resnet_regressor(in_channels=3):
    """获取 SE-ResNet 直接回归器"""
    return SEResNet_Regressor(in_channels=in_channels)


# ============================================================
# 跨注意力多分支融合模型 (Cross-Attention Multi-Branch Fusion)
# ============================================================
# 改进动机：
#   简单的 9 通道拼接 (channel concatenation) 存在两个不足：
#   1. 不同编码策略（GAF/Real/RP）的特征空间和数值范围差异大，
#      直接拼接后由单一 CNN 处理难以平衡各策略贡献；
#   2. 通道维度拼接丧失了策略间语义边界信息，
#      SE 注意力只能在通道级加权，无法建模跨策略的交互关系。
#
# 改进策略 — 多分支 + 跨注意力融合：
#   (a) 三个独立 SE-ResNet 分支分别提取 GAF/Real/RP 特征，
#       每个分支接收 3 通道输入，各自学习特定编码的最优滤波器；
#   (b) 跨注意力融合 (Cross-Attention)：
#       将三分支特征拼接为序列 [f_GAF; f_Real; f_RP] ∈ R^{3×d}，
#       通过多头自注意力让各策略特征互相查询：
#         - GAF 分支可以从 Real 和 RP 中获取互补信息
#         - 注意力权重揭示了不同编码策略的相对重要性（可解释性）
#   (c) 融合后的特征经 Transformer Encoder 建模时序依赖。
#
# 参考文献：
#   [1] Vaswani et al. "Attention Is All You Need", NeurIPS 2017
#   [2] Jaegle et al. "Perceiver: General Perception with Iterative
#       Attention", ICML 2021 (multi-modal cross-attention)
#   [3] Nagrani et al. "Attention Bottlenecks for Multimodal Fusion",
#       NeurIPS 2021
# ============================================================


class CrossAttentionFusionBlock(nn.Module):
    """
    跨注意力融合模块

    将 K 个分支特征视为一个长度为 K 的序列，
    使用标准多头自注意力建模分支间交互。

    输入: (B, K, d_model)  —  K 个分支各产出 d_model 维特征
    输出: (B, d_model)     —  融合后的单一特征向量

    融合方式：注意力加权求和后取均值池化，
    等效于让每个分支自适应地聚合其余分支的信息。
    """
    def __init__(self, d_model=256, nhead=4, num_layers=1, dropout=0.1):
        super(CrossAttentionFusionBlock, self).__init__()
        self.norm_in = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.cross_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, branch_features):
        """
        branch_features: (B, K, d_model)
        """
        x = self.norm_in(branch_features)
        x = self.cross_attn(x)          # (B, K, d_model)
        x = self.norm_out(x)
        # 均值池化融合
        return x.mean(dim=1)            # (B, d_model)


class MultiBranch_SEResNet_Transformer(nn.Module):
    """
    多分支跨注意力融合 + Transformer 时序预测模型

    架构图：
      ┌─ Branch_GAF:   SEResNet(3ch) → 2048 → Linear → d_model ─┐
      │  Branch_Real:  SEResNet(3ch) → 2048 → Linear → d_model  ├─→ CrossAttn
      └─ Branch_RP:    SEResNet(3ch) → 2048 → Linear → d_model ─┘    → d_model
                                                                         ↓
                                                              PositionalEncoding
                                                                         ↓
                                                              TransformerEncoder
                                                                   (时序建模)
                                                                         ↓
                                                                  FC → SOH 预测

    输入格式 (每帧):
      x_gaf:   (B, S, 3, 60, 60)
      x_real:  (B, S, 3, 60, 60)
      x_rp:    (B, S, 3, 60, 60)

    输出: (B, 1)
    """
    def __init__(self, seq_len=3, d_model=256, nhead=4, num_layers=2,
                 cross_attn_layers=1):
        super(MultiBranch_SEResNet_Transformer, self).__init__()

        # 1. 三个独立 SE-ResNet 分支
        self.branch_gaf  = SEResNet(in_channels=3)
        self.branch_real = SEResNet(in_channels=3)
        self.branch_rp   = SEResNet(in_channels=3)
        cnn_out = self.branch_gaf.out_dim  # 2048

        # 2. 各分支特征投影到 d_model
        self.proj_gaf  = nn.Sequential(nn.Linear(cnn_out, d_model), nn.ReLU(True))
        self.proj_real = nn.Sequential(nn.Linear(cnn_out, d_model), nn.ReLU(True))
        self.proj_rp   = nn.Sequential(nn.Linear(cnn_out, d_model), nn.ReLU(True))

        # 3. 跨注意力融合
        self.cross_fusion = CrossAttentionFusionBlock(
            d_model=d_model, nhead=nhead,
            num_layers=cross_attn_layers, dropout=0.1
        )

        # 4. 可学习位置编码（时序维度）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # 5. Transformer Encoder（时序建模）
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1, activation='gelu',
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers
        )

        # 6. 回归头
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def _extract_branch(self, branch_cnn, proj, x):
        """
        对单个分支逐帧提取特征并投影。
        x: (B*S, 3, 60, 60) → (B*S, d_model)
        """
        feat = branch_cnn(x)            # (B*S, 128, 4, 4)
        feat = feat.view(feat.size(0), -1)  # (B*S, 2048)
        return proj(feat)               # (B*S, d_model)

    def forward(self, x_gaf, x_real, x_rp):
        """
        x_gaf, x_real, x_rp: 各 (B, S, 3, 60, 60)
        """
        B, S, C, H, W = x_gaf.size()

        # 展开时序维度
        gaf_flat  = x_gaf.reshape(B * S, C, H, W)
        real_flat = x_real.reshape(B * S, C, H, W)
        rp_flat   = x_rp.reshape(B * S, C, H, W)

        # 各分支独立提取 + 投影
        f_gaf  = self._extract_branch(self.branch_gaf,  self.proj_gaf,  gaf_flat)
        f_real = self._extract_branch(self.branch_real, self.proj_real, real_flat)
        f_rp   = self._extract_branch(self.branch_rp,   self.proj_rp,   rp_flat)
        # 各 (B*S, d_model)

        # 拼接为 (B*S, 3, d_model) 供跨注意力融合
        branch_stack = torch.stack([f_gaf, f_real, f_rp], dim=1)
        fused = self.cross_fusion(branch_stack)  # (B*S, d_model)

        # 恢复时序维度
        fused = fused.view(B, S, -1)    # (B, S, d_model)

        # 位置编码 + Transformer 时序建模
        fused = fused + self.pos_embedding[:, :S, :]
        temporal = self.temporal_transformer(fused)  # (B, S, d_model)

        # 取最后时间步
        out = temporal[:, -1, :]        # (B, d_model)
        return self.fc(out)
