import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeatureFusionAttention(nn.Module):
    def __init__(self, high_dim, low_dim, num_heads=4, head_dim=32, dropout=0.1):
        super(FeatureFusionAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5  # 缩放因子，防止梯度消失
        self.q_proj = nn.Linear(high_dim, num_heads * head_dim)  # 查询向量
        self.k_proj = nn.Linear(low_dim, num_heads * head_dim)  # 键向量
        self.v_proj = nn.Linear(low_dim, num_heads * head_dim)  # 值向量
        self.out_proj = nn.Linear(num_heads * head_dim, low_dim)  # 输出映射
        self.dropout = nn.Dropout(dropout)

    def forward(self, high_feat, low_feat):
        # 输入：
        # high_feat: 高层特征 (B, C_high, H, W)
        # low_feat: 低层特征 (B, C_low, H_low, W_low)

        B, C_high, H, W = high_feat.shape
        B, C_low, H_low, W_low = low_feat.shape

        # 下采样低层特征到高层分辨率
        low_feat = F.adaptive_avg_pool2d(low_feat, (H, W))  # 调整到 (B, C_low, H, W)

        # 展平特征
        high_feat_flat = high_feat.view(B, C_high, -1).transpose(1, 2)  # (B, H*W, C_high)
        low_feat_flat = low_feat.view(B, C_low, -1).transpose(1, 2)  # (B, H*W, C_low)

        # 计算 Q, K, V
        Q = self.q_proj(high_feat_flat).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        K = self.k_proj(low_feat_flat).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # (B, num_heads, head_dim, H*W)
        V = self.v_proj(low_feat_flat).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)

        # 计算注意力
        attn_weights = torch.matmul(Q, K) * self.scale  # (B, num_heads, H*W, H*W)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权特征
        fused_feat = torch.matmul(attn_weights, V)  # (B, num_heads, H*W, head_dim)
        fused_feat = fused_feat.permute(0, 2, 1, 3).contiguous().view(B, -1, self.num_heads * self.head_dim)  # (B, H*W, num_heads*head_dim)
        fused_feat = self.out_proj(fused_feat)  # (B, H*W, C_low)
        fused_feat = fused_feat.transpose(1, 2).view(B, C_low, H, W)  # 恢复形状

        return fused_feat
