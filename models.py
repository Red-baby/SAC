# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码，用于 Self-Attention"""
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class TemporalEncoderAttention(nn.Module):
    """
    使用 Self-Attention 替代 GRU 的时序编码器
    
    优势：
    1. 并行计算，速度更快
    2. 直接建模长距离依赖
    3. I 帧等关键帧可被显式关注
    4. Attention 权重可视化，便于调试
    """
    def __init__(self, in_channels=6, T=64, hidden=128, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden = hidden
        
        # 1D 卷积提取局部特征
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, hidden, 3, padding=1)
        self.conv_ln = nn.LayerNorm(hidden)
        
        # 位置编码
        self.pos_enc = PositionalEncoding(hidden, max_len=max(T, 256))
        
        # Self-Attention 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出汇聚：使用 [CLS] token 或平均池化
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.out_ln = nn.LayerNorm(hidden)
    
    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        
        # 卷积提取局部特征
        h = F.gelu(self.conv1(x))          # [B, 64, T]
        h = F.gelu(self.conv2(h))          # [B, hidden, T]
        h = h.transpose(1, 2)              # [B, T, hidden]
        h = self.conv_ln(h)
        
        # 添加位置编码
        h = self.pos_enc(h)
        
        # 添加 [CLS] token 用于汇聚
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden]
        h = torch.cat([cls_tokens, h], dim=1)          # [B, T+1, hidden]
        
        # Self-Attention
        h = self.transformer(h)            # [B, T+1, hidden]
        
        # 使用 [CLS] token 作为输出（也可以用平均池化）
        out = h[:, 0, :]                   # [B, hidden]
        return self.out_ln(out)


class TemporalEncoder(nn.Module):
    """原始 GRU 版本（保留用于对比或小序列）"""
    def __init__(self, in_channels=6, T=16, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.ln = nn.LayerNorm([128, T])
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.out_ln = nn.LayerNorm(hidden)

    def forward(self, x):           # [B,C,T]
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = self.ln(h)              # [B,128,T]
        h = h.transpose(1, 2)       # [B,T,128]
        h, _ = self.gru(h)          # [B,T,H]
        h = h[:, -1, :]
        return self.out_ln(h)


class FeatureEncoder(nn.Module):
    """特征编码器：结合序列特征和标量特征"""
    def __init__(self, in_channels=6, seq_T=64, scalar_dim=11, hidden=512, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # 选择时序编码器
        if use_attention:
            self.temporal = TemporalEncoderAttention(in_channels, seq_T, 128, num_heads=4, num_layers=2)
        else:
            self.temporal = TemporalEncoder(in_channels, seq_T, 128)
        
        self.fc_in = nn.Linear(128 + scalar_dim, hidden)
        self.b1 = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden))
        self.b2 = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(),
                                nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden))
        self.out_ln = nn.LayerNorm(hidden)

    def forward(self, seq_bc_t, scalars):
        z = torch.cat([self.temporal(seq_bc_t), scalars], dim=-1)
        h = F.gelu(self.fc_in(z))
        h = F.gelu(h + self.b1(h))
        h = F.gelu(h + self.b2(h))
        return self.out_ln(h)


class Actor(nn.Module):
    def __init__(self, state_scalar_dim, in_channels=6, seq_T=64, hidden=512, num_discrete_actions=41):
        super().__init__()
        self.enc = FeatureEncoder(in_channels, seq_T, state_scalar_dim, hidden, use_attention=True)
        self.action_dim = 5
        self.num_discrete_actions = int(num_discrete_actions)
        self.logits = nn.Linear(hidden, self.action_dim * self.num_discrete_actions)

    def forward(self, seq_bc_t, scalars):
        h = self.enc(seq_bc_t, scalars)
        logits = self.logits(h).view(-1, self.action_dim, self.num_discrete_actions)
        return logits


class Critic(nn.Module):
    def __init__(self, state_scalar_dim, in_channels=6, seq_T=64, hidden=512, num_discrete_actions=41, action_embed_dim=64):
        super().__init__()
        self.action_dim = 5
        self.num_discrete_actions = int(num_discrete_actions)
        self.action_embed = nn.Embedding(self.action_dim * self.num_discrete_actions, action_embed_dim)
        self.enc1 = FeatureEncoder(in_channels, seq_T, state_scalar_dim + action_embed_dim, hidden, use_attention=True)
        self.enc2 = FeatureEncoder(in_channels, seq_T, state_scalar_dim + action_embed_dim, hidden, use_attention=True)
        self.q1 = nn.Linear(hidden, 1)
        self.q2 = nn.Linear(hidden, 1)

    def forward(self, seq_bc_t, scalars, a_indices):
        offsets = torch.arange(self.action_dim, device=a_indices.device).unsqueeze(0) * self.num_discrete_actions
        global_idx = (offsets + a_indices.long()).clamp(min=0, max=self.action_dim * self.num_discrete_actions - 1)
        action_embeds = self.action_embed(global_idx)
        action_embed = action_embeds.mean(dim=1)
        s = torch.cat([scalars, action_embed], dim=-1)
        h1 = self.enc1(seq_bc_t, s)
        h2 = self.enc2(seq_bc_t, s)
        return self.q1(h1), self.q2(h2)


class DuelingQNet(nn.Module):
    """Dueling DQN 网络（用于 D3QN）"""
    def __init__(self, state_scalar_dim, in_channels=6, seq_T=64, hidden=512, num_discrete_actions=41, action_dim=1):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_discrete_actions = int(num_discrete_actions)
        self.enc = FeatureEncoder(in_channels, seq_T, state_scalar_dim, hidden, use_attention=True)
        self.value = nn.Linear(hidden, self.action_dim)
        self.adv = nn.Linear(hidden, self.action_dim * self.num_discrete_actions)

    def forward(self, seq_bc_t, scalars):
        h = self.enc(seq_bc_t, scalars)
        v = self.value(h).view(-1, self.action_dim, 1)
        a = self.adv(h).view(-1, self.action_dim, self.num_discrete_actions)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q
