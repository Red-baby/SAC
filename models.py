# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels=6, T=16, hidden=128):  # 更新为 6 通道（包含 qp）
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
    def __init__(self, in_channels=6, seq_T=16, scalar_dim=9, hidden=512):  # 更新通道数和 scalar 维度
        super().__init__()
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
    def __init__(self, state_scalar_dim, in_channels=6, seq_T=16, hidden=512, num_discrete_actions=41):
        super().__init__()
        self.enc = FeatureEncoder(in_channels, seq_T, state_scalar_dim, hidden)
        self.action_dim = 5
        self.num_discrete_actions = int(num_discrete_actions)
        self.logits = nn.Linear(hidden, self.action_dim * self.num_discrete_actions)

    def forward(self, seq_bc_t, scalars):
        h = self.enc(seq_bc_t, scalars)
        logits = self.logits(h).view(-1, self.action_dim, self.num_discrete_actions)
        return logits

class Critic(nn.Module):
    def __init__(self, state_scalar_dim, in_channels=6, seq_T=16, hidden=512, num_discrete_actions=41, action_embed_dim=64):
        super().__init__()
        self.action_dim = 5
        self.num_discrete_actions = int(num_discrete_actions)
        self.action_embed = nn.Embedding(self.action_dim * self.num_discrete_actions, action_embed_dim)
        self.enc1 = FeatureEncoder(in_channels, seq_T, state_scalar_dim + action_embed_dim, hidden)
        self.enc2 = FeatureEncoder(in_channels, seq_T, state_scalar_dim + action_embed_dim, hidden)
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

