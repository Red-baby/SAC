# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels=5, T=16, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.ln = nn.LayerNorm([128, T])
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.out_ln = nn.LayerNorm(hidden)

    def forward(self, x):           # [B,C,T]
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h)) + h
        h = F.gelu(self.conv3(h))
        h = self.ln(h)              # [B,128,T]
        h = h.transpose(1, 2)       # [B,T,128]
        h, _ = self.gru(h)          # [B,T,H]
        h = h[:, -1, :]
        return self.out_ln(h)

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=5, seq_T=16, scalar_dim=10, hidden=512):
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
    def __init__(self, state_scalar_dim, in_channels=5, seq_T=16, hidden=512):
        super().__init__()
        self.enc = FeatureEncoder(in_channels, seq_T, state_scalar_dim, hidden)
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -5.0, 2.0

    def forward(self, seq_bc_t, scalars):
        h = self.enc(seq_bc_t, scalars)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, state_scalar_dim, in_channels=5, seq_T=16, hidden=512):
        super().__init__()
        self.enc1 = FeatureEncoder(in_channels, seq_T, state_scalar_dim+1, hidden)
        self.enc2 = FeatureEncoder(in_channels, seq_T, state_scalar_dim+1, hidden)
        self.q1 = nn.Linear(hidden, 1)
        self.q2 = nn.Linear(hidden, 1)

    def forward(self, seq_bc_t, scalars, a):
        s1 = torch.cat([scalars, a], dim=-1)
        s2 = torch.cat([scalars, a], dim=-1)
        h1 = self.enc1(seq_bc_t, s1)
        h2 = self.enc2(seq_bc_t, s2)
        return self.q1(h1), self.q2(h2)
