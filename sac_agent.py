# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DuelingQNet

class D3QNAgent:
    def __init__(self, cfg, state_scalar_dim: int, seq_T: int, seq_C: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = 1
        self.num_discrete_actions = int(getattr(cfg, "num_discrete_actions", 0) or 0)
        values = getattr(cfg, "discrete_action_values", None)
        action_min = float(getattr(cfg, "action_min", 30))
        action_max = float(getattr(cfg, "action_max", 210))
        action_step = max(1, int(getattr(cfg, "action_step", 1)))
        if not values:
            if self.num_discrete_actions <= 0:
                values = list(range(int(action_min), int(action_max) + 1, action_step))
                if not values:
                    values = [action_min]
            elif self.num_discrete_actions <= 1:
                values = [action_min]
            else:
                step = (action_max - action_min) / float(self.num_discrete_actions - 1)
                values = [action_min + i * step for i in range(self.num_discrete_actions)]
        self.discrete_action_values = torch.tensor(values, device=self.device, dtype=torch.float32)
        self.num_discrete_actions = int(self.discrete_action_values.numel())

        self.q_net = DuelingQNet(
            state_scalar_dim,
            seq_C,
            seq_T,
            cfg.hidden_dim,
            num_discrete_actions=self.num_discrete_actions,
            action_dim=self.action_dim,
        ).to(self.device)
        self.q_tgt = DuelingQNet(
            state_scalar_dim,
            seq_C,
            seq_T,
            cfg.hidden_dim,
            num_discrete_actions=self.num_discrete_actions,
            action_dim=self.action_dim,
        ).to(self.device)
        self.q_tgt.load_state_dict(self.q_net.state_dict())

        lr = float(getattr(cfg, "lr_critic", 3e-4))
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.target_update_interval = int(getattr(cfg, "dqn_target_update_interval", 200))

        self.eps_start = float(getattr(cfg, "dqn_eps_start", 1.0))
        self.eps_end = float(getattr(cfg, "dqn_eps_end", 0.05))
        self.eps_decay = float(getattr(cfg, "dqn_eps_decay", 20000))
        self._train_steps = 0
        self.last_epsilon = self.eps_start

    def _epsilon(self) -> float:
        if self.eps_decay <= 0:
            return self.eps_end
        decay = math.exp(-float(self._train_steps) / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * decay

    def act(self, seq1c_t, scalars, deterministic: bool = False):
        with torch.no_grad():
            q = self.q_net(seq1c_t, scalars)
        greedy = torch.argmax(q, dim=-1)
        if deterministic:
            self.last_epsilon = 0.0
            return greedy, None

        eps = self._epsilon()
        self.last_epsilon = eps
        if eps <= 0.0:
            return greedy, None

        rand_mask = (torch.rand((q.shape[0], 1), device=q.device) < eps).expand(-1, self.action_dim)
        rand_actions = torch.randint(0, self.num_discrete_actions, greedy.shape, device=q.device)
        a_idx = torch.where(rand_mask, rand_actions, greedy)
        return a_idx, None

    def train_step(self, batch):
        seq, sca, a_idx, r, seq2, sca2, d = batch
        a_idx = a_idx.long()

        q = self.q_net(seq, sca)
        q_a = q.gather(-1, a_idx.unsqueeze(-1)).squeeze(-1)
        q_pred = q_a.mean(dim=1, keepdim=True)

        with torch.no_grad():
            q_next_online = self.q_net(seq2, sca2)
            a2_idx = torch.argmax(q_next_online, dim=-1)
            q_next_tgt = self.q_tgt(seq2, sca2)
            q_next = q_next_tgt.gather(-1, a2_idx.unsqueeze(-1)).squeeze(-1)
            q_next_mean = q_next.mean(dim=1, keepdim=True)
            y = r + (1.0 - d) * self.gamma * q_next_mean

        loss_q = F.mse_loss(q_pred, y)
        self.opt.zero_grad(set_to_none=True)
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.opt.step()

        self._train_steps += 1
        if self.target_update_interval > 0 and (self._train_steps % self.target_update_interval) == 0:
            self.q_tgt.load_state_dict(self.q_net.state_dict())

        return float(loss_q.item()), float(self.last_epsilon)

    def save_checkpoint(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "q_net_state_dict": self.q_net.state_dict(),
            "q_tgt_state_dict": self.q_tgt.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "gamma": self.gamma,
            "target_update_interval": self.target_update_interval,
            "train_steps": self._train_steps,
        }
        torch.save(checkpoint, path)
        print(f"[Checkpoint] saved -> {path}")

    def load_checkpoint(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.q_tgt.load_state_dict(checkpoint["q_tgt_state_dict"])
        self.opt.load_state_dict(checkpoint["opt_state_dict"])
        if "gamma" in checkpoint:
            self.gamma = float(checkpoint["gamma"])
        if "target_update_interval" in checkpoint:
            self.target_update_interval = int(checkpoint["target_update_interval"])
        if "train_steps" in checkpoint:
            self._train_steps = int(checkpoint["train_steps"])

        print(f"[Checkpoint] loaded <- {path}")
        print(f"[Checkpoint] restore: gamma={self.gamma}, target_update_interval={self.target_update_interval}")
