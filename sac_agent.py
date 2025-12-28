# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from models import Actor, Critic


class SACAgent:
    def __init__(self, cfg, state_scalar_dim: int, seq_T: int, seq_C: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.action_dim = 5
        self.num_discrete_actions = int(getattr(cfg, "num_discrete_actions", 0) or 0)
        self.num_action_samples = max(1, int(getattr(cfg, "num_action_samples", 8)))
        values = getattr(cfg, "discrete_action_values", None)
        if not values:
            if self.num_discrete_actions <= 0:
                self.num_discrete_actions = max(1, int(getattr(cfg, "delta_qp_max", 0)) * 2 + 1)
            if self.num_discrete_actions <= 1:
                values = [0.0]
            else:
                step = (2 * float(cfg.delta_qp_max)) / float(self.num_discrete_actions - 1)
                values = [-float(cfg.delta_qp_max) + i * step for i in range(self.num_discrete_actions)]
        self.discrete_action_values = torch.tensor(values, device=self.device, dtype=torch.float32)
        self.num_discrete_actions = int(self.discrete_action_values.numel())

        self.actor = Actor(
            state_scalar_dim,
            seq_C,
            seq_T,
            cfg.hidden_dim,
            num_discrete_actions=self.num_discrete_actions,
        ).to(self.device)
        self.critic = Critic(
            state_scalar_dim,
            seq_C,
            seq_T,
            cfg.hidden_dim,
            num_discrete_actions=self.num_discrete_actions,
        ).to(self.device)
        self.critic_tgt = Critic(
            state_scalar_dim,
            seq_C,
            seq_T,
            cfg.hidden_dim,
            num_discrete_actions=self.num_discrete_actions,
        ).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        if cfg.target_entropy == 0.0:
            max_entropy = math.log(self.num_discrete_actions) * self.action_dim if self.num_discrete_actions > 0 else 0.0
            self.target_entropy = 0.5 * max_entropy
        else:
            self.target_entropy = cfg.target_entropy

    def act(self, seq1c_t, scalars, deterministic=False):
        logits = self.actor(seq1c_t, scalars)
        if deterministic:
            action_idx = torch.argmax(logits, dim=-1)
            logp = torch.zeros((logits.shape[0], 1), device=logits.device)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx).sum(dim=-1, keepdim=True)
        return action_idx, logp

    def train_step(self, batch):
        seq, sca, a_idx, r, seq2, sca2, d = batch
        a_idx = a_idx.long()
        with torch.no_grad():
            a2_idx, logp2 = self.act(seq2, sca2, deterministic=False)
            q1_t, q2_t = self.critic_tgt(seq2, sca2, a2_idx)
            alpha = self.log_alpha.exp()
            y = r + (1.0 - d) * self.gamma * (torch.min(q1_t, q2_t) - alpha * logp2)

        q1, q2 = self.critic(seq, sca, a_idx)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_critic.step()

        logits = self.actor(seq, sca)
        dists = torch.distributions.Categorical(logits=logits)
        entropy = dists.entropy().sum(dim=-1, keepdim=True)

        K = self.num_action_samples
        a_samples = dists.sample((K,))  # [K, B, action_dim]
        logp = dists.log_prob(a_samples).sum(dim=-1, keepdim=True)  # [K, B, 1]
        a_flat = a_samples.reshape(K * seq.shape[0], self.action_dim)
        seq_rep = seq.unsqueeze(0).repeat(K, 1, 1, 1).reshape(K * seq.shape[0], seq.shape[1], seq.shape[2])
        sca_rep = sca.unsqueeze(0).repeat(K, 1, 1).reshape(K * sca.shape[0], sca.shape[1])
        q1_s, q2_s = self.critic(seq_rep, sca_rep, a_flat)
        q_min = torch.min(q1_s, q2_s).reshape(K, seq.shape[0], 1)

        alpha = self.log_alpha.exp()
        loss_actor = (alpha * logp - q_min).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_actor.step()

        loss_alpha = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return (float(loss_q.item()), float(loss_actor.item()), float(alpha.detach().cpu()))

    def save_checkpoint(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_tgt_state_dict": self.critic_tgt.state_dict(),
            "opt_actor_state_dict": self.opt_actor.state_dict(),
            "opt_critic_state_dict": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "opt_alpha_state_dict": self.opt_alpha.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
        }
        torch.save(checkpoint, path)
        print(f"[Checkpoint] saved -> {path}")

    def load_checkpoint(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_tgt.load_state_dict(checkpoint["critic_tgt_state_dict"])
        self.opt_actor.load_state_dict(checkpoint["opt_actor_state_dict"])
        self.opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])

        self.log_alpha = torch.tensor(
            checkpoint["log_alpha"],
            device=self.device,
            requires_grad=True,
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        self.opt_alpha.load_state_dict(checkpoint["opt_alpha_state_dict"])

        if "gamma" in checkpoint:
            self.gamma = checkpoint["gamma"]
        if "tau" in checkpoint:
            self.tau = checkpoint["tau"]
        if "target_entropy" in checkpoint:
            self.target_entropy = checkpoint["target_entropy"]

        print(f"[Checkpoint] loaded <- {path}")
        print(f"[Checkpoint] restore: gamma={self.gamma}, tau={self.tau}, target_entropy={self.target_entropy}")
