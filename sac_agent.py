# -*- coding: utf-8 -*-
import math, torch, torch.nn.functional as F
import torch.nn as nn
from models import Actor, Critic

class SACAgent:
    def __init__(self, cfg, state_scalar_dim: int, seq_T: int, seq_C: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = Actor(state_scalar_dim, seq_C, seq_T, cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_scalar_dim, seq_C, seq_T, cfg.hidden_dim).to(self.device)
        self.critic_tgt = Critic(state_scalar_dim, seq_C, seq_T, cfg.hidden_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic= torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.gamma = cfg.gamma; self.tau = cfg.tau
        self.target_entropy = -1.0 if cfg.target_entropy == 0.0 else cfg.target_entropy

    @torch.no_grad()
    def act(self, seq1c_t, scalars, deterministic=False):
        mu, log_std = self.actor(seq1c_t, scalars)
        if deterministic:
            a = torch.tanh(mu); logp = torch.zeros_like(mu)
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            z = mu + std * eps
            a = torch.tanh(z)
            log_prob_gauss = -0.5 * (((z - mu) / (std + 1e-6))**2 + 2*log_std + math.log(2*math.pi))
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_det = torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            logp = log_prob_gauss - log_det
        return a.clamp(-1,1), logp

    def train_step(self, batch):
        seq, sca, a, r, seq2, sca2, d = batch
        with torch.no_grad():
            a2, logp2 = self.act(seq2, sca2, deterministic=False)
            q1_t, q2_t = self.critic_tgt(seq2, sca2, a2)
            alpha = self.log_alpha.exp()
            y = r + (1.0 - d) * self.gamma * (torch.min(q1_t, q2_t) - alpha * logp2)

        q1, q2 = self.critic(seq, sca, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_critic.step()

        a_new, logp = self.act(seq, sca, deterministic=False)
        q1_new, q2_new = self.critic(seq, sca, a_new)
        alpha = self.log_alpha.exp()
        loss_actor = (alpha * logp - torch.min(q1_new, q2_new)).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return (float(loss_q.item()), float(loss_actor.item()), float(alpha.detach().cpu()))
    
    def save_checkpoint(self, path: str) -> None:
        """保存模型检查点"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_tgt_state_dict': self.critic_tgt.state_dict(),
            'opt_actor_state_dict': self.opt_actor.state_dict(),
            'opt_critic_state_dict': self.opt_critic.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'opt_alpha_state_dict': self.opt_alpha.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'target_entropy': self.target_entropy,
        }
        torch.save(checkpoint, path)
        print(f"[Checkpoint] 已保存模型 -> {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载模型检查点"""
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_tgt.load_state_dict(checkpoint['critic_tgt_state_dict'])
        self.opt_actor.load_state_dict(checkpoint['opt_actor_state_dict'])
        self.opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
        
        # 恢复 log_alpha
        self.log_alpha = torch.tensor(
            checkpoint['log_alpha'], 
            device=self.device, 
            requires_grad=True
        )
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        self.opt_alpha.load_state_dict(checkpoint['opt_alpha_state_dict'])
        
        print(f"[Checkpoint] 已加载模型 <- {path}")