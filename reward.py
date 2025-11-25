# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math

class EMA:
    def __init__(self, beta=0.9, init=None):
        self.beta = float(beta)
        self.val = None if init is None else float(init)
    def update(self, x: float) -> float:
        if self.val is None:
            self.val = float(x)
        else:
            self.val = self.beta * self.val + (1.0 - self.beta) * float(x)
        return self.val
    def get(self) -> float:
        return 0.0 if self.val is None else float(self.val)

@dataclass
class RewardCfg:
    gamma: float
    smooth_penalty: float
    lambda_init: float
    lambda_lr: float
    shaping_w_score_ema: float
    term_bonus: float
    term_tau: float

class RewardComputer:
    def __init__(self, cfg: RewardCfg):
        self.cfg = cfg
        self.lam = float(cfg.lambda_init)
        self.score_ema = EMA(beta=0.9, init=0.0)
        self._phi_prev = 0.0

        self.gop_bits_sum = 0.0
        self.gop_score_sum = 0.0
        self.gop_bits_alloc_sum = 0.0
        self.gop_score_alloc_sum = 0.0
        self.gop_frames_sum = 0  # 累计帧数

        self.mg_in_gop = 0
        self.episode_return = 0.0

    def reset_gop(self):
        self.gop_bits_sum = 0.0
        self.gop_score_sum = 0.0
        self.gop_bits_alloc_sum = 0.0
        self.gop_score_alloc_sum = 0.0
        self.gop_frames_sum = 0
        self.mg_in_gop = 0
        self.episode_return = 0.0
        self._phi_prev = 0.0
        self.score_ema = EMA(beta=0.9, init=0.0)

    def step(self, bits: float, score: float, bits_alloc: float, score_alloc: float, delta_qp: float, num_frames: int = 0) -> float:
        eps = 1e-6
        
        # Calculate cumulative values including current step
        cum_bits = self.gop_bits_sum + float(bits)
        cum_score = self.gop_score_sum + float(score)
        cum_bits_alloc = self.gop_bits_alloc_sum + max(float(bits_alloc), 0.0)
        cum_score_alloc = self.gop_score_alloc_sum + max(float(score_alloc), 0.0)
        
        # Calculate cumulative normalized deviation
        # dQ_cum: Cumulative Quality Deviation
        dq_cum = (cum_score - cum_score_alloc) / 100.0
        
        # dB_cum: Cumulative Bitrate Deviation
        if cum_bits_alloc > eps:
            db_cum = (cum_bits - cum_bits_alloc) / cum_bits_alloc
        else:
            db_cum = 0.0

        # Reward formula: Cumulative Quality - Lambda * Cumulative Bits
        r = dq_cum - self.lam * db_cum

        # Reward Shaping (Potential-based)
        ema_val = self.score_ema.update(float(score))
        phi_t = self.cfg.shaping_w_score_ema * (ema_val / 100.0)
        r += self.cfg.gamma * phi_t - self._phi_prev
        self._phi_prev = phi_t

        # Update state
        self.gop_bits_sum = cum_bits
        self.gop_score_sum = cum_score
        self.gop_bits_alloc_sum = cum_bits_alloc
        self.gop_score_alloc_sum = cum_score_alloc
        self.gop_frames_sum += max(int(num_frames), 0)
        self.mg_in_gop += 1

        self.episode_return += float(r)
        return float(r)

    def on_gop_end(self):
        eps = 1e-6
        B_alloc_T = max(self.gop_bits_alloc_sum, eps)
        Q_alloc_T = max(self.gop_score_alloc_sum, eps)
        dB_T_norm = (self.gop_bits_sum  - B_alloc_T) / B_alloc_T
        dQ_T_norm = (self.gop_score_sum - Q_alloc_T) / Q_alloc_T

        self.lam = max(0.0, self.lam + self.cfg.lambda_lr * float(dB_T_norm))

        term = 0.0
        if self.cfg.term_bonus > 0.0:
            z = (float(dQ_T_norm) - abs(float(dB_T_norm))) / max(self.cfg.term_tau, eps)
            term = self.cfg.term_bonus * (1.0 / (1.0 + math.exp(-z)) - 0.5) * 2.0
            self.episode_return += term

        info = {
            "steps": self.mg_in_gop,
            "sum_bits": self.gop_bits_sum,
            "sum_bits_alloc": B_alloc_T,
            "delta_bits_norm": dB_T_norm,
            "sum_score": self.gop_score_sum,
            "sum_score_alloc": Q_alloc_T,
            "delta_score_norm": dQ_T_norm,
            "num_frames": self.gop_frames_sum,
            "lambda": self.lam,
            "term_bonus": term,
            "episode_return": self.episode_return,
        }

        self.reset_gop()
        return info
