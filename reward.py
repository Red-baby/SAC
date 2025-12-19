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
    bitrate_tolerance: float
    bitrate_hard_ratio: float
    over_bitrate_penalty: float
    shaping_w_score_ema: float
    term_bonus: float
    term_tau: float

class RewardComputer:
    def __init__(self, cfg: RewardCfg):
        self.cfg = cfg
        self.lam = float(cfg.lambda_init)
        self.score_ema = EMA(beta=0.9, init=0.0)
        self._phi_prev = 0.0
        self._db_cum_prev = 0.0

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
        self._db_cum_prev = 0.0
        self.score_ema = EMA(beta=0.9, init=0.0)

    def step(self, bits: float, score: float, bits_alloc: float, score_alloc: float, delta_qp: float, num_frames: int = 0) -> float:
        eps = 1e-6
        hard_ratio = max(0.0, float(self.cfg.bitrate_hard_ratio))
        over_penalty = max(0.0, float(self.cfg.over_bitrate_penalty))
        
        # Calculate cumulative values including current step
        cum_bits = self.gop_bits_sum + float(bits)
        cum_score = self.gop_score_sum + float(score)
        cum_bits_alloc = self.gop_bits_alloc_sum + max(float(bits_alloc), 0.0)
        cum_score_alloc = self.gop_score_alloc_sum + max(float(score_alloc), 0.0)
        
        # Use per-step quality delta to avoid early penalty dominating the rest of the GOP.
        dq_step = (float(score) - float(score_alloc)) / 100.0
        
        # dB_cum: Cumulative Bitrate Deviation
        if cum_bits_alloc > eps:
            db_cum_raw = (cum_bits - cum_bits_alloc) / cum_bits_alloc
        else:
            db_cum_raw = 0.0
        # Apply tolerance band so small抖动不影响奖励；超出±bit_tol部分才计入惩罚/奖励
        # Two-stage reward:
        # - If bitrate exceeds reference by > hard_ratio (e.g. 5%), apply a strict pure-bit penalty,
        #   ignoring objective quality (score) to avoid bitrate/quality tradeoff.
        # - Otherwise, only compare objective quality (score), ignoring bitrate penalty entirely.
        if db_cum_raw > hard_ratio:
            # Penalize only the *incremental* excess to prevent early overshoot
            # from causing a persistent negative reward across the whole GOP.
            prev_excess = max(0.0, self._db_cum_prev - hard_ratio)
            curr_excess = db_cum_raw - hard_ratio
            delta_excess = max(0.0, curr_excess - prev_excess)
            r = -over_penalty * delta_excess
            apply_shaping = False
        else:
            r = dq_step
            apply_shaping = True
        self._db_cum_prev = db_cum_raw

        # Reward Shaping (Potential-based)
        if apply_shaping and self.cfg.shaping_w_score_ema != 0.0:
            ema_val = self.score_ema.update(float(score))
            phi_t = self.cfg.shaping_w_score_ema * (ema_val / 100.0)
            r += self.cfg.gamma * phi_t - self._phi_prev
            self._phi_prev = phi_t
        else:
            self._phi_prev = 0.0

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
        bit_tol = max(0.0, float(self.cfg.bitrate_tolerance))
        hard_ratio = max(0.0, float(self.cfg.bitrate_hard_ratio))
        B_alloc_T = max(self.gop_bits_alloc_sum, eps)
        Q_alloc_T = max(self.gop_score_alloc_sum, eps)
        dB_T_norm = (self.gop_bits_sum  - B_alloc_T) / B_alloc_T
        dQ_T_norm = (self.gop_score_sum - Q_alloc_T) / Q_alloc_T
        dB_T_eff = math.copysign(max(0.0, abs(dB_T_norm) - bit_tol), dB_T_norm)
        dB_over_excess = max(0.0, float(dB_T_norm) - hard_ratio)

        # Keep lambda update for backward compatibility (mostly for logging/checkpoint),
        # but only react to the excess over hard_ratio so it matches the hard constraint.
        self.lam = max(0.0, self.lam + self.cfg.lambda_lr * float(dB_over_excess))

        term = 0.0
        if self.cfg.term_bonus > 0.0:
            tau = max(float(self.cfg.term_tau), eps)
            if float(dB_T_norm) > hard_ratio:
                # Over hard bitrate cap: only penalize bits (no quality term).
                term = -abs(self.cfg.term_bonus) * (1.0 - math.exp(-dB_over_excess / tau))
            else:
                # Within hard bitrate cap: only use quality (no bitrate term).
                z = float(dQ_T_norm) / tau
                term = self.cfg.term_bonus * (1.0 / (1.0 + math.exp(-z)) - 0.5) * 2.0
            self.episode_return += term

        info = {
            "steps": self.mg_in_gop,
            "sum_bits": self.gop_bits_sum,
            "sum_bits_alloc": B_alloc_T,
            "delta_bits_norm": dB_T_norm,
            "delta_bits_norm_eff": dB_T_eff,
            "delta_bits_over_excess": dB_over_excess,
            "bitrate_hard_ratio": hard_ratio,
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
