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
    # GOP-level 新增配置
    bitrate_save_weight: float = 1.0      # 质量达标时码率节省的奖励权重
    quality_smooth_weight: float = 0.1    # GOP 间质量平滑惩罚权重


class RewardComputer:
    """原始的 Mini-GOP 级别奖励计算器（保持向后兼容）"""
    
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
        
        # dB_step: per-MG bitrate deviation (do not use GOP cumulative for penalty)
        if bits_alloc > eps:
            db_step_raw = (float(bits) - float(bits_alloc)) / float(bits_alloc)
        else:
            db_step_raw = 0.0
        # Apply tolerance band so small抖动不影响奖励；超出±bit_tol部分才计入惩罚/奖励
        # Two-stage reward:
        # - If bitrate exceeds reference by > hard_ratio (e.g. 5%), apply a strict pure-bit penalty,
        #   ignoring objective quality (score) to avoid bitrate/quality tradeoff.
        # - Otherwise, only compare objective quality (score), ignoring bitrate penalty entirely.
        if db_step_raw > hard_ratio:
            excess = db_step_raw - hard_ratio
            r = -over_penalty * excess
            apply_shaping = False
        else:
            r = dq_step
            apply_shaping = True
        self._db_cum_prev = db_step_raw

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


class GOPRewardComputer:
    """
    GOP 级别的奖励计算器
    
    设计目标：
    1. 当质量达到 target_score 时，优先奖励码率节省
    2. 减少 GOP 之间的质量震荡
    """
    
    def __init__(self, cfg: RewardCfg):
        self.cfg = cfg
        
        # 上一个 GOP 的质量（用于平滑计算）
        self.last_gop_score = None
        
        # 累计 episode 信息
        self.episode_return = 0.0
        self.gop_count = 0
        self.total_bits = 0.0
        self.total_score = 0.0
        self.total_target_bits = 0.0
        self.total_target_score = 0.0
        self.total_frames = 0
        
        # EMA 用于追踪平均质量
        self.score_ema = EMA(beta=0.9, init=None)
        
    def reset(self):
        """重置 episode 状态"""
        self.last_gop_score = None
        self.episode_return = 0.0
        self.gop_count = 0
        self.total_bits = 0.0
        self.total_score = 0.0
        self.total_target_bits = 0.0
        self.total_target_score = 0.0
        self.total_frames = 0
        self.score_ema = EMA(beta=0.9, init=None)
    
    def step(self, 
             bitrate: float, 
             score: float, 
             target_bitrate: float, 
             target_score: float,
             gop_size: int,
             is_first_gop: bool = False) -> float:
        """
        计算单个 GOP 的奖励
        
        Args:
            bitrate: 实际编码比特率
            score: 实际编码质量
            target_bitrate: 目标比特率
            target_score: 目标质量
            gop_size: GOP 帧数
            is_first_gop: 是否为第一个 GOP
            
        Returns:
            reward: 当前 GOP 的奖励值
        """
        eps = 1e-6
        
        # 权重配置
        bitrate_save_weight = float(getattr(self.cfg, 'bitrate_save_weight', 1.0))
        quality_smooth_weight = float(getattr(self.cfg, 'quality_smooth_weight', 0.1))
        
        # 计算码率比值
        target_bitrate = max(target_bitrate, eps)
        bitrate_ratio = bitrate / target_bitrate
        
        # 计算质量差异
        quality_gap = (target_score - score) / max(target_score, eps)
        
        # === 主要奖励逻辑 ===
        r = 0.0
        
        if score >= target_score:
            # 码率节省奖励：节省越多奖励越高（clip 到合理范围）
            # bitrate_ratio < 1.0 表示节省了码率
            bitrate_saved = min(0.5, max(0.0, 1.0 - bitrate_ratio))  # clip 到 [0, 0.5]
            r_bitrate_save = bitrate_save_weight * bitrate_saved
            
            # 质量超标奖励（小幅度，clip 防止过大）
            quality_bonus = min(0.1, (score - target_score) / 100.0)  # clip 到 [0, 0.1]
            r_quality = 0.1 * quality_bonus
            
            r += r_bitrate_save + r_quality
            
        else:
            # === 质量未达标：惩罚质量差距 ===
            # 质量差距惩罚（clip 到 [0, 0.5] 与码率节省奖励对称）
            quality_gap_clipped = min(0.5, quality_gap)  # clip 到 [0, 0.5]
            r_quality_penalty = -quality_gap_clipped
            
            # 如果码率已经很高但质量仍未达标，额外惩罚
            if bitrate_ratio > 1.0:
                over_bitrate = min(0.5, bitrate_ratio - 1.0)  # clip 超标部分
                r_over_bitrate = -0.5 * over_bitrate
            else:
                r_over_bitrate = 0.0
            
            r += r_quality_penalty + r_over_bitrate
        
        # === 质量平滑惩罚（减少 GOP 间震荡）===
        r_smooth = 0.0
        if self.last_gop_score is not None and not is_first_gop:
            # 计算与上一个 GOP 的质量差异
            score_diff = abs(score - self.last_gop_score)
            # 归一化差异（假设 5 dB 差异是很大的波动）
            score_diff_norm = score_diff / 5.0
            # 应用平滑惩罚
            r_smooth = -quality_smooth_weight * score_diff_norm
            r += r_smooth
        
        # 更新状态
        self.last_gop_score = score
        self.score_ema.update(score)
        self.episode_return += r
        self.gop_count += 1
        self.total_bits += bitrate * gop_size  # 近似总比特数
        self.total_score += score * gop_size   # 加权总质量
        self.total_target_bits += target_bitrate * gop_size
        self.total_target_score += target_score * gop_size
        self.total_frames += gop_size
        
        return float(r)
    
    def on_episode_end(self) -> dict:
        """
        Episode 结束时的总结
        
        Returns:
            info: 包含 episode 统计信息的字典
        """
        eps = 1e-6
        
        # 计算整体统计
        avg_bitrate = self.total_bits / max(self.total_frames, 1)
        avg_score = self.total_score / max(self.total_frames, 1)
        avg_target_bitrate = self.total_target_bits / max(self.total_frames, 1)
        avg_target_score = self.total_target_score / max(self.total_frames, 1)
        
        # 计算比值
        bitrate_ratio = avg_bitrate / max(avg_target_bitrate, eps)
        score_ratio = avg_score / max(avg_target_score, eps)
        
        # 计算码率节省和质量变化
        bitrate_saved_pct = (1.0 - bitrate_ratio) * 100.0
        score_diff_pct = (score_ratio - 1.0) * 100.0
        
        info = {
            "gop_count": self.gop_count,
            "total_frames": self.total_frames,
            "episode_return": self.episode_return,
            "avg_bitrate": avg_bitrate,
            "avg_score": avg_score,
            "avg_target_bitrate": avg_target_bitrate,
            "avg_target_score": avg_target_score,
            "bitrate_ratio": bitrate_ratio,
            "score_ratio": score_ratio,
            "bitrate_saved_pct": bitrate_saved_pct,
            "score_diff_pct": score_diff_pct,
            "score_ema": self.score_ema.get(),
        }
        
        # 重置状态
        self.reset()
        
        return info
