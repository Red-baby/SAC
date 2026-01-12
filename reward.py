# -*- coding: utf-8 -*-
"""
GOP 级别 Reward 模块（简化版）

只保留 GOPRewardComputer，删除了 Mini-GOP 相关的 RewardComputer。
"""
from dataclasses import dataclass


class EMA:
    """指数移动平均"""
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
    """GOP Reward 配置（简化版）"""
    bitrate_save_weight: float = 1.0      # 质量达标时码率节省的奖励权重
    quality_smooth_weight: float = 0.1    # GOP 间质量平滑惩罚权重


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
        
        # 记录每个 GOP 的 score（用于计算 episode bonus）
        self.gop_scores = []
        
    def reset(self):
        """重置 episode 状态"""
        self.last_gop_score = None
        self.gop_scores = []
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
        bitrate_save_weight = float(self.cfg.bitrate_save_weight)
        quality_smooth_weight = float(self.cfg.quality_smooth_weight)
        
        # 计算码率比值
        target_bitrate = max(target_bitrate, eps)
        bitrate_ratio = bitrate / target_bitrate
        
        # 计算质量差异
        quality_gap = (target_score - score) / max(target_score, eps)
        
        # === 码率硬约束检查（30% 涨幅限制）===
        BITRATE_HARD_CAP = 1.30  # 允许最多 +30%
        if bitrate_ratio > BITRATE_HARD_CAP:
            # 严重超标：立即给予大幅惩罚，忽略质量
            over_ratio = bitrate_ratio - BITRATE_HARD_CAP
            r = -min(5.0, over_ratio * 5.0)  # 超标越多惩罚越重
            # 记录状态后直接返回
            self.last_gop_score = score
            self.score_ema.update(score)
            self.episode_return += r
            self.gop_count += 1
            self.total_bits += bitrate * gop_size
            self.total_score += score * gop_size
            self.total_target_bits += target_bitrate * gop_size
            self.total_target_score += target_score * gop_size
            self.total_frames += gop_size
            self.gop_scores.append(score)
            return float(r)
        
        # === 主要奖励逻辑 ===
        r = 0.0
        
        if score >= target_score:
            # 质量达标：奖励码率节省
            if bitrate_ratio <= 1.0:
                # 码率节省：奖励
                bitrate_saved = min(0.5, max(0.0, 1.0 - bitrate_ratio))
                r_bitrate_save = bitrate_save_weight * bitrate_saved
            else:
                # 码率超标但在硬约束内：轻微惩罚
                over_ratio = min(0.3, bitrate_ratio - 1.0)  # clip 到 [0, 0.3]
                r_bitrate_save = -over_ratio * 1.5  # 惩罚系数 1.5
            
            # 质量超标奖励（小幅度）
            quality_bonus = min(0.1, (score - target_score) / 100.0)
            r_quality = 0.1 * quality_bonus
            
            r += r_bitrate_save + r_quality
            
        else:
            # === 质量未达标：惩罚质量差距 ===
            quality_gap_clipped = min(0.5, quality_gap)
            r_quality_penalty = -quality_gap_clipped
            
            # 如果码率已经很高但质量仍未达标，额外惩罚
            if bitrate_ratio > 1.0:
                over_bitrate = min(0.5, bitrate_ratio - 1.0)
                r_over_bitrate = -0.5 * over_bitrate
            else:
                r_over_bitrate = 0.0
            
            r += r_quality_penalty + r_over_bitrate
        
        # === 质量平滑惩罚（减少 GOP 间震荡）===
        if self.last_gop_score is not None and not is_first_gop:
            score_diff = abs(score - self.last_gop_score)
            score_diff_norm = score_diff / 5.0
            r_smooth = -quality_smooth_weight * score_diff_norm
            r += r_smooth

        
        # 更新状态
        self.last_gop_score = score
        self.score_ema.update(score)
        self.episode_return += r
        self.gop_count += 1
        self.total_bits += bitrate * gop_size
        self.total_score += score * gop_size
        self.total_target_bits += target_bitrate * gop_size
        self.total_target_score += target_score * gop_size
        self.total_frames += gop_size
        
        # 记录当前 GOP 的 score（用于 episode bonus）
        self.gop_scores.append(score)
        
        return float(r)
    
    def compute_episode_bonus(self) -> float:
        """
        计算 episode 级别的奖励 bonus
        
        考虑因素：
        1. 整体码率控制精度
        2. 整体质量达标情况
        3. 质量一致性（方差）
        """
        eps = 1e-6
        bonus = 0.0
        
        if self.total_frames == 0 or self.gop_count == 0:
            return 0.0
        
        # 1. 整体码率控制精度
        avg_bitrate = self.total_bits / self.total_frames
        avg_target_bitrate = self.total_target_bits / self.total_frames
        bitrate_error = abs(avg_bitrate - avg_target_bitrate) / max(avg_target_bitrate, eps)
        
        if bitrate_error < 0.05:  # 在 ±5% 内
            bonus += 0.5
        else:
            bonus -= min(1.0, bitrate_error * 2.0)
        
        # 2. 整体质量达标
        avg_score = self.total_score / self.total_frames
        avg_target_score = self.total_target_score / self.total_frames
        quality_error = max(0, avg_target_score - avg_score) / max(avg_target_score, eps)
        
        if quality_error < 0.02:  # 质量差距小于 2%
            bonus += 0.5
        else:
            bonus -= min(1.0, quality_error * 3.0)
        
        # 3. 质量一致性（方差越小越好）
        if len(self.gop_scores) > 1:
            import numpy as np
            score_variance = float(np.var(self.gop_scores))
            consistency_penalty = min(0.5, score_variance / 100.0)  # 归一化
            bonus -= consistency_penalty
        
        # Clip 到合理范围
        import numpy as np
        return float(np.clip(bonus, -3.0, 3.0))
    
    def on_episode_end(self) -> dict:
        """Episode 结束时的总结"""
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
