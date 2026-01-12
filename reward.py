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
        bitrate_save_weight = float(self.cfg.bitrate_save_weight)
        quality_smooth_weight = float(self.cfg.quality_smooth_weight)
        
        # 计算码率比值
        target_bitrate = max(target_bitrate, eps)
        bitrate_ratio = bitrate / target_bitrate
        
        # 计算质量差异
        quality_gap = (target_score - score) / max(target_score, eps)
        
        # === 主要奖励逻辑 ===
        r = 0.0
        
        if score >= target_score:
            # 码率节省奖励：节省越多奖励越高（clip 到合理范围）
            bitrate_saved = min(0.5, max(0.0, 1.0 - bitrate_ratio))  # clip 到 [0, 0.5]
            r_bitrate_save = bitrate_save_weight * bitrate_saved
            
            # 质量超标奖励（小幅度，clip 防止过大）
            quality_bonus = min(0.1, (score - target_score) / 100.0)  # clip 到 [0, 0.1]
            r_quality = 0.1 * quality_bonus
            
            r += r_bitrate_save + r_quality
            
        else:
            # === 质量未达标：惩罚质量差距 ===
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
        if self.last_gop_score is not None and not is_first_gop:
            score_diff = abs(score - self.last_gop_score)
            score_diff_norm = score_diff / 5.0  # 归一化（5 dB 差异是很大的波动）
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
        
        return float(r)
    
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
