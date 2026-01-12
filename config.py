# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    # I/O
    rl_dir: str = "./encoder/rl_io"
    poll_ms: int = 10
    fb_timeout_ms: int = 120_000    # per-MG feedback wait
    drain_timeout_ms: int = 10_000  # after encoder exit, wait to drain queue

    # Encoder
    encoder_path: str = "./encoder/qav1enc"  # override via --encoder
    show_encoder_output: bool = False
    encoder_log_to_file: bool = True
    encoder_log_dir: str = "./logs/encoder"
    fps: float = 25.0  # 帧率，用于计算 kbps

    # MG / QP / Q_VAL
    frames_per_mg: int = 16
    qp_min: int = 30
    qp_max: int = 210
    q_val_min: float = 20.0  # q_val 的最小值
    q_val_max: float = 160.0  # q_val 的最大值
    action_min: int = 30
    action_max: int = 210
    action_step: int = 1
    action_space_type: str = "discrete"  # "continuous" or "discrete"
    num_discrete_actions: int = 0  # 0 => auto (by action_step)
    discrete_action_values: Optional[List[float]] = None

    # Preproc (feature)
    apply_log_comp: bool = True
    apply_log_rdcost: bool = True
    apply_log_bit_target: bool = True
    normalize_score_target: bool = True
    robust_scale_seq: bool = True
    robust_clip: float = 5.0

    # D3QN
    device: str = "cuda"
    hidden_dim: int = 512
    lr_critic: float = 3e-4
    gamma: float = 0.99

    # D3QN specific
    dqn_target_update_interval: int = 200
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay: float = 20000.0

    # Training
    batch_size: int = 128
    replay_size: int = 2000
    start_steps: int = 10
    updates_per_step: int = 4
    seed: int = 42
    gamma: float = 0.99  # 折扣因子（D3QN 使用）

    # GOP-level processing
    gop_size_standard: int = 225      # 标准 GOP 大小
    default_qp: int = 127             # 默认 QP 值（用于第一个 GOP）

    # GOP Reward 配置
    bitrate_save_weight: float = 1.0      # 质量达标时码率节省的奖励权重
    quality_smooth_weight: float = 0.1    # GOP 间质量平滑惩罚权重
    
    # 序列下采样配置
    enable_smart_downsample: bool = True  # 是否开启智能下采样（保留 I/P 帧，池化 B 帧）
    seq_target_T: int = 64                # 下采样目标长度（设为 225 则不下采样）

    # Checkpoint
    ckpt_dir: str = "./checkpoints"
    ckpt_interval: int = 5  # 每 N 个 epoch 保存一次
    save_replay_buffer: bool = True  # 是否保存 replay buffer
    load_checkpoint: Optional[str] = None  # 加载检查点路径

    # Mode
    mode: str = "train"  # "train" or "infer"
    
    # Logging
    log_level: int = 2  # 0=静默, 1=简洁, 2=详细, 3=调试
    log_interval_mg: int = 20
    log_action_qp: bool = True  # print per-MG action QP

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    tb_log_interval: int = 1  # 每 N 个训练步记录一次


    def __post_init__(self) -> None:
        if self.action_space_type != "discrete":
            return
        step = max(1, int(getattr(self, "action_step", 1)))
        action_min = int(getattr(self, "action_min", 0))
        action_max = int(getattr(self, "action_max", action_min))
        if action_max < action_min:
            action_max = action_min
        if self.discrete_action_values is None:
            if self.num_discrete_actions <= 0:
                values = list(range(action_min, action_max + 1, step))
                if not values:
                    values = [action_min]
                self.discrete_action_values = [float(v) for v in values]
                self.num_discrete_actions = len(self.discrete_action_values)
            else:
                if self.num_discrete_actions <= 1:
                    self.discrete_action_values = [float(action_min)]
                else:
                    span = float(action_max - action_min)
                    step_f = span / float(self.num_discrete_actions - 1)
                    self.discrete_action_values = [
                        float(action_min) + i * step_f for i in range(self.num_discrete_actions)
                    ]
        else:
            self.num_discrete_actions = int(len(self.discrete_action_values))
