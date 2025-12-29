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
    qp_min: int = 48  # 保留用于向后兼容
    qp_max: int = 252  # 保留用于向后兼容
    q_val_min: float = 20.0  # q_val 的最小值
    q_val_max: float = 160.0  # q_val 的最大值
    delta_qp_max: int = 20
    delta_qp_step: int = 2  # step size for discrete delta_qp values
    action_space_type: str = "discrete"  # "continuous" or "discrete"
    num_discrete_actions: int = 0  # 0 => auto (by delta_qp_step)
    discrete_action_values: Optional[List[float]] = None

    # Preproc (feature)
    apply_log_comp: bool = True
    apply_log_rdcost: bool = True
    apply_log_bit_target: bool = True
    normalize_score_target: bool = True
    robust_scale_seq: bool = True
    robust_clip: float = 5.0

    # SAC v2
    device: str = "cuda"
    hidden_dim: int = 512
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    target_entropy: float = 3.0  # use default -1 if 0
    num_action_samples: int = 8  # samples for discrete policy update
    init_alpha: float = 0.1
    tau: float = 0.005
    gamma: float = 0.99

    # Training
    batch_size: int = 128
    replay_size: int = 2000
    start_steps: int = 10
    updates_per_step: int = 4
    seed: int = 42
    baseline_stats_path: Optional[str] = None
    baseline_action_prob: float = 0.1  # chance to use zero-delta action during training

    # Reward / constraint
    smooth_penalty: float = 0.02
    lambda_init: float = 5.0
    lambda_lr: float = 1e-2
    bitrate_tolerance: float = 0.10  # 允许码率在 +/-10% 波动区间内不计惩罚
    bitrate_hard_ratio: float = 0.05  # hard cap: strictly penalize when > +5% over reference
    over_bitrate_penalty: float = 50.0  # pure-bit penalty scale for excess over hard cap
    term_bonus: float = 0.0
    term_tau: float = 0.01
    shaping_w_score_ema: float = 0

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
    log_delta_qvals: bool = True  # print per-MG delta_qps list

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    tb_log_interval: int = 1  # 每 N 个训练步记录一次


    def __post_init__(self) -> None:
        if self.action_space_type != "discrete":
            return
        step = max(1, int(getattr(self, "delta_qp_step", 1)))
        if self.discrete_action_values is None:
            if self.num_discrete_actions <= 0:
                values = list(range(-int(self.delta_qp_max), int(self.delta_qp_max) + 1, step))
                if not values:
                    values = [0]
                self.discrete_action_values = [float(v) for v in values]
                self.num_discrete_actions = len(self.discrete_action_values)
            else:
                if self.num_discrete_actions <= 1:
                    self.discrete_action_values = [0.0]
                elif self.num_discrete_actions == int(self.delta_qp_max) * 2 + 1 and step == 1:
                    self.discrete_action_values = [
                        float(v) for v in range(-int(self.delta_qp_max), int(self.delta_qp_max) + 1)
                    ]
                else:
                    span = 2 * float(self.delta_qp_max)
                    step_f = span / float(self.num_discrete_actions - 1)
                    self.discrete_action_values = [
                        -float(self.delta_qp_max) + i * step_f for i in range(self.num_discrete_actions)
                    ]
        else:
            self.num_discrete_actions = int(len(self.discrete_action_values))
