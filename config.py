# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

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

    # MG / QP
    frames_per_mg: int = 16
    qp_min: int = 48
    qp_max: int = 252
    delta_qp_max: int = 10

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
    target_entropy: float = 0.0  # use default -1 if 0
    init_alpha: float = 0.1
    tau: float = 0.005
    gamma: float = 0.99

    # Training
    batch_size: int = 128
    replay_size: int = 200_000
    start_steps: int = 500
    updates_per_step: int = 1
    seed: int = 42
    baseline_stats_path: Optional[str] = None

    # Reward / constraint
    smooth_penalty: float = 0.02
    lambda_init: float = 0.0
    lambda_lr: float = 1e-4
    term_bonus: float = 0.5
    term_tau: float = 0.01
    shaping_w_score_ema: float = 0.05

    # Checkpoint
    ckpt_dir: str = "./checkpoints"
    ckpt_interval: int = 5  # 每 N 个 epoch 保存一次
    save_replay_buffer: bool = True  # 是否保存 replay buffer
    load_checkpoint: Optional[str] = None  # 加载检查点路径
    
    # Logging
    log_level: int = 1  # 0=静默, 1=简洁, 2=详细, 3=调试
    log_interval_mg: int = 20
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    tb_log_interval: int = 1  # 每 N 个训练步记录一次
