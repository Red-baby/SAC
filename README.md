# RL Agent for Video Encoding QP Control (v3.0)

基于强化学习的视频编码 QP（量化参数）控制代理，支持 **Mini-GOP 级别** 和 **GOP 级别** 两种处理模式。

## ✨ 新功能（v3.0）

- 🎯 **GOP 级别 QP 控制**：支持 GOP 级别的单 QP 输出，实现质量达标时自动降码率
- 🧠 **Self-Attention 架构**：替代 GRU，更好地捕捉帧间复杂度关系
- ⚖️ **平衡 Reward 设计**：质量达标优先节省码率，GOP 间质量平滑
- 📉 **序列下采样**：225 帧 → 64 帧，加速推理

## 处理模式

### GOP 级别模式（推荐）
```python
from gop_runner import GOPRunner

runner = GOPRunner(cfg)
runner.serve_loop(stop_event)
```

**文件格式**：
- 编码器写：`gop%04d_rq.json`、`gop%04d_fb.json`
- RL 写：`gop%04d_qp.json`（`{"qp": 127}`）

### Mini-GOP 级别模式
```python
from io_runner import RLRunner

runner = RLRunner(cfg)
runner.serve_loop(stop_event)
```

## 快速开始

### 训练
```bash
python main.py --epochs 20 --log-level 2 --ckpt-interval 5
```

### 从检查点继续
```bash
python main.py --load-checkpoint ./checkpoints/checkpoint_epoch_10.pt
```

### TensorBoard 监控
```bash
tensorboard --logdir=./runs
```

## 核心配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `gop_size_standard` | 225 | 标准 GOP 大小 |
| `seq_target_T` | 64 | 序列下采样长度 |
| `bitrate_save_weight` | 1.0 | 码率节省奖励权重 |
| `quality_smooth_weight` | 0.1 | 质量平滑惩罚权重 |
| `default_qp` | 127 | 默认 QP 值 |

## 网络架构

```
Seq[6, 64] → Conv1D → PositionalEncoding → [CLS] + Tokens
           → TransformerEncoder (2层, 4头)
           → [CLS] Output → + Scalars[11] → MLP → Q-values
```

**Scalars 特征 (11维)**：
1. gop_progress, bitrate_ratio, encoded_score, encoded_comp
2. last_bitrate_ratio, last_score, last_comp, last_qpbase
3. target_score, target_bitrate, **is_first_gop**

## Reward 设计

```
if score >= target_score:
    r = bitrate_save_weight * min(0.5, 1 - bitrate_ratio)  # 奖励码率节省
else:
    r = -min(0.5, quality_gap)  # 惩罚质量差距

r -= quality_smooth_weight * |score - last_score|  # 平滑惩罚
```

## 项目结构
```
SAC/
├── main.py              # 主入口
├── config.py            # 配置（含 GOP 级别参数）
├── gop_runner.py        # GOP 级别 Runner（新）
├── io_runner.py         # Mini-GOP 级别 Runner
├── models.py            # 网络（Self-Attention）
├── state.py             # 状态构建（含下采样）
├── reward.py            # 奖励函数（含 GOPRewardComputer）
├── sac_agent.py         # D3QN 算法
├── replay.py            # Replay Buffer
├── utils.py             # 工具函数
├── checkpoints/         # 检查点
└── runs/                # TensorBoard 日志
```

## 依赖
```bash
pip install torch numpy tensorboard
```

- Python 3.9+
- PyTorch >= 2.0
