# RL Agent for MiniGOP ΔQP (SAC v2)

- 进程交互：编码器与 RL 通过目录 `rl_io/` 文件握手
  - 编码器写：`mg%04d_rq.json`（请求）、`mg%04d_fb.json`（反馈）
  - RL 写：`mg%04d_qp.json`（QP 决策，JSON: {"qp": N}）
- 终止：`fb.gop_end==1` 表示一个 episode（GOP）结束，RL 在 GOP 末更新约束拉格朗日系数 λ。

## ✨ 新功能（v2.0）

- 🎚️ **日志级别控制**：4 级日志（静默/重要/详细/调试），提高可读性
- 💾 **Checkpoint 管理**：定期保存和加载完整训练状态（模型+Replay Buffer）
- 📊 **TensorBoard 可视化**：实时监控训练曲线和性能指标

📖 **详细文档**：查看 [FEATURES.md](FEATURES.md) 了解完整使用说明。

## 快速开始

### 基础训练
```bash
# 训练 20 个 epoch，详细日志，每 5 个 epoch 保存检查点
python main.py --epochs 20 --log-level 2 --ckpt-interval 5
```

### 从检查点继续训练
```bash
# 加载 epoch 10 的检查点，继续训练
python main.py --load-checkpoint ./checkpoints/checkpoint_epoch_10.pt --epochs 30
```

### 查看训练曲线
```bash
# 启动 TensorBoard（在另一个终端）
tensorboard --logdir=./runs
# 然后在浏览器打开: http://localhost:6006
```

## 运行模式

### 单视频命令（用 `|` 分隔键值）
```bash
python main.py --rl-dir ./rl_io --encoder /path/to/qav1enc \
  --videos "--input|/data/in.yuv|--input-res|1920x1080|--frames|0|--o|./out.ivf|--csv|./out.csv|--bitrate|2125|--pass|2|--stat-in|./p1.log|--stat-out|./p2.log|--fps|24|--preset|1|--rc-mode|1" \
  --epochs 20 --ckpt-interval 5
```

### 数据集模式
```bash
python main.py --rl-dir ./rl_io --encoder /path/to/qav1enc \
  --use-dataset --dataset-inputs "/dataset/*.yuv" --stat-dir ./1pass_logs --out-dir ./outputs \
  --epochs 50 --log-level 1
```

## 命令行参数

### 训练控制
- `--epochs N`：训练的 epoch 数量（默认：2）
- `--start-epoch N`：起始 epoch（默认：1）
- `--mode {train,infer}`：训练或推理模式（默认：train）

### 日志控制
- `--log-level {0,1,2,3}`：日志级别（0=静默, 1=重要, 2=详细, 3=调试）
- `--no-tensorboard`：禁用 TensorBoard

### Checkpoint
- `--ckpt-dir DIR`：检查点保存目录（默认：./checkpoints）
- `--ckpt-interval N`：每 N 个 epoch 保存一次（默认：5）
- `--load-checkpoint PATH`：加载检查点继续训练
- `--save-replay-buffer`：同时保存 Replay Buffer

### 其他
- `--device {cpu,cuda,cuda:0}`：训练设备
- `--baseline-stats PATH`：基线统计文件路径

完整参数列表：`python main.py --help`

## 依赖
- Python 3.9+
- PyTorch >= 2.0
- NumPy
- TensorBoard（可选，用于可视化）

```bash
# 安装核心依赖
pip install torch torchvision numpy

# 安装可选依赖（TensorBoard）
pip install tensorboard
```

## 项目结构
```
SAC/
├── main.py              # 主入口
├── config.py            # 配置文件（新增日志和 checkpoint 配置）
├── sac_agent.py         # SAC 算法（新增 checkpoint 方法）
├── io_runner.py         # RL 循环（新增日志控制和 TensorBoard）
├── models.py            # 神经网络模型
├── replay.py            # Replay Buffer
├── reward.py            # 奖励函数
├── state.py             # 状态构建
├── encoder_proc.py      # 编码器进程管理
├── checkpoints/         # 检查点保存目录（自动创建）
├── runs/                # TensorBoard 日志（自动创建）
└── FEATURES.md          # 新功能详细文档
```
