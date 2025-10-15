# RL Agent for MiniGOP ΔQP (SAC v2)

- 进程交互：编码器与 RL 通过目录 `rl_io/` 文件握手
  - 编码器写：`mg%04d_rq.json`（请求）、`mg%04d_fb.json`（反馈）
  - RL 写：`mg%04d_qp.json`（QP 决策，JSON: {"qp": N}）
- 终止：`fb.gop_end==1` 表示一个 episode（GOP）结束，RL 在 GOP 末更新约束拉格朗日系数 λ。

## 运行

### 单视频命令（用 `|` 分隔键值）
```bash
python -m rl_agent.main --rl-dir ./rl_io --encoder /path/to/qav1enc \
  --videos "--input|/data/in.yuv|--input-res|1920x1080|--frames|0|--o|./out.ivf|--csv|./out.csv|--bitrate|2125|--pass|2|--stat-in|./p1.log|--stat-out|./p2.log|--fps|24|--preset|1|--rc-mode|1"
```

### 数据集模式
```bash
python -m rl_agent.main --rl-dir ./rl_io --encoder /path/to/qav1enc \
  --use-dataset --dataset-inputs "/dataset/*.yuv" --stat-dir ./1pass_logs --out-dir ./outputs
```

## 依赖
- Python 3.9+
- PyTorch >= 2.0
