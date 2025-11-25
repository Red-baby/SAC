# -*- coding: utf-8 -*-
import os, argparse, threading
from typing import List, Dict, Tuple
from config import Config
from io_runner import RLRunner
from encoder_proc import launch_encoder, start_monitor
from dataset import add_dataset_args, build_cmds_from_dataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", type=str, default=Config.rl_dir)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--mode", type=str, default="train", choices=["train","infer"])
    ap.add_argument("--encoder", type=str, default=Config.encoder_path)
    ap.add_argument("--device", type=str, default=None,
                    help="PyTorch 设备，例如 cpu、cuda、cuda:0；默认自动检测")
    ap.add_argument("--baseline-stats", type=str, default="./encoder/raw2pass.log",
                    help="基线帧统计日志路径，用于比较 bits/score")
    ap.add_argument("--fps", type=float, default=None,
                    help="视频帧率（用于计算 kbps），如果未指定则从编码器命令中提取")
    
    # Checkpoint 参数
    ap.add_argument("--ckpt-dir", type=str, default=Config.ckpt_dir,
                    help="检查点保存目录")
    ap.add_argument("--ckpt-interval", type=int, default=Config.ckpt_interval,
                    help="每 N 个 epoch 保存一次检查点")
    ap.add_argument("--load-checkpoint", type=str, default=None,
                    help="加载检查点路径（完整模型状态）")
    ap.add_argument("--save-replay-buffer", action="store_true", default=Config.save_replay_buffer,
                    help="是否保存 replay buffer")
    
    # 日志参数
    ap.add_argument("--log-level", type=int, default=Config.log_level,
                    choices=[0, 1, 2, 3],
                    help="日志级别：0=静默, 1=重要, 2=详细, 3=调试")
    ap.add_argument("--no-tensorboard", action="store_true",
                    help="禁用 TensorBoard")

    # 切换：单视频/数据集
    ap.add_argument("--use-dataset", action="store_true", help="启用数据集模式（从 --dataset-inputs 自动发现 YUV）")

    # 单视频命令模式（每条内部用 | 分隔）
    ap.add_argument("--videos", type=str, nargs="+", default=[
        "--input|E:/ftp/bhutan_1920x1080_25.yuv"
        "|--input-res|1920x1080|--frames|0"
        "|--o|ac_origins_test_cqp.ivf"
        "|--csv|ac_test_cqp.csv"
        "|--rc-mode|1|--pass|2"
        "|--stat-in|1pass.log"
        "|--stat-out|2pass.log"
        "|--score-max|50.5|--score-avg|40.5|--score-min|38.5"
        "|--fps|25|--preset|1|--keyint|225|--bframes|15|--threads|1|--parallel-frames|1"
        "|--bitrate|2125"
    ])

    add_dataset_args(ap)
    return ap.parse_args()

def _split_cmd_bar(cmd_str: str) -> List[str]:
    return [p for p in cmd_str.split("|") if p]

def _extract_fps_from_argv(argv: List[str]) -> float:
    """从编码器命令参数中提取 fps"""
    try:
        if "--fps" in argv:
            idx = argv.index("--fps")
            if idx + 1 < len(argv):
                return float(argv[idx + 1])
    except (ValueError, IndexError):
        pass
    return 25.0  # 默认值

def run_one(runner: RLRunner, cfg: Config, argv: List[str], epoch_id: int, epoch_total: int):
    # track epoch for encoder logs
    cfg.curr_epoch = epoch_id
    if hasattr(runner, "set_epoch"):
        runner.set_epoch(epoch_id)
    
    # 从当前视频命令中提取 fps（支持多视频不同 fps）
    current_fps = _extract_fps_from_argv(argv)
    if hasattr(runner, "set_current_fps"):
        runner.set_current_fps(current_fps)
        print(f"[MAIN] 当前视频 FPS: {current_fps}")
    
    # 清理 rl_io 目录中的残留文件
    print(f"[MAIN] 清理 rl_io 目录...")
    _cleanup_rl_dir(cfg.rl_dir)
    
    stop_evt = threading.Event()
    try:
        # launch encoder
        proc = launch_encoder(cfg, argv)
        # start monitor
        _ = start_monitor(proc, cfg, runner, stop_evt)
        # serve loop
        runner.serve_loop(stop_evt)
        # 打印 epoch 总结
        runner.print_epoch_summary(epoch_id, epoch_total)
    except KeyboardInterrupt:
        print(f"\n[MAIN] 捕获到键盘中断信号，正在优雅退出...")
        stop_evt.set()
        # 打印当前统计
        runner.print_epoch_summary(epoch_id, epoch_total, interrupted=True)
        raise  # 重新抛出以终止程序

def _cleanup_rl_dir(rl_dir: str):
    """清理 rl_io 目录中的残留 JSON 文件"""
    import glob
    patterns = ["mg????_rq.json", "mg????_qp.json", "mg????_fb.json"]
    removed_count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(rl_dir, pattern))
        for f in files:
            try:
                os.remove(f)
                removed_count += 1
            except Exception as e:
                print(f"[MAIN][WARN] 无法删除 {f}: {e}")
    if removed_count > 0:
        print(f"[MAIN] 已清理 {removed_count} 个残留文件")

def save_full_checkpoint(runner: RLRunner, epoch_id: int, cfg: Config):
    """保存完整的训练状态（模型 + Replay Buffer + 其他状态）"""
    import torch
    ckpt_path = os.path.join(cfg.ckpt_dir, f"checkpoint_epoch_{epoch_id}.pt")
    
    # 保存模型
    if runner.agent:
        runner.agent.save_checkpoint(ckpt_path)
        
        # 保存 replay buffer 和其他训练状态
        if cfg.save_replay_buffer:
            extra_path = os.path.join(cfg.ckpt_dir, f"extra_epoch_{epoch_id}.pt")
            extra_state = {
                'total_steps': runner.total_steps,
                'epoch_id': epoch_id,
                'lambda': runner.rw.lam,
                'score_ema': runner.rw.score_ema.val,
            }
            # 保存 replay buffer（如果存在且不为空）
            if runner.buf and len(runner.buf) > 0:
                extra_state['replay_buffer'] = runner.buf.export_state()
            torch.save(extra_state, extra_path)
            print(f"[Checkpoint] 已保存训练状态 -> {extra_path}")

def load_full_checkpoint(runner: RLRunner, ckpt_path: str):
    """加载完整的训练状态"""
    import torch
    
    # 等待模型初始化（需要第一个 RQ 到达后才能确定维度）
    # 这里只加载基本检查点路径，实际加载在 _ensure_models 后
    runner._pending_checkpoint_load = ckpt_path
    print(f"[Checkpoint] 标记待加载检查点: {ckpt_path}")
    
    # 尝试加载额外状态（如果存在）
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    extra_name = ckpt_name.replace("checkpoint_", "extra_")
    extra_path = os.path.join(ckpt_dir, extra_name)
    
    if os.path.exists(extra_path):
        try:
            extra_state = torch.load(extra_path, map_location=runner.cfg.device)
            runner.total_steps = extra_state.get('total_steps', 0)
            runner.rw.lam = extra_state.get('lambda', runner.cfg.lambda_init)
            if 'score_ema' in extra_state and extra_state['score_ema'] is not None:
                runner.rw.score_ema.val = extra_state['score_ema']
            print(f"[Checkpoint] 已加载训练状态: total_steps={runner.total_steps}, lambda={runner.rw.lam:.6f}")
            
            # 恢复 replay buffer（需要在模型初始化后）
            if 'replay_buffer' in extra_state:
                runner._pending_replay_buffer = extra_state['replay_buffer']
                print(f"[Checkpoint] 标记待恢复 Replay Buffer: size={extra_state['replay_buffer']['size']}")
        except Exception as e:
            print(f"[Checkpoint][WARN] 加载训练状态失败: {e}")

def main():
    args = parse_args()
    import torch
    import os
    if args.device is not None:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[MAIN][WARN] requested device '{device}' unavailable; fallback to CPU")
            device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_dir_abs = os.path.abspath(args.rl_dir)
    baseline_path = args.baseline_stats
    if baseline_path is not None:
        baseline_path = os.path.abspath(baseline_path)
    
    # 提取 fps（如果用户指定了就使用，否则从第一个视频命令中提取）
    fps = args.fps
    if fps is None:
        if bool(args.use_dataset):
            # 数据集模式：稍后从每个命令中提取
            fps = 25.0  # 默认值
        else:
            # 单视频模式：从第一个视频命令中提取
            if args.videos:
                first_argv = _split_cmd_bar(args.videos[0])
                fps = _extract_fps_from_argv(first_argv)
            else:
                fps = 25.0
    
    # 更新 Config
    cfg = Config(
        rl_dir=rl_dir_abs, 
        device=device, 
        baseline_stats_path=baseline_path,
        fps=fps,
        ckpt_dir=args.ckpt_dir,
        ckpt_interval=args.ckpt_interval,
        load_checkpoint=args.load_checkpoint,
        save_replay_buffer=args.save_replay_buffer,
        log_level=args.log_level,
        use_tensorboard=not args.no_tensorboard,
    )
    
    if args.encoder:
        cfg.encoder_path = os.path.abspath(args.encoder)
    else:
        cfg.encoder_path = os.path.abspath(cfg.encoder_path)

    os.makedirs(cfg.rl_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    runner = RLRunner(cfg)
    
    # 加载 checkpoint（如果指定）
    if cfg.load_checkpoint and os.path.exists(cfg.load_checkpoint):
        print(f"[MAIN] 加载 checkpoint: {cfg.load_checkpoint}")
        load_full_checkpoint(runner, cfg.load_checkpoint)
    elif cfg.load_checkpoint:
        print(f"[MAIN][WARN] checkpoint 文件不存在: {cfg.load_checkpoint}")

    start_ep = int(max(1, args.start_epoch))
    end_ep   = start_ep + int(max(1, args.epochs)) - 1

    if bool(args.use_dataset):
        cmds = build_cmds_from_dataset(args, cfg)
        if not cmds:
            print("[MAIN] no dataset commands built; exit."); return
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            for cmd_argv in cmds:
                run_one(runner, cfg, cmd_argv, epoch_id=eid, epoch_total=len(cmds))
                eid += 1
            # 每隔 N 个 epoch 保存一次
            if cfg.ckpt_interval > 0 and (ep % cfg.ckpt_interval) == 0:
                save_full_checkpoint(runner, ep, cfg)
        print("[MAIN] dataset mode finished.")
        # 保存最终模型
        save_full_checkpoint(runner, end_ep, cfg)
    else:
        epoch_total = (end_ep - start_ep + 1) * max(1, len(args.videos))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            for cmd_bar in args.videos:
                argv = _split_cmd_bar(cmd_bar)
                run_one(runner, cfg, argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
            # 每隔 N 个 epoch 保存一次
            if cfg.ckpt_interval > 0 and (ep % cfg.ckpt_interval) == 0:
                save_full_checkpoint(runner, ep, cfg)
        print("[MAIN] single-video list finished.")
        # 保存最终模型
        save_full_checkpoint(runner, end_ep, cfg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] 程序被用户中断，已退出。")
        import sys
        sys.exit(0)
