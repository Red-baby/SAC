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
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--start-epoch", type=int, default=1)
    ap.add_argument("--mode", type=str, default="train", choices=["train","infer"])
    ap.add_argument("--encoder", type=str, default=Config.encoder_path)
    ap.add_argument("--device", type=str, default=None,
                    help="PyTorch 设备，例如 cpu、cuda、cuda:0；默认自动检测")
    ap.add_argument("--baseline-stats", type=str, default="./encoder/raw2pass.log",
                    help="基线帧统计日志路径，用于比较 bits/score")

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

def run_one(runner: RLRunner, cfg: Config, argv: List[str], epoch_id: int, epoch_total: int):
    # track epoch for encoder logs
    cfg.curr_epoch = epoch_id
    # launch encoder
    proc = launch_encoder(cfg, argv)
    # start monitor
    stop_evt = threading.Event()
    _ = start_monitor(proc, cfg, runner, stop_evt)
    # serve loop
    runner.serve_loop(stop_evt)

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
    cfg = Config(rl_dir=rl_dir_abs, device=device, baseline_stats_path=baseline_path)
    if args.encoder:
        cfg.encoder_path = os.path.abspath(args.encoder)
    else:
        cfg.encoder_path = os.path.abspath(cfg.encoder_path)

    os.makedirs(cfg.rl_dir, exist_ok=True)
    runner = RLRunner(cfg)

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
        print("[MAIN] dataset mode finished.")
    else:
        epoch_total = (end_ep - start_ep + 1) * max(1, len(args.videos))
        eid = start_ep
        for ep in range(start_ep, end_ep + 1):
            for cmd_bar in args.videos:
                argv = _split_cmd_bar(cmd_bar)
                run_one(runner, cfg, argv, epoch_id=eid, epoch_total=epoch_total)
                eid += 1
        print("[MAIN] single-video list finished.")

if __name__ == "__main__":
    main()
