# -*- coding: utf-8 -*-
import os, sys, glob, time, threading, subprocess, datetime
from utils import now_ms

def _win_no_window_flags(cfg):
    if os.name != "nt":
        return {}
    flags = {}
    if bool(getattr(cfg, "hide_encoder_console_window", False)):
        flags["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        import msvcrt  # noqa
        flags["close_fds"] = False
    except Exception:
        pass
    return flags

def _build_log_path(cfg):
    base_dir = str(getattr(cfg, "encoder_log_dir", "./logs/encoder"))
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ep = getattr(cfg, "curr_epoch", None) or getattr(cfg, "epoch_idx", None)
    tag = f"_ep{int(ep)}" if ep is not None else ""
    return os.path.join(base_dir, f"encoder_{ts}{tag}.log")

def launch_encoder(cfg, video_args: list[str]):
    enc_exec = os.path.abspath(cfg.encoder_path)
    enc_dir = os.path.dirname(enc_exec)
    cmd = [enc_exec] + list(video_args)

    env = os.environ.copy()
    rl_dir_abs = os.path.abspath(getattr(cfg, "rl_dir", "./rl_io"))
    try:
        rl_dir_rel = os.path.relpath(rl_dir_abs, enc_dir)
    except Exception:
        rl_dir_rel = rl_dir_abs
    env["QAV1_RL_DIR"] = rl_dir_rel

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
        cwd=enc_dir,
        **_win_no_window_flags(cfg)
    )

    log_path = None
    log_fp = None
    to_file = bool(getattr(cfg, "encoder_log_to_file", True))
    to_console = bool(getattr(cfg, "show_encoder_output", True))

    if to_file:
        log_path = _build_log_path(cfg)
        try: cfg.last_encoder_log_path = log_path
        except Exception: pass
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

        if log_fp:
            log_fp.write(f"[ENC] launch: {' '.join(cmd)}\n")
            log_fp.write(f"[ENC] rl_dir={env.get('QAV1_RL_DIR','')}\n")

    def _tee():
        try:
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, ""):
                    if not line: break
                    if to_console:
                        try: sys.stdout.write(line)
                        except Exception: pass
                    if log_fp:
                        try: log_fp.write(line)
                        except Exception: pass
        finally:
            try: 
                if proc.stdout is not None:
                    proc.stdout.close()
            except Exception: pass
            if log_fp:
                try: log_fp.flush(); log_fp.close()
                except Exception: pass

    t = threading.Thread(target=_tee, daemon=True)
    t.start()

    print(f"[ENC] started | pid={proc.pid} | log={'on' if to_file else 'off'}" + (f" -> {log_path}" if to_file else ""))
    return proc

def start_monitor(enc: subprocess.Popen, cfg, runner, stop_evt: threading.Event):
    def monitor_thread():
        print(f"[ENC][MONITOR] started monitoring encoder process (pid={enc.pid})")
        # 当编码器进程仍在运行时，poll()返回None
        while enc.poll() is None:
            time.sleep(0.2)

        # 获取编码器的退出码
        return_code = enc.returncode
        print(f"\n[ENC][MONITOR] ===== 编码器进程已退出 =====")
        print(f"[ENC][MONITOR] 退出码: {return_code}")

        # 给 RL 循环一些时间处理最后的文件
        deadline = time.time() + 5.0  # 5 秒
        print(f"[ENC][MONITOR] 等待 RL 处理剩余的请求和反馈...")
        all_clear = False
        last_print_time = 0
        
        while time.time() < deadline:
            # 检查是否还有未处理的请求或反馈文件
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_fb.json")))
            has_qp = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_qp.json")))
            in_flight = len(getattr(runner, "pending", {})) > 0
            
            if not has_rq and not has_fb and not has_qp and not in_flight:
                print("[ENC][MONITOR] ✓ 所有请求和反馈已处理完毕")
                all_clear = True
                break
            
            # 每秒打印一次状态
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                status_parts = []
                if has_rq:
                    status_parts.append(f"待处理RQ")
                if has_fb:
                    status_parts.append(f"待处理FB")
                if has_qp:
                    status_parts.append(f"待消费QP")
                if in_flight:
                    status_parts.append(f"{in_flight}个MG等待FB")
                
                status_str = ", ".join(status_parts) if status_parts else "无"
                print(f"[ENC][MONITOR] 等待中... ({status_str})")
                last_print_time = current_time
            
            time.sleep(0.1)
        
        if not all_clear:
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_fb.json")))
            in_flight = len(getattr(runner, "pending", {})) > 0
            
            print(f"[ENC][MONITOR] ⚠ 等待超时，仍有未完成项")
            if has_rq:
                rq_files = glob.glob(os.path.join(cfg.rl_dir, "mg????_rq.json"))
                print(f"[ENC][MONITOR]   - {len(rq_files)} 个未处理的 RQ: {[os.path.basename(f) for f in rq_files]}")
            if has_fb:
                fb_files = glob.glob(os.path.join(cfg.rl_dir, "mg????_fb.json"))
                print(f"[ENC][MONITOR]   - {len(fb_files)} 个未处理的 FB: {[os.path.basename(f) for f in fb_files]}")
            if in_flight:
                pending_ids = list(getattr(runner, "pending", {}).keys())
                print(f"[ENC][MONITOR]   - {len(pending_ids)} 个 MG 等待 FB: {pending_ids}")
                print(f"[ENC][MONITOR]   原因: 编码器退出前未发送这些 MG 的反馈（可能异常退出）")

        # 发送停止信号给RL循环
        stop_evt.set()
        print("[ENC][MONITOR] stop signal sent to RL loop.")

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return t