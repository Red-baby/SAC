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
            for line in iter(proc.stdout.readline, ""):
                if not line: break
                if to_console:
                    try: sys.stdout.write(line)
                    except Exception: pass
                if log_fp:
                    try: log_fp.write(line)
                    except Exception: pass
        finally:
            try: proc.stdout.close()
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
        while enc.poll() is None:
            time.sleep(0.2)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            has_rq = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_rq.json")))
            has_fb = bool(glob.glob(os.path.join(cfg.rl_dir, "mg????_fb.json")))
            in_flight = len(getattr(runner, "pending", {})) > 0
            if not has_rq and not has_fb and not in_flight:
                break
            time.sleep(0.1)

        stop_evt.set()
        print("[ENC] finished; stop signal sent to RL loop.")

    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()
    return t
