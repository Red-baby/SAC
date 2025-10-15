# -*- coding: utf-8 -*-
"""
MiniGOP I/O runner:
- Watch rl_dir for mg????_rq.json / mg????_fb.json (encoder handshake via rl_sync.*)
- Build a [C=5, T] feature block via state.build_state_from_rq
- Actor outputs ΔQP (scalar per MG). We clamp baseqp+ΔQP to [qp_min, qp_max] and write mg????_qp.json
- Reward: per-MG via reward.RewardComputer.step; episode ends when fb.gop_end == 1
"""
import os, glob, time, json, numpy as np, torch
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from config import Config
from utils import safe_read_json, safe_write_json_atomic, now_ms
from sac_agent import SACAgent
from replay import ReplayBuffer
from state import build_state_from_rq
from reward import RewardComputer, RewardCfg

class BaselineStats:
    def __init__(self, path: str):
        self.path = path
        self.frames: List[Dict] = []
        self._poc_to_idx: Dict[int, int] = {}
        self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                frame = self._parse_line(line)
                if frame is None:
                    continue
                idx = len(self.frames)
                self.frames.append(frame)
                self._poc_to_idx[frame["poc"]] = idx

    @staticmethod
    def _parse_line(line: str) -> Optional[Dict]:
        parts = line.split()
        if len(parts) < 3:
            return None
        try:
            poc = int(parts[0])
        except ValueError:
            return None
        frame_type = parts[2].upper()
        def _find_float(key: str, default: float = 0.0) -> float:
            if key not in parts:
                return default
            idx = parts.index(key)
            if idx + 1 >= len(parts):
                return default
            try:
                return float(parts[idx + 1])
            except ValueError:
                return default
        score = _find_float("score", 0.0)
        bits = _find_float("bits", 0.0)
        return {"poc": poc, "type": frame_type, "score": score, "bits": bits}

    def accumulate_minigop(self, last_poc: int) -> Tuple[float, float]:
        if not self.frames:
            raise RuntimeError("baseline stats is empty")
        if last_poc not in self._poc_to_idx:
            raise KeyError(f"baseline stats missing poc={last_poc}")
        idx = self._poc_to_idx[last_poc]
        sum_bits = 0.0
        sum_score = 0.0
        found_p = False
        while idx >= 0:
            frame = self.frames[idx]
            sum_bits += float(frame["bits"])
            sum_score += float(frame["score"])
            if frame["type"] == "P":
                found_p = True
                break
            idx -= 1
        if not found_p:
            raise RuntimeError(f"no P-frame found when accumulating for poc={last_poc}")
        return sum_bits, sum_score

def _scan_mg_rq_files(rl_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(rl_dir, "mg????_rq.json")))

def _scan_mg_fb_files(rl_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(rl_dir, "mg????_fb.json")))

class RLRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Probe state dims from a dummy rq if exists later
        self.agent: Optional[SACAgent] = None
        self.buf: Optional[ReplayBuffer] = None

        # reward computer
        self.rw = RewardComputer(RewardCfg(
            gamma=cfg.gamma,
            smooth_penalty=cfg.smooth_penalty,
            lambda_init=cfg.lambda_init,
            lambda_lr=cfg.lambda_lr,
            shaping_w_score_ema=cfg.shaping_w_score_ema,
            term_bonus=cfg.term_bonus,
            term_tau=cfg.term_tau,
        ))

        self.pending: Dict[int, Dict] = {}
        self._last_mg_id: Optional[int] = None

        self.total_steps = 0
        self.baseline: Optional[BaselineStats] = None
        self._baseline_warn_count = 0
        self._mg_seen = 0
        baseline_path = getattr(cfg, "baseline_stats_path", None)
        if baseline_path:
            try:
                self.baseline = BaselineStats(baseline_path)
                print(f"[Baseline] loaded {len(self.baseline.frames)} frames from {baseline_path}")
            except Exception as e:
                print(f"[Baseline][WARN] failed to load '{baseline_path}': {e}")
                self.baseline = None

    def _ensure_models(self, seq_shape: Tuple[int,int], scalar_dim: int):
        if self.agent is not None:
            return
        C,T = seq_shape
        self.agent = SACAgent(self.cfg, state_scalar_dim=scalar_dim, seq_T=T, seq_C=C)
        self.buf = ReplayBuffer(self.cfg.replay_size, (C,T), scalar_dim)
        print(f"[RL] Models ready. State(seq)={C}x{T}, scalars={scalar_dim}")

    def serve_loop(self, stop_evt) -> None:
        rl_dir = self.cfg.rl_dir
        print(f"[Run] RL loop started. rl_dir={rl_dir}")

        # Wait until any rq arrives (or stop)
        wait_ms = 0
        while not stop_evt.is_set():
            rq_files = _scan_mg_rq_files(rl_dir)
            if rq_files: break
            wait_ms += self.cfg.poll_ms
            if wait_ms % 1000 == 0:
                pending_fb = len(_scan_mg_fb_files(rl_dir))
                print(f"[MG][WAIT] no rq yet (waited {wait_ms/1000:.1f}s) pending_fb={pending_fb}")
            if wait_ms in (10000, 20000):
                print(f"[MG][WAIT] still waiting for first rq after {wait_ms/1000:.0f}s …")
            time.sleep(self.cfg.poll_ms/1000.0)

        processed_rq = set()
        rq_read_failures: Dict[str, int] = defaultdict(int)
        idle_loops = 0
        while not stop_evt.is_set():
            progressed = False

            # Handle RQ
            for rq_path in [p for p in _scan_mg_rq_files(rl_dir) if p not in processed_rq]:
                try:
                    rq = safe_read_json(rq_path)
                    if rq_path in rq_read_failures:
                        rq_read_failures.pop(rq_path, None)
                except Exception as e:
                    rq_read_failures[rq_path] += 1
                    fail_cnt = rq_read_failures[rq_path]
                    if fail_cnt <= 3 or (fail_cnt % 10) == 0:
                        print(f"[RL][WARN] bad rq (retry #{fail_cnt}): {rq_path}: {e}")
                    continue

                # Build state
                g_state = dict(
                    score_ema = self.rw.score_ema.get(),
                    last_delta = getattr(self, "_last_delta", 0.0),
                )
                seq, scalars, baseqp, mg_id, mg_size, bits_alloc, score_alloc = build_state_from_rq(self.cfg, rq, g_state)
                if self.agent is None:
                    self._ensure_models(seq.shape, scalars.shape[0])

                self._mg_seen = max(self._mg_seen, mg_id + 1)
                print(f"[MG][RQ] path={rq_path} id={mg_id} size={mg_size} base_qp={baseqp} pending={len(self.pending)} total_mg_seen={self._mg_seen}")

                # Action
                seq1 = torch.from_numpy(seq).unsqueeze(0).to(self.cfg.device).float()
                sca1 = torch.from_numpy(scalars).unsqueeze(0).to(self.cfg.device).float()
                if self.total_steps < self.cfg.start_steps:
                    a_norm = float(np.random.uniform(-1, 1))
                    act_src = "explore"
                else:
                    a_t, _ = self.agent.act(seq1, sca1, deterministic=False)
                    a_norm = float(a_t.squeeze().detach().cpu().numpy())
                    act_src = "policy"
                delta_qp = int(round(a_norm * self.cfg.delta_qp_max))
                qp_new = int(np.clip(baseqp + delta_qp, self.cfg.qp_min, self.cfg.qp_max))
                print(f"[MG][ACT] id={mg_id} src={act_src} delta_qp={delta_qp} -> qp={qp_new} (base={baseqp})")

                # Write QP json for this mg
                qp_path = rq_path.replace("_rq.json", "_qp.json")
                safe_write_json_atomic(qp_path, {"qp": int(qp_new)})
                print(f"[MG][QP] wrote -> {qp_path} (exists={os.path.exists(qp_path)})")

                # Stash pending (we'll fill next_state upon next rq)
                self.pending[mg_id] = dict(
                    seq=seq, scalars=scalars, a=np.array([a_norm], dtype=np.float32),
                    delta_qp=delta_qp, bits_alloc=bits_alloc, score_alloc=score_alloc
                )
                if self._last_mg_id is not None and self._last_mg_id in self.pending:
                    prev = self.pending[self._last_mg_id]
                    prev["next_seq"] = seq; prev["next_scalars"] = scalars
                self._last_mg_id = mg_id
                processed_rq.add(rq_path)
                progressed = True

            # Handle FB
            for fb_path in _scan_mg_fb_files(rl_dir):
                try:
                    fb = safe_read_json(fb_path)
                except Exception as e:
                    print(f"[RL][WARN] bad fb: {fb_path}: {e}")
                    try:
                        os.remove(fb_path)
                    except Exception:
                        pass
                    continue

                mg_id = int(fb.get("mg_id", -1))
                if mg_id not in self.pending:
                    try: os.remove(fb_path)
                    except Exception: pass
                    continue

                pend = self.pending.pop(mg_id)
                bits = float(fb.get("bits", 0.0))
                score = float(fb.get("score", 0.0))
                bits_alloc = float(pend.get("bits_alloc", 0.0))
                score_alloc = float(pend.get("score_alloc", 0.0))
                if self.baseline is not None:
                    last_poc = fb.get("last_poc", None)
                    if last_poc is not None:
                        try:
                            b_bits, b_score = self.baseline.accumulate_minigop(int(last_poc))
                            bits_alloc = float(b_bits)
                            score_alloc = float(b_score)
                        except Exception as e:
                            self._baseline_warn_count += 1
                            if self._baseline_warn_count <= 3 or (self._baseline_warn_count % 10) == 0:
                                print(f"[Baseline][WARN] cannot accumulate for last_poc={last_poc}: {e}")
                    else:
                        self._baseline_warn_count += 1
                        if self._baseline_warn_count <= 3 or (self._baseline_warn_count % 10) == 0:
                            print("[Baseline][WARN] fb missing last_poc; fallback to rq alloc values.")
                gop_end = int(fb.get("gop_end", 0)) == 1

                # Reward step
                r = self.rw.step(bits=bits, score=score,
                                 bits_alloc=bits_alloc, score_alloc=score_alloc,
                                 delta_qp=pend["delta_qp"])
                print(f"[MG][FB] path={fb_path} id={mg_id} baseline_bits={bits_alloc:.1f} baseline_score={score_alloc:.3f} "
                      f"| actual_bits={bits:.1f} actual_score={score:.3f} reward={r:.4f}")

                # Replay push
                seq = pend["seq"]; sca = pend["scalars"]
                a = pend["a"]; done = gop_end
                if "next_seq" in pend:
                    seq2 = pend["next_seq"]; sca2 = pend["next_scalars"]
                else:
                    seq2 = np.zeros_like(seq); sca2 = np.zeros_like(sca)
                self.buf.push(seq, sca, a, r, seq2, sca2, done)

                # Train
                self.total_steps += 1
                if self.total_steps >= self.cfg.start_steps and len(self.buf) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        b = self.buf.sample(self.cfg.batch_size, self.cfg.device)
                        loss_q, loss_actor, alpha = self.agent.train_step(b)
                        if (self.total_steps % 50) == 0:
                            print(f"[Train] step={self.total_steps} Lq={loss_q:.4f} La={loss_actor:.4f} alpha={alpha:.4f}")

                self._last_delta = float(pend["delta_qp"])

                # Episode end?
                if gop_end:
                    info = self.rw.on_gop_end()
                    print(f"[GOP END][FB] path={fb_path} gop_steps={info['steps']} reward_ep={info['episode_return']:.4f}")
                    self._last_mg_id = None
                    print(f"[GOP END] steps={info['steps']} "
                          f"Σbits={info['sum_bits']:.1f} (alloc={info['sum_bits_alloc']:.1f}, Δnorm={info['delta_bits_norm']:.5f}) "
                          f"Σscore={info['sum_score']:.3f} (alloc={info['sum_score_alloc']:.3f}, Δnorm={info['delta_score_norm']:.5f}) "
                          f"λ={info['lambda']:.6f} term={info['term_bonus']:.4f} R_ep={info['episode_return']:.4f}")

                try: os.remove(fb_path)
                except Exception: pass

                progressed = True

            if not progressed:
                idle_loops += 1
                if idle_loops * self.cfg.poll_ms >= 1000:
                    print(f"[MG][WAIT] idle loop (pending={len(self.pending)}, last_mg={self._last_mg_id})")
                    idle_loops = 0
                time.sleep(self.cfg.poll_ms/1000.0)
            else:
                idle_loops = 0
