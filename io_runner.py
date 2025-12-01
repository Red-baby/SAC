# -*- coding: utf-8 -*-
"""
MiniGOP I/O runner:
- Watch rl_dir for mg????_rq.json / mg????_fb.json (encoder handshake via rl_sync.*)
- Build a [C=6, T] feature block via state.build_state_from_rq (channels: poise, comp, rdcost, score_target, bit_target, q_val/256)
- Actor outputs ΔQP per frame (mg_size deltas). We apply each ΔQP to corresponding frame's q_val and write mg????_qp.json with {"q_vals": [...]}
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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


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

    def accumulate_minigop(self, last_poc: int) -> Tuple[float, float, int]:
        """返回 (sum_bits, sum_score, num_frames)"""
        if not self.frames:
            raise RuntimeError("baseline stats is empty")
        if last_poc not in self._poc_to_idx:
            raise KeyError(f"baseline stats missing poc={last_poc}")
        idx = self._poc_to_idx[last_poc]
        sum_bits = 0.0
        sum_score = 0.0
        num_frames = 0
        found_p = False
        while idx >= 0:
            frame = self.frames[idx]
            sum_bits += float(frame["bits"])
            sum_score += float(frame["score"])
            num_frames += 1
            if frame["type"] == "P":
                found_p = True
                break
            idx -= 1
        if not found_p:
            raise RuntimeError(f"no P-frame found when accumulating for poc={last_poc}")
        return sum_bits, sum_score, num_frames


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
        self.current_epoch: int = 1
        self.current_fps: float = cfg.fps  # 当前视频的 fps，支持多视频不同 fps

        # reward computer
        self.rw = RewardComputer(
            RewardCfg(
                gamma=cfg.gamma,
                smooth_penalty=cfg.smooth_penalty,
                lambda_init=cfg.lambda_init,
                lambda_lr=cfg.lambda_lr,
                shaping_w_score_ema=cfg.shaping_w_score_ema,
                term_bonus=cfg.term_bonus,
                term_tau=cfg.term_tau,
            )
        )

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
                self._log(1, f"[Baseline] loaded {len(self.baseline.frames)} frames from {baseline_path}")
            except Exception as e:
                self._log(1, f"[Baseline][WARN] failed to load '{baseline_path}': {e}")
                self.baseline = None
        
        # Epoch 统计
        self.epoch_episodes = 0  # 当前 epoch 完成的 episode 数量
        self.epoch_total_reward = 0.0  # 当前 epoch 累计奖励
        self.epoch_total_bits = 0.0  # 当前 epoch 累计码率（新）
        self.epoch_total_score = 0.0  # 当前 epoch 累计质量分数（新）
        self.epoch_total_bits_alloc = 0.0  # 当前 epoch 累计码率（原）
        self.epoch_total_score_alloc = 0.0  # 当前 epoch 累计质量分数（原）
        self.epoch_total_frames = 0  # 当前 epoch 累计帧数
        self.epoch_bits_saved = 0.0  # 当前 epoch 节省的码率（相对基线）
        self.epoch_score_gain = 0.0  # 当前 epoch 提升的质量（相对基线）
        self.epoch_train_count = 0  # 当前 epoch 训练次数
        
        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if cfg.use_tensorboard and TENSORBOARD_AVAILABLE:
            from datetime import datetime
            log_dir = os.path.join(cfg.tensorboard_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.writer = SummaryWriter(log_dir)
            self._log(1, f"[TensorBoard] 日志目录: {log_dir}")
        elif cfg.use_tensorboard and not TENSORBOARD_AVAILABLE:
            self._log(1, "[TensorBoard][WARN] tensorboard 未安装，请运行: pip install tensorboard")
        
        # Checkpoint 加载标记（等待模型初始化）
        self._pending_checkpoint_load: Optional[str] = None
        self._pending_replay_buffer: Optional[Dict] = None
        
        # FB 读取重试机制
        self.fb_read_failures: Dict[str, int] = defaultdict(int)
        self.fb_max_retries: int = 5  # 最多重试 5 次
    
    def _log(self, level: int, msg: str) -> None:
        """根据日志级别打印信息
        level: 0=静默, 1=重要, 2=详细, 3=调试
        """
        if level <= self.cfg.log_level:
            print(msg)
    
    def _cleanup_stale_tmp_files(self, rl_dir: str) -> None:
        """清理过期的 .tmp 文件（编码器写入失败的残留）"""
        tmp_files = glob.glob(os.path.join(rl_dir, "*.tmp"))
        current_time = time.time()
        for tmp_path in tmp_files:
            try:
                # 检查文件年龄
                mtime = os.path.getmtime(tmp_path)
                age = current_time - mtime
                if age > 5.0:  # 超过 5 秒认为是失败残留
                    os.remove(tmp_path)
                    self._log(2, f"[MG] 清理过期 .tmp 文件: {os.path.basename(tmp_path)} (age={age:.1f}s)")
            except Exception as e:
                self._log(3, f"[MG][WARN] 清理 .tmp 失败: {e}")

    def set_epoch(self, epoch_id: int) -> None:
        """更新当前 epoch 编号，便于日志打印。"""
        self.current_epoch = int(max(1, epoch_id))
    
    def set_current_fps(self, fps: float) -> None:
        """更新当前视频的 fps，用于计算 kbps（支持多视频不同 fps）。"""
        self.current_fps = float(max(1.0, fps))

    def _ensure_models(self, seq_shape: Tuple[int, int], scalar_dim: int):
        if self.agent is not None:
            return
        C, T = seq_shape
        self.agent = SACAgent(self.cfg, state_scalar_dim=scalar_dim, seq_T=T, seq_C=C)
        self.buf = ReplayBuffer(self.cfg.replay_size, (C, T), scalar_dim)
        self._log(1, f"[RL] Models ready. State(seq)={C}x{T}, scalars={scalar_dim}")
        
        # 加载 checkpoint（如果有待加载的）
        if self._pending_checkpoint_load:
            try:
                self.agent.load_checkpoint(self._pending_checkpoint_load)
                self._pending_checkpoint_load = None
            except Exception as e:
                self._log(1, f"[Checkpoint][ERROR] 加载失败: {e}")
        
        # 恢复 replay buffer
        if self._pending_replay_buffer and self.buf:
            try:
                rb = self._pending_replay_buffer
                self.buf.load_state(rb)
                self._log(1, f"[Checkpoint] 已恢复 Replay Buffer: size={len(self.buf)}")
                self._pending_replay_buffer = None
            except Exception as e:
                self._log(1, f"[Checkpoint][ERROR] 恢复 Replay Buffer 失败: {e}")

    def serve_loop(self, stop_evt) -> None:
        rl_dir = self.cfg.rl_dir
        self._log(1, f"[Run] RL loop started. rl_dir={rl_dir}")

        # Wait until any rq arrives (or stop)
        wait_ms = 0
        max_wait_ms = 30000  # 最多等待 30 秒
        while not stop_evt.is_set():
            rq_files = _scan_mg_rq_files(rl_dir)
            if rq_files:
                break
            wait_ms += self.cfg.poll_ms
            if wait_ms % 1000 == 0:
                pending_fb = len(_scan_mg_fb_files(rl_dir))
                self._log(3, f"[MG][WAIT] no rq yet (waited {wait_ms/1000:.1f}s) pending_fb={pending_fb}")
            if wait_ms >= max_wait_ms:
                self._log(2, f"[MG][WARN] 超过 {max_wait_ms/1000:.0f}s 未收到 rq，编码器可能未启动或已完成")
                return
            time.sleep(self.cfg.poll_ms / 1000.0)

        rq_read_failures: Dict[str, int] = defaultdict(int)
        idle_loops = 0
        consecutive_idle_count = 0  # 连续空闲计数
        max_consecutive_idle = 300  # 连续空闲 300 次（约 3 秒）后检查是否应该退出
        waiting_for_fb = False  # 标记是否正在等待 FB
        fb_wait_start_time = None  # FB 等待开始时间（用于显示等待时长）
        
        while not stop_evt.is_set():
            progressed = False

            # 如果正在等待 FB，优先处理 FB，不处理新的 RQ
            if not waiting_for_fb:
                # Handle RQ - 每次只处理一个 RQ
                rq_files = _scan_mg_rq_files(rl_dir)
                if rq_files:
                    rq_path = rq_files[0]  # 只取第一个
                    try:
                        rq = safe_read_json(rq_path)
                        if rq_path in rq_read_failures:
                            rq_read_failures.pop(rq_path, None)
                    except Exception as e:
                        rq_read_failures[rq_path] += 1
                        fail_cnt = rq_read_failures[rq_path]
                        if fail_cnt <= 3 or (fail_cnt % 10) == 0:
                            self._log(2, f"[RL][WARN] bad rq (retry #{fail_cnt}): {rq_path}: {e}")
                        time.sleep(self.cfg.poll_ms / 1000.0)
                        continue

                    # Build state
                    g_state = dict(
                        score_ema=self.rw.score_ema.get(),
                        last_delta=getattr(self, "_last_delta", 0.0),
                    )
                    seq, scalars, q_vals, mg_id, mg_size, bits_alloc, score_alloc = build_state_from_rq(
                        self.cfg, rq, g_state
                    )
                    if self.agent is None:
                        self._ensure_models(seq.shape, scalars.shape[0])

                    self._mg_seen = max(self._mg_seen, mg_id + 1)
                    avg_q_val = float(np.mean(q_vals)) if len(q_vals) > 0 else 0.0
                    self._log(2, f"[MG][RQ] ① 接收请求 -> {rq_path} | id={mg_id} size={mg_size} avg_q_val={avg_q_val:.2f}")

                    # Action
                    seq1 = torch.from_numpy(seq).unsqueeze(0).to(self.cfg.device).float()
                    sca1 = torch.from_numpy(scalars).unsqueeze(0).to(self.cfg.device).float()
                    if self.total_steps < self.cfg.start_steps:
                        # 探索：为每帧生成随机 delta
                        a_norm = np.random.uniform(-1, 1, size=(seq.shape[1],)).astype(np.float32)
                        act_src = "explore"
                    else:
                        a_t, _ = self.agent.act(seq1, sca1, deterministic=False)
                        # a_t: [1, seq_T]，提取为 numpy array
                        a_norm = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [seq_T]
                        act_src = "policy"
                    
                    # 提取前 mg_size 个 delta（因为 mg_size 可能小于 seq_T）
                    delta_norm = a_norm[:mg_size]  # [mg_size]
                    # 将归一化的 delta 转换为实际的 delta_qp
                    delta_qps = delta_norm * self.cfg.delta_qp_max  # [mg_size]
                    
                    # 为每一帧生成新的 q_val：q_val_new = clip(q_val_old + delta_qp, q_val_min, q_val_max)
                    # q_vals 长度已经是 mg_size
                    q_vals_new = np.clip(q_vals[:mg_size] + delta_qps, self.cfg.q_val_min, self.cfg.q_val_max).astype(np.float32)
                    
                    # 计算平均 q_val 和平均 delta 用于日志
                    avg_q_val_new = float(np.mean(q_vals_new))
                    avg_q_val_old = float(np.mean(q_vals[:mg_size]))
                    avg_delta_qp = float(np.mean(delta_qps))
                    self._log(2, f"[MG][ACT] ② 决策动作 -> id={mg_id} src={act_src} avg_delta_qp={avg_delta_qp:+.2f} avg_q_val={avg_q_val_old:.2f}->{avg_q_val_new:.2f}")

                    # Write QP json for this mg (新格式：q_vals 数组)
                    qp_path = rq_path.replace("_rq.json", "_qp.json")
                    q_vals_list = [float(q) for q in q_vals_new]
                    safe_write_json_atomic(qp_path, {"q_vals": q_vals_list})
                    self._log(3, f"[MG][QP] ③ 写入决策 -> {qp_path} (q_vals={len(q_vals_list)} frames)")

                    # Stash pending (we'll fill next_state upon next rq)
                    # 存储完整的 action（seq_T 维），用于 replay buffer
                    a_full = np.zeros(seq.shape[1], dtype=np.float32)  # [seq_T]
                    a_full[:mg_size] = a_norm[:mg_size]
                    self.pending[mg_id] = dict(
                        seq=seq,
                        scalars=scalars,
                        a=a_full,  # [seq_T] 维的 action
                        delta_qp=avg_delta_qp,  # 用于 reward 计算（保持标量）
                        bits_alloc=bits_alloc,
                        score_alloc=score_alloc,
                    )
                    if self._last_mg_id is not None and self._last_mg_id in self.pending:
                        prev = self.pending[self._last_mg_id]
                        prev["next_seq"] = seq
                        prev["next_scalars"] = scalars
                    self._last_mg_id = mg_id
                    
                    # 立即删除已处理的 RQ 文件
                    try:
                        os.remove(rq_path)
                        self._log(3, f"[MG][RQ] ④ 删除请求 -> {rq_path}")
                    except Exception as e:
                        self._log(2, f"[MG][WARN] 删除 RQ 失败: {e}")
                    
                    # 现在等待对应的 FB
                    waiting_for_fb = True
                    fb_wait_start_time = time.time()
                    self._log(3, f"[MG] >>> 等待 mg{mg_id:04d} 的反馈 (FB)...")
                    progressed = True

            # Handle FB
            for fb_path in _scan_mg_fb_files(rl_dir):
                try:
                    fb = safe_read_json(fb_path)
                    # 读取成功，清除失败记录
                    if fb_path in self.fb_read_failures:
                        self.fb_read_failures.pop(fb_path)
                except Exception as e:
                    # 读取失败，记录重试次数
                    self.fb_read_failures[fb_path] += 1
                    fail_cnt = self.fb_read_failures[fb_path]
                    
                    if fail_cnt < self.fb_max_retries:
                        # 还在重试范围内，不删除文件，等待下次循环
                        self._log(3, f"[MG][FB] read failed (retry {fail_cnt}/{self.fb_max_retries}): {fb_path}: {e}")
                        continue
                    else:
                        # 重试次数用尽，删除文件并清理对应的 pending
                        self._log(1, f"[MG][FB][ERROR] bad fb after {fail_cnt} retries, deleting: {fb_path}")
                        
                        # 尝试从文件名提取 mg_id 并清理 pending
                        try:
                            import re
                            match = re.search(r'mg(\d{4})_fb', fb_path)
                            if match:
                                bad_mg_id = int(match.group(1))
                                if bad_mg_id in self.pending:
                                    self.pending.pop(bad_mg_id)
                                    self._log(1, f"[MG][FB] 清理 bad fb 对应的 pending: mg_id={bad_mg_id}")
                        except Exception as clean_e:
                            self._log(2, f"[MG][WARN] failed to clean pending for bad fb: {clean_e}")
                        
                        # 删除坏文件并清除失败记录
                        try:
                            os.remove(fb_path)
                            self.fb_read_failures.pop(fb_path, None)
                            self._log(2, f"[MG][FB] deleted bad file -> {fb_path}")
                        except Exception as del_e:
                            self._log(2, f"[MG][WARN] failed to delete bad fb: {del_e}")
                        continue

                mg_id = int(fb.get("mg_id", -1))
                if mg_id not in self.pending:
                    self._log(2, f"[MG][WARN] fb for mg_id={mg_id} has no pending RQ, skipping and deleting")
                    try:
                        os.remove(fb_path)
                        print(f"[MG][FB] deleted orphan file -> {fb_path}")
                    except Exception as del_e:
                        self._log(2, f"[MG][WARN] failed to delete orphan fb: {del_e}")
                    continue

                pend = self.pending.pop(mg_id)
                bits = float(fb.get("bits", 0.0))
                score = float(fb.get("score", 0.0))
                bits_alloc = float(pend.get("bits_alloc", 0.0))
                score_alloc = float(pend.get("score_alloc", 0.0))
                num_frames = 0  # 该 mini-GOP 的帧数
                if self.baseline is not None:
                    last_poc = fb.get("last_poc", None)
                    if last_poc is not None:
                        try:
                            b_bits, b_score, n_frames = self.baseline.accumulate_minigop(int(last_poc))
                            bits_alloc = float(b_bits)
                            score_alloc = float(b_score)
                            num_frames = int(n_frames)
                        except Exception as e:
                            self._baseline_warn_count += 1
                            if self._baseline_warn_count <= 3 or (self._baseline_warn_count % 10) == 0:
                                self._log(1, f"[Baseline][WARN] cannot accumulate for last_poc={last_poc}: {e}")
                    else:
                        self._baseline_warn_count += 1
                        if self._baseline_warn_count <= 3 or (self._baseline_warn_count % 10) == 0:
                            print("[Baseline][WARN] fb missing last_poc; fallback to rq alloc values.")
                gop_end = int(fb.get("gop_end", 0)) == 1

                # Reward step
                r = self.rw.step(bits=bits, score=score, bits_alloc=bits_alloc, score_alloc=score_alloc, delta_qp=pend["delta_qp"], num_frames=num_frames)
                
                info = None
                if gop_end:
                    info = self.rw.on_gop_end()
                    r += info['term_bonus']

                print(
                    f"[MG][FB] ⑤ 接收反馈 -> {fb_path} | id={mg_id} "
                    f"bits={bits:.1f}(原{bits_alloc:.1f}) score={score:.3f}(原{score_alloc:.3f}) reward={r:.4f}"
                )

                # Replay push
                seq = pend["seq"]
                sca = pend["scalars"]
                a = pend["a"]
                done = gop_end
                if "next_seq" in pend:
                    seq2 = pend["next_seq"]
                    sca2 = pend["next_scalars"]
                else:
                    seq2 = np.zeros_like(seq)
                    sca2 = np.zeros_like(sca)
                self.buf.push(seq, sca, a, r, seq2, sca2, done)

                # Train
                self.total_steps += 1
                if self.total_steps >= self.cfg.start_steps and len(self.buf) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        b = self.buf.sample(self.cfg.batch_size, self.cfg.device)
                        loss_q, loss_actor, alpha = self.agent.train_step(b)
                        self.epoch_train_count += 1
                        
                        # TensorBoard 记录
                        if self.writer and (self.total_steps % self.cfg.tb_log_interval) == 0:
                            self.writer.add_scalar('Loss/Critic', loss_q, self.total_steps)
                            self.writer.add_scalar('Loss/Actor', loss_actor, self.total_steps)
                            self.writer.add_scalar('SAC/Alpha', alpha, self.total_steps)
                            self.writer.add_scalar('SAC/Lambda', self.rw.lam, self.total_steps)
                        
                        if (self.total_steps % 50) == 0:
                            self._log(2, f"[Train] step={self.total_steps} Lq={loss_q:.4f} La={loss_actor:.4f} alpha={alpha:.4f}")

                self._last_delta = float(pend["delta_qp"])

                # Episode end?
                if gop_end and info is not None:
                    # info already calculated above
                    self._last_mg_id = None
                    self._last_mg_id = None
                    
                    # 更新 epoch 统计
                    self.epoch_episodes += 1
                    self.epoch_total_reward += info['episode_return']
                    self.epoch_total_bits += info['sum_bits']
                    self.epoch_total_score += info['sum_score']
                    self.epoch_total_bits_alloc += info['sum_bits_alloc']
                    self.epoch_total_score_alloc += info['sum_score_alloc']
                    self.epoch_total_frames += info['num_frames']
                    bits_saved = info['sum_bits_alloc'] - info['sum_bits']
                    score_gained = info['sum_score'] - info['sum_score_alloc']
                    self.epoch_bits_saved += bits_saved
                    self.epoch_score_gain += score_gained
                    
                    # TensorBoard 记录 episode 指标
                    if self.writer:
                        self.writer.add_scalar('Episode/Return', info['episode_return'], self.epoch_episodes)
                        self.writer.add_scalar('Episode/Steps', info['steps'], self.epoch_episodes)
                        self.writer.add_scalar('Episode/Bits', info['sum_bits'], self.epoch_episodes)
                        self.writer.add_scalar('Episode/Score', info['sum_score'], self.epoch_episodes)
                        self.writer.add_scalar('Episode/Bits_Saved', bits_saved, self.epoch_episodes)
                        self.writer.add_scalar('Episode/Score_Gain', score_gained, self.epoch_episodes)
                        self.writer.add_scalar('Episode/Lambda', info['lambda'], self.epoch_episodes)
                    
                    # 计算平均 PSNR 和 kbps
                    num_frames = info['num_frames']
                    avg_psnr_new = info['sum_score'] / max(num_frames, 1) if num_frames > 0 else 0.0
                    avg_psnr_orig = info['sum_score_alloc'] / max(num_frames, 1) if num_frames > 0 else 0.0
                    kbps_new = (info['sum_bits'] / max(num_frames, 1)) * self.current_fps / 1000.0 if num_frames > 0 else 0.0
                    kbps_orig = (info['sum_bits_alloc'] / max(num_frames, 1)) * self.current_fps / 1000.0 if num_frames > 0 else 0.0
                    
                    # 打印详细的 episode 总结
                    self._log(1, f"\n{'='*80}")
                    self._log(1, f"[EPISODE END] Epoch #{self.current_epoch} | Episode #{self.epoch_episodes}")
                    self._log(1, f"{'='*80}")
                    self._log(1, f"  步数(Steps):           {info['steps']}")
                    self._log(1, f"  帧数(Frames):          {num_frames}")
                    self._log(1, f"  Episode 总回报:        {info['episode_return']:+.4f}")
                    self._log(1, f"  终止奖励(Term):        {info['term_bonus']:+.4f}")
                    self._log(1, f"  Lambda 值:             {info['lambda']:.6f}")
                    self._log(1, f"")
                    self._log(1, f"  码率统计:")
                    self._log(1, f"    新码率:              {info['sum_bits']:.1f} bits ({kbps_new:.2f} kbps)")
                    self._log(1, f"    原码率:              {info['sum_bits_alloc']:.1f} bits ({kbps_orig:.2f} kbps)")
                    self._log(1, f"    节省码率:            {bits_saved:+.1f} bits ({info['delta_bits_norm']*100:+.2f}%)")
                    self._log(1, f"")
                    self._log(1, f"  质量统计:")
                    self._log(1, f"    新质量分:            {info['sum_score']:.3f} (平均 PSNR: {avg_psnr_new:.3f} dB)")
                    self._log(1, f"    原质量分:            {info['sum_score_alloc']:.3f} (平均 PSNR: {avg_psnr_orig:.3f} dB)")
                    self._log(1, f"    质量提升:            {score_gained:+.3f} ({info['delta_score_norm']*100:+.2f}%)")
                    self._log(1, f"")
                    self._log(1, f"  平均 Score EMA:        {self.rw.score_ema.get():.3f}")
                    self._log(1, f"{'='*80}\n")

                # 立即删除已处理的 FB 文件
                try:
                    os.remove(fb_path)
                    self._log(3, f"[MG][FB] ⑥ 删除反馈 -> {fb_path}")
                except Exception as e:
                    self._log(2, f"[MG][WARN] 删除 FB 失败: {e}")

                # FB 处理完毕，可以处理下一个 RQ 了
                waiting_for_fb = False
                fb_wait_start_time = None
                self._log(2, f"[MG] <<< 反馈已处理，准备接收下一个请求 (RQ)...\n")
                progressed = True

            if not progressed:
                idle_loops += 1
                consecutive_idle_count += 1
                
                # 每秒打印一次等待信息
                if idle_loops * self.cfg.poll_ms >= 1000:
                    wait_status = "等待FB" if waiting_for_fb else "等待RQ"
                    if waiting_for_fb and fb_wait_start_time is not None:
                        wait_elapsed = time.time() - fb_wait_start_time
                        self._log(3, f"[MG][WAIT] {wait_status} (已等待 {wait_elapsed:.1f}s, pending={len(self.pending)}, last_mg={self._last_mg_id})")
                    else:
                        self._log(3, f"[MG][WAIT] {wait_status} (pending={len(self.pending)}, last_mg={self._last_mg_id})")
                    idle_loops = 0
                    
                    # 检查编码器是否已退出
                    if stop_evt.is_set():
                        self._log(2, f"[MG][INFO] 检测到编码器已退出，停止等待")
                        break
                    
                    # 定期清理过期的 .tmp 文件（编码器写入失败的残留）
                    self._cleanup_stale_tmp_files(rl_dir)
                
                # 如果连续空闲时间过长且没有 pending，检查是否应该退出
                if consecutive_idle_count >= max_consecutive_idle:
                    has_rq = bool(_scan_mg_rq_files(rl_dir))
                    has_fb = bool(_scan_mg_fb_files(rl_dir))
                    has_pending = len(self.pending) > 0
                    
                    if not has_rq and not has_fb and not has_pending:
                        self._log(2, f"[MG][INFO] 连续空闲 {consecutive_idle_count * self.cfg.poll_ms / 1000:.1f}s 且无待处理项，可能编码器已完成")
                        # 不直接退出，让 monitor 线程设置 stop_evt
                        consecutive_idle_count = 0  # 重置计数器继续等待
                
                time.sleep(self.cfg.poll_ms / 1000.0)
            else:
                idle_loops = 0
                consecutive_idle_count = 0  # 有进展时重置计数器
        
        # serve_loop 退出前的状态检查和清理
        print(f"\n[Run] RL loop 收到停止信号 (编码器已退出)")
        self._log(1, f"[Run] 当前状态: pending={len(self.pending)}, waiting_for_fb={waiting_for_fb}, total_mg_seen={self._mg_seen}")
        
        # 检查是否在等待 FB 时编码器退出
        if waiting_for_fb and self._last_mg_id is not None:
            wait_duration = time.time() - fb_wait_start_time if fb_wait_start_time else 0
            self._log(1, f"[Run][WARN] 编码器退出时，RL 正在等待 mg{self._last_mg_id:04d} 的反馈 (FB)")
            self._log(1, f"[Run][WARN] 已等待 {wait_duration:.1f}s，但编码器未发送该 FB（可能编码器异常退出或编码已完成）")
            if self._last_mg_id in self.pending:
                self._log(1, f"[Run] 清理未完成的 MG: {self._last_mg_id}")
                self.pending.pop(self._last_mg_id)
        
        # 清理其他残留的 pending 项
        if len(self.pending) > 0:
            self._log(1, f"[Run][WARN] 退出时还有 {len(self.pending)} 个待处理的 MG: {list(self.pending.keys())}")
            for mg_id in list(self.pending.keys()):
                self._log(1, f"[Run] 清理 pending MG: {mg_id}")
            self.pending.clear()
        
        # 清理残留的 RQ 文件
        remaining_rq = _scan_mg_rq_files(rl_dir)
        remaining_fb = _scan_mg_fb_files(rl_dir)
        
        if remaining_rq:
            self._log(1, f"[Run][WARN] 退出时还有 {len(remaining_rq)} 个未处理的 RQ 文件（编码器退出时这些请求尚未被 RL 处理）")
            for rq_path in remaining_rq:
                try:
                    os.remove(rq_path)
                    self._log(1, f"[Run] 清理残留 RQ -> {os.path.basename(rq_path)}")
                except Exception as e:
                    self._log(1, f"[Run][WARN] 清理失败: {e}")
        
        if remaining_fb:
            self._log(1, f"[Run][WARN] 退出时还有 {len(remaining_fb)} 个未处理的 FB 文件")
            for fb_path in remaining_fb:
                try:
                    os.remove(fb_path)
                    self._log(1, f"[Run] 清理残留 FB -> {os.path.basename(fb_path)}")
                except Exception as e:
                    self._log(1, f"[Run][WARN] 清理失败: {e}")
        
        if not remaining_rq and not remaining_fb and len(self.pending) == 0:
            self._log(1, f"[Run] 所有任务已完成，正常退出")
        
        self._log(1, f"[Run] RL loop exited.\n")

    def print_epoch_summary(self, epoch_id: int, epoch_total: int, interrupted: bool = False):
        """打印 epoch 结束后的详细统计信息"""
        self._log(1, f"\n{'#'*80}")
        if interrupted:
            self._log(1, f"# EPOCH #{epoch_id}/{epoch_total} 统计（已中断）")
        else:
            self._log(1, f"# EPOCH #{epoch_id}/{epoch_total} 统计")
        self._log(1, f"{'#'*80}")
        
        if self.epoch_episodes == 0:
            self._log(1, "  本 Epoch 未完成任何 Episode")
            self._log(1, f"{'#'*80}\n")
            return
        
        avg_reward = self.epoch_total_reward / self.epoch_episodes
        avg_bits = self.epoch_total_bits / self.epoch_episodes
        avg_score = self.epoch_total_score / self.epoch_episodes
        avg_bits_alloc = self.epoch_total_bits_alloc / self.epoch_episodes
        avg_score_alloc = self.epoch_total_score_alloc / self.epoch_episodes
        avg_frames = self.epoch_total_frames / self.epoch_episodes
        avg_bits_saved = self.epoch_bits_saved / self.epoch_episodes
        avg_score_gain = self.epoch_score_gain / self.epoch_episodes
        
        # 计算平均 PSNR 和 kbps（使用当前视频的 fps）
        avg_psnr = avg_score / max(avg_frames, 1) if avg_frames > 0 else 0.0
        avg_psnr_alloc = avg_score_alloc / max(avg_frames, 1) if avg_frames > 0 else 0.0
        avg_kbps = (avg_bits / max(avg_frames, 1)) * self.current_fps / 1000.0 if avg_frames > 0 else 0.0
        avg_kbps_alloc = (avg_bits_alloc / max(avg_frames, 1)) * self.current_fps / 1000.0 if avg_frames > 0 else 0.0
        
        # 计算平均码率节省（kbps）和平均质量提升（PSNR）
        avg_kbps_saved = avg_kbps_alloc - avg_kbps  # 原码率 - 新码率
        avg_psnr_gain = avg_psnr - avg_psnr_alloc  # 新质量 - 原质量
        
        # TensorBoard 记录 epoch 指标
        if self.writer:
            self.writer.add_scalar('Epoch/Episodes', self.epoch_episodes, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Return', avg_reward, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Bits', avg_bits, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Score', avg_score, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Bits_Alloc', avg_bits_alloc, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Score_Alloc', avg_score_alloc, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Bits_Saved', avg_bits_saved, epoch_id)
            self.writer.add_scalar('Epoch/Avg_Score_Gain', avg_score_gain, epoch_id)
            self.writer.add_scalar('Epoch/Avg_KBPS', avg_kbps, epoch_id)
            self.writer.add_scalar('Epoch/Avg_KBPS_Alloc', avg_kbps_alloc, epoch_id)
            self.writer.add_scalar('Epoch/Avg_KBPS_Saved', avg_kbps_saved, epoch_id)
            self.writer.add_scalar('Epoch/Avg_PSNR', avg_psnr, epoch_id)
            self.writer.add_scalar('Epoch/Avg_PSNR_Alloc', avg_psnr_alloc, epoch_id)
            self.writer.add_scalar('Epoch/Avg_PSNR_Gain', avg_psnr_gain, epoch_id)
            self.writer.add_scalar('Epoch/Buffer_Size', len(self.buf) if self.buf else 0, epoch_id)
        
        self._log(1, f"  完成 Episodes:         {self.epoch_episodes}")
        self._log(1, f"  总训练步数:            {self.total_steps}")
        self._log(1, f"  本 Epoch 训练次数:     {self.epoch_train_count}")
        self._log(1, f"  Replay Buffer 大小:    {len(self.buf) if self.buf else 0}")
        self._log(1, f"")
        self._log(1, f"  平均 Episode 回报:     {avg_reward:+.4f}")
        self._log(1, f"  平均 Episode 帧数:     {avg_frames:.1f}")
        self._log(1, f"")
        self._log(1, f"  码率统计:")
        self._log(1, f"    平均新码率:          {avg_kbps:.2f} kbps")
        self._log(1, f"    平均原码率:          {avg_kbps_alloc:.2f} kbps")
        self._log(1, f"    平均码率节省:        {avg_kbps_saved:+.2f} kbps")
        self._log(1, f"")
        self._log(1, f"  质量统计:")
        self._log(1, f"    平均新质量:          {avg_psnr:.3f} dB")
        self._log(1, f"    平均原质量:          {avg_psnr_alloc:.3f} dB")
        self._log(1, f"    平均质量提升:        {avg_psnr_gain:+.3f} dB")
        self._log(1, f"")
        self._log(1, f"  当前 Lambda:           {self.rw.lam:.6f}")
        self._log(1, f"  当前 Score EMA:        {self.rw.score_ema.get():.3f}")
        
        if self.agent:
            alpha_val = self.agent.log_alpha.exp().item()
            self._log(1, f"  当前 SAC Alpha:        {alpha_val:.4f}")
        
        self._log(1, f"{'#'*80}\n")
        
        # 重置 epoch 统计
        self.epoch_episodes = 0
        self.epoch_total_reward = 0.0
        self.epoch_total_bits = 0.0
        self.epoch_total_score = 0.0
        self.epoch_total_bits_alloc = 0.0
        self.epoch_total_score_alloc = 0.0
        self.epoch_total_frames = 0
        self.epoch_bits_saved = 0.0
        self.epoch_score_gain = 0.0
        self.epoch_train_count = 0
