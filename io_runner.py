# -*- coding: utf-8 -*-
"""
MiniGOP I/O runner:
- Watch rl_dir for mg????_rq.json / mg????_fb.json (encoder handshake via rl_sync.*)
- Build a [C=6, T] feature block via state.build_state_from_rq (channels: poise, comp, rdcost, score_target, bit_target, qp/256)
- Actor outputs 5 deltas (对应 5 个时域等级：1, 2, 3, 4, 6). 根据每帧的 temporal_level 映射对应的 delta 到该帧，然后应用到 qp
- Write mg????_qp.json with {"qps": [...]}
- Reward: per-MG via reward.RewardComputer.step; episode ends when fb.gop_end == 1

【重要时序逻辑】Replay buffer push 时机：
1. RQ_t 到达 → 构建 s_t，输出 a_t，写 QP，暂存 pending[t] = {seq, scalars, a, ...}
2. 如果 pending[t-1] 已有 "reward" 字段（说明 FB_{t-1} 已到）：
   - 用 s_t 作为 s'，push (s_{t-1}, a_{t-1}, r_{t-1}, s_t, done_{t-1})
   - 删除 pending[t-1]
3. FB_t 到达 → 计算 r_t 和 done_t，记录到 pending[t]["reward"], pending[t]["done"]
   - 如果 done_t == True（episode 结束）：直接 push (s_t, a_t, r_t, zeros, True)，删除 pending[t]
   - 否则：等待下一个 RQ 来补齐 s'

这样确保所有非终止步都有正确的 next_state，避免用全 0 的 s' 训练。
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
                bitrate_tolerance=cfg.bitrate_tolerance,
                bitrate_hard_ratio=cfg.bitrate_hard_ratio,
                over_bitrate_penalty=cfg.over_bitrate_penalty,
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
        
        while not stop_evt.is_set():
            progressed = False

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
                seq, scalars, qps, temporal_level, mg_id, mg_size, bits_alloc, score_alloc = build_state_from_rq(
                    self.cfg, rq, g_state
                )
                if self.agent is None:
                    self._ensure_models(seq.shape, scalars.shape[0])

                self._mg_seen = max(self._mg_seen, mg_id + 1)
                avg_qp = float(np.mean(qps)) if len(qps) > 0 else 0.0
                self._log(2, f"[MG][RQ] ① 接收请求 -> {rq_path} | id={mg_id} size={mg_size} avg_qp={avg_qp:.2f}")
                
                # 【关键修改】如果上一个 MG 已经收到 FB（有 reward 字段），用当前 state 作为其 s'，并 push
                if self._last_mg_id is not None and self._last_mg_id in self.pending:
                    prev = self.pending[self._last_mg_id]
                    if "reward" in prev:
                        # 上一个 transition 已经有 reward 了，现在补齐 s' 并 push
                        self.buf.push(
                            prev["seq"], prev["scalars"], prev["a"], 
                            prev["reward"], 
                            seq, scalars,  # 用当前 RQ 的 state 作为 s'
                            prev["done"]
                        )
                        self._log(3, f"[Replay] Push transition: mg_id={self._last_mg_id} -> {mg_id} (done={prev['done']})")
                        # 删除已经 push 的 pending
                        self.pending.pop(self._last_mg_id)
                        self._last_mg_id = None
                    else:
                        # 上一个MG还没有reward，无法闭合transition
                        self._log(1, f"[Replay][WARN] prev mg_id={self._last_mg_id} has no reward yet; cannot build transition")

                # Action
                seq1 = torch.from_numpy(seq).unsqueeze(0).to(self.cfg.device).float()
                sca1 = torch.from_numpy(scalars).unsqueeze(0).to(self.cfg.device).float()
                num_actions = int(getattr(self.agent, 'num_discrete_actions', 0) or 0)
                discrete_values = getattr(self.agent, 'discrete_action_values', None)
                if discrete_values is None:
                    if num_actions <= 0:
                        num_actions = max(1, int(self.cfg.delta_qp_max) * 2 + 1)
                    discrete_values = np.linspace(-self.cfg.delta_qp_max, self.cfg.delta_qp_max, num_actions, dtype=np.float32)
                else:
                    discrete_values = discrete_values.detach().cpu().numpy().astype(np.float32)
                
                # Temporal level to index map: 1->0, 2->1, 3->2, 4->3, 6->4
                def level_to_idx(level):
                    level_map = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}
                    return level_map.get(int(level), 4)
                
                # Inference mode: deterministic policy (no exploration)
                # Train mode: random explore before start_steps, then policy
                baseline_prob = float(getattr(self.cfg, "baseline_action_prob", 0.0))
                if baseline_prob < 0.0:
                    baseline_prob = 0.0
                if baseline_prob > 1.0:
                    baseline_prob = 1.0
                use_baseline = (self.cfg.mode == "train" and baseline_prob > 0.0 and np.random.rand() < baseline_prob)
                if self.cfg.mode == "infer":
                    with torch.no_grad():
                        a_idx_t, _ = self.agent.act(seq1, sca1, deterministic=True)
                    a_idx = a_idx_t.squeeze(0).detach().cpu().numpy().astype(np.int32)
                    act_src = "policy_det"
                elif use_baseline:
                    zero_idx = int(np.argmin(np.abs(discrete_values)))
                    a_idx = np.full((5,), zero_idx, dtype=np.int32)
                    act_src = "baseline"
                elif self.total_steps < self.cfg.start_steps:
                    a_idx = np.random.randint(0, num_actions, size=(5,), dtype=np.int32)
                    act_src = "explore"
                else:
                    with torch.no_grad():
                        a_idx_t, _ = self.agent.act(seq1, sca1, deterministic=False)
                    a_idx = a_idx_t.squeeze(0).detach().cpu().numpy().astype(np.int32)
                    act_src = "policy"
                
                # Apply delta per frame based on temporal_level
                # temporal_level length equals mg_size
                delta_qps = np.zeros(mg_size, dtype=np.float32)
                for i in range(mg_size):
                    level_idx = level_to_idx(temporal_level[i])
                    action_idx = int(a_idx[level_idx])
                    delta_qps[i] = float(discrete_values[action_idx])
                
                # New qp per frame: qp_new = clip(qp_old + delta_qp, qp_min, qp_max)
                # qps length equals mg_size
                qps_new = np.clip(qps[:mg_size] + delta_qps, self.cfg.qp_min, self.cfg.qp_max)
                qps_new = np.rint(qps_new).astype(np.int32)
                if bool(getattr(self.cfg, "log_delta_qvals", False)):
                    delta_list = [float(d) for d in delta_qps.tolist()]
                    level_list = temporal_level[:mg_size].tolist()
                    delta_by_level = {level: float(discrete_values[a_idx[level_to_idx(level)]]) for level in [1, 2, 3, 4, 6]}
                    self._log(2, f"[MG][DELTA_QP] id={mg_id} temporal_level={level_list} delta_qps={delta_list} delta_by_level={delta_by_level}")
                
                # Log average qp and delta
                avg_qp_new = float(np.mean(qps_new))
                avg_qp_old = float(np.mean(qps[:mg_size]))
                avg_delta_qp = float(np.mean(delta_qps))
                # Log encoder/RL qp sequences when log_level>=3
                self._log(3, f"[MG][QPS] enc_qps(id={mg_id}): {qps[:mg_size].tolist()}")
                self._log(3, f"[MG][QPS] rl_qps(id={mg_id}):  {qps_new.tolist()}")
                self._log(2, f"[MG][ACT] action -> id={mg_id} src={act_src} avg_delta_qp={avg_delta_qp:+.2f} avg_qp={avg_qp_old:.2f}->{avg_qp_new:.2f}")

                # Write QP json for this mg (新格式：qps 数组)
                qp_path = rq_path.replace("_rq.json", "_qp.json")
                qps_list = [int(q) for q in qps_new.tolist()]
                safe_write_json_atomic(qp_path, {"qps": qps_list})
                self._log(3, f"[MG][QP] ③ 写入决策 -> {qp_path} (qps={len(qps_list)} frames)")

                # 【新逻辑】暂存当前 MG 的 state 和 action，等待 FB 补齐 reward
                if mg_id in self.pending:
                    # GOP重置或乱序可能导致覆盖
                    self._log(1, f"[Replay][WARN] pending already has mg_id={mg_id}; overwriting (pending_keys={list(self.pending.keys())})")
                self.pending[mg_id] = dict(
                    seq=seq,
                    scalars=scalars,
                    a=a_idx.copy(),  # [5] 维的 action（对应 5 个时域等级）
                    delta_qp=avg_delta_qp,  # 用于 reward 计算（保持标量）
                    bits_alloc=bits_alloc,
                    score_alloc=score_alloc,
                    mg_size=mg_size,
                    # reward 和 done 将在 FB 到来时填充
                )
                self._last_mg_id = mg_id
                
                # 立即删除已处理的 RQ 文件
                try:
                    os.remove(rq_path)
                    self._log(3, f"[MG][RQ] ④ 删除请求 -> {rq_path}")
                except Exception as e:
                    self._log(2, f"[MG][WARN] 删除 RQ 失败: {e}")
                
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

                # 【修复】不要立即 pop，先获取引用，后续根据情况再决定是否删除
                pend = self.pending[mg_id]
                bits = float(fb.get("bits", 0.0))
                score = float(fb.get("score", 0.0))
                bits_alloc = float(pend.get("bits_alloc", 0.0))
                score_alloc = float(pend.get("score_alloc", 0.0))
                num_frames = int(fb.get("num_frames", fb.get("n_frames", pend.get("mg_size", 0))))  # 该 mini-GOP 的帧数
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
                if num_frames <= 0:
                    num_frames = int(pend.get("mg_size", 0))
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

                # 【新逻辑】在 pending 中记录 reward 和 done
                pend["reward"] = r
                pend["done"] = gop_end
                
                # 如果是终止步（done=True），直接 push（不需要真实的 s'）
                if gop_end:
                    if len(self.pending) > 1:
                        self._log(1, f"[Replay][WARN] gop_end with other pending MGs: {list(self.pending.keys())}")
                    seq = pend["seq"]
                    sca = pend["scalars"]
                    a = pend["a"]
                    seq2 = np.zeros_like(seq)  # 终止步的 s' 不重要
                    sca2 = np.zeros_like(sca)
                    self.buf.push(seq, sca, a, r, seq2, sca2, done=True)
                    self._log(3, f"[Replay] Push terminal transition: mg_id={mg_id} (done=True)")
                    # 删除已 push 的 pending
                    self.pending.pop(mg_id)
                    self._last_mg_id = None
                else:
                    # 非终止步，等待下一个 RQ 来补齐 s'（pending 保留，不 pop）
                    # pend 已经是 self.pending[mg_id] 的引用，修改会自动同步
                    self._log(3, f"[Replay] Waiting for next RQ to complete transition: mg_id={mg_id}")

                # Train（仅训练模式，推理模式跳过）
                self.total_steps += 1
                if self.cfg.mode == "train" and self.total_steps >= self.cfg.start_steps and len(self.buf) >= self.cfg.batch_size:
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

                self._log(2, f"[MG] <<< 反馈已处理，继续处理 RQ/FB...\n")
                progressed = True

            if not progressed:
                idle_loops += 1
                consecutive_idle_count += 1
                
                # 每秒打印一次等待信息
                if idle_loops * self.cfg.poll_ms >= 1000:
                    self._log(3, f"[MG][WAIT] 等待 RQ/FB (pending={len(self.pending)}, last_mg={self._last_mg_id})")
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
        self._log(1, f"[Run] 当前状态: pending={len(self.pending)}, total_mg_seen={self._mg_seen}")
        
        # 检查是否有待处理的 pending（可能是编码器意外退出）
        if self._last_mg_id is not None and self._last_mg_id in self.pending:
            pend = self.pending[self._last_mg_id]
            if "reward" in pend:
                self._log(1, f"[Run][WARN] 编码器退出时，MG {self._last_mg_id} 已收到 FB 但未收到后续 RQ 补齐 s'")
            else:
                self._log(1, f"[Run][WARN] 编码器退出时，MG {self._last_mg_id} 尚未收到 FB")
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
