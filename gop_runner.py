# -*- coding: utf-8 -*-
"""
GOP-Level I/O Runner:
- Watch rl_dir for gop????_rq.json / gop????_fb.json
- Build state via state.build_state_from_gop_rq
- Actor outputs single QP value
- Write gop????_qp.json with {"qp": value}
- Reward: per-GOP via reward.GOPRewardComputer.step

【时序逻辑】
1. RQ_t 到达 → 构建 s_t，输出 a_t (单个 QP)，写 QP 文件
2. FB_t 到达 → 计算 r_t，存入 replay buffer
3. Episode 结束条件：视频编码完成（可通过 is_last_gop 标志）
"""
import os, glob, time, json, numpy as np, torch
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from config import Config
from utils import safe_read_json, safe_write_json_atomic, now_ms
from sac_agent import D3QNAgent
from replay import ReplayBuffer
from state import build_state_from_gop_rq
from reward import GOPRewardComputer, RewardCfg

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def _scan_gop_rq_files(rl_dir: str) -> List[str]:
    """扫描 GOP 级别的 RQ 文件"""
    return sorted(glob.glob(os.path.join(rl_dir, "gop????_rq.json")))


def _scan_gop_fb_files(rl_dir: str) -> List[str]:
    """扫描 GOP 级别的 FB 文件"""
    return sorted(glob.glob(os.path.join(rl_dir, "gop????_fb.json")))


class GOPRunner:
    """GOP 级别的 RL 运行器"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.agent: Optional[D3QNAgent] = None
        self.buf: Optional[ReplayBuffer] = None
        self.current_epoch: int = 1
        self.current_fps: float = cfg.fps
        
        # GOP 级别的 reward 计算器
        self.rw = GOPRewardComputer(
            RewardCfg(
                bitrate_save_weight=getattr(cfg, 'bitrate_save_weight', 1.0),
                quality_smooth_weight=getattr(cfg, 'quality_smooth_weight', 0.1),
            )
        )
        
        # Pending transitions: gop_id -> {seq, scalars, action, ...}
        self.pending: Dict[int, Dict] = {}
        self._last_gop_id: Optional[int] = None
        
        self.total_steps = 0
        self._gop_seen = 0
        
        # 记录已处理的 RQ (gop_id)，防止重复处理
        self.processed_rq_ids = set()
        
        # Epoch 统计
        self.epoch_episodes = 0
        self.epoch_total_reward = 0.0
        self.epoch_total_bits = 0.0
        self.epoch_total_score = 0.0
        self.epoch_total_frames = 0
        self.epoch_train_count = 0
        
        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if cfg.use_tensorboard and TENSORBOARD_AVAILABLE:
            from datetime import datetime
            log_dir = os.path.join(cfg.tensorboard_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.writer = SummaryWriter(log_dir)
            self._log(1, f"[TensorBoard] 日志目录: {log_dir}")
        
        # FB 读取重试机制
        self.fb_read_failures: Dict[str, int] = defaultdict(int)
        self.fb_max_retries: int = 1800  # 1800 次重试
        self.fb_retry_wait_ms: int = 1000  # 每次重试等待 1000ms (总计 30 分钟)
        
    def _log(self, level: int, msg: str) -> None:
        """日志输出"""
        if level <= self.cfg.log_level:
            print(msg)
    
    def set_epoch(self, epoch_id: int) -> None:
        """更新当前 epoch"""
        self.current_epoch = max(1, epoch_id)
    
    def set_current_fps(self, fps: float) -> None:
        """更新当前视频 fps"""
        self.current_fps = max(1.0, fps)
    
    def _ensure_models(self, seq_shape: Tuple[int, int], scalar_dim: int):
        """确保模型已初始化"""
        if self.agent is not None:
            return
        C, T = seq_shape
        self.agent = D3QNAgent(self.cfg, state_scalar_dim=scalar_dim, seq_T=T, seq_C=C)
        self.buf = ReplayBuffer(self.cfg.replay_size, (C, T), scalar_dim)
        self._log(1, f"[RL] Models ready. State(seq)={C}x{T}, scalars={scalar_dim}")
    
    def serve_loop(self, stop_evt) -> None:
        """主循环：处理 GOP 级别的 RQ 和 FB"""
        rl_dir = self.cfg.rl_dir
        self._log(1, f"[Run] GOP RL loop started. rl_dir={rl_dir}")
        
        # 等待第一个 RQ 到达
        wait_ms = 0
        max_wait_ms = 30000
        while not stop_evt.is_set():
            rq_files = _scan_gop_rq_files(rl_dir)
            if rq_files:
                break
            wait_ms += self.cfg.poll_ms
            if wait_ms % 1000 == 0:
                self._log(3, f"[GOP][WAIT] no rq yet (waited {wait_ms/1000:.1f}s)")
            if wait_ms >= max_wait_ms:
                self._log(2, f"[GOP][WARN] 超过 {max_wait_ms/1000:.0f}s 未收到 rq")
                return
            time.sleep(self.cfg.poll_ms / 1000.0)
        
        rq_read_failures: Dict[str, int] = defaultdict(int)
        idle_loops = 0
        
        # 清空已处理的RQ记录（新视频/新episode）
        self.processed_rq_ids.clear()
        
        # 死锁检测：跟踪等待 FB 的时间
        waiting_for_fb_since: Dict[int, float] = {}  # gop_id -> 开始等待的时间戳
        fb_wait_timeout_sec = 1800  # 30分钟超时
        
        # 记录已跳过的 RQ，避免重复打印日志
        skipped_rq_ids = set()
        
        while not stop_evt.is_set():
            progressed = False
            
            # === 优先处理 FB，再处理 RQ ===
            # 先处理所有可用的 FB
            fb_files = _scan_gop_fb_files(rl_dir)
            for fb_path in fb_files:
                try:
                    fb = safe_read_json(fb_path)
                    if fb_path in self.fb_read_failures:
                        self.fb_read_failures.pop(fb_path)
                except Exception as e:
                    self.fb_read_failures[fb_path] += 1
                    retry_count = self.fb_read_failures[fb_path]
                    
                    if retry_count < self.fb_max_retries:
                        # 还有重试机会，等待后继续
                        if retry_count % 10 == 1:  # 每 10 次重试记录一次日志
                            self._log(2, f"[GOP][FB][WARN] FB 读取失败 (重试 {retry_count}/{self.fb_max_retries}): {fb_path}: {e}")
                        time.sleep(self.fb_retry_wait_ms / 1000.0)
                        continue
                    else:
                        # 达到最大重试次数，程序退出
                        error_msg = (
                            f"\n{'='*60}\n"
                            f"[GOP][FB][FATAL] FB 文件读取失败，已重试 {self.fb_max_retries} 次\n"
                            f"文件路径: {fb_path}\n"
                            f"错误信息: {e}\n"
                            f"程序将退出以避免数据不一致\n"
                            f"{'='*60}\n"
                        )
                        self._log(1, error_msg)
                        
                        # 抛出异常，让程序退出
                        raise RuntimeError(f"FB 文件读取失败: {fb_path}") from e
                
                gop_id = int(fb.get("gop_id", -1))
                if gop_id not in self.pending:
                    self._log(2, f"[GOP][WARN] fb for gop_id={gop_id} has no pending RQ")
                    try:
                        os.remove(fb_path)
                    except:
                        pass
                    continue
                
                pend = self.pending[gop_id]
                
                # 解析 FB 数据
                bitrate = float(fb.get("bitrate", 0.0))
                score = float(fb.get("score", 0.0))
                target_bitrate = float(pend.get("target_bitrate", 2000.0))
                target_score = float(pend.get("target_score", 40.0))
                gop_size = int(pend.get("gop_size", 225))
                is_first_gop = bool(pend.get("is_first_gop", False))
                
                # 计算 reward
                r = self.rw.step(
                    bitrate=bitrate,
                    score=score,
                    target_bitrate=target_bitrate,
                    target_score=target_score,
                    gop_size=gop_size,
                    is_first_gop=is_first_gop,
                )
                
                # 判断是否为 episode 结束（可通过 is_last 字段或其他逻辑）
                is_last_gop = bool(fb.get("is_last", fb.get("is_last_gop", False)))
                
                self._log(2, 
                    f"[GOP][FB] ④ 接收反馈 -> {fb_path} | gop_id={gop_id} "
                    f"bitrate={bitrate:.1f}/{target_bitrate:.1f} score={score:.2f}/{target_score:.2f} reward={r:.4f}"
                )
                
                # 记录此 FB 的 score 到 pending，用于下一个 RQ 的验证
                pend["fb_score"] = score
                
                # 记录 reward 和 done
                pend["reward"] = r
                pend["done"] = is_last_gop
                
                # 如果这个 GOP 之前在等待列表中，清除等待状态
                if gop_id in waiting_for_fb_since:
                    waiting_for_fb_since.pop(gop_id)
                
                # 如果是最后一个 GOP，直接 push（终止步）
                if is_last_gop:
                    # 计算 episode bonus（全局优化信号）
                    episode_bonus = self.rw.compute_episode_bonus()
                    
                    # 将 episode bonus 加到最后一个 GOP 的 reward
                    r_final = r + episode_bonus * 0.3  # 权重 0.3
                    
                    seq = pend["seq"]
                    sca = pend["scalars"]
                    a = pend["a"]
                    seq2 = np.zeros_like(seq)
                    sca2 = np.zeros_like(sca)
                    self.buf.push(seq, sca, a, r_final, seq2, sca2, done=True)
                    self._log(3, f"[Replay] Push terminal: gop_id={gop_id} r_gop={r:.4f} bonus={episode_bonus:.4f} r_total={r_final:.4f}")
                    self.pending.pop(gop_id)
                    self._last_gop_id = None
                    
                    # Episode 结束统计
                    info = self.rw.on_episode_end()
                    self.epoch_episodes += 1
                    self.epoch_total_reward += info['episode_return']
                    self.epoch_total_frames += info['total_frames']
                    
                    self._log(1, f"\n{'='*60}")
                    self._log(1, f"[EPISODE END] Epoch #{self.current_epoch}")
                    self._log(1, f"  GOP 数量: {info['gop_count']}")
                    self._log(1, f"  总帧数: {info['total_frames']}")
                    self._log(1, f"  Episode 回报: {info['episode_return']:+.4f}")
                    self._log(1, f"  平均码率: {info['avg_bitrate']:.2f} ({info['bitrate_saved_pct']:+.1f}%)")
                    self._log(1, f"  平均质量: {info['avg_score']:.2f} ({info['score_diff_pct']:+.1f}%)")
                    self._log(1, f"{'='*60}\n")
                else:
                    # 非终止 GOP：标记为 last_gop_id，等待下一个 RQ 到来时存储转移
                    self._last_gop_id = gop_id
                    
                    # === 新增：检查是否有后续 GOP 已经在 pending 中 ===
                    # 如果 GOP 顺序是：RQ2 -> RQ3 -> FB2 -> FB3
                    # 那么收到 FB2 时，GOP3 已经在 pending 中，可以立即存储转移
                    next_gop_id = gop_id + 1
                    if next_gop_id in self.pending:
                        next_pend = self.pending[next_gop_id]
                        if "seq" in next_pend and "scalars" in next_pend:
                            # 下一个 GOP 的状态已经构建好，可以存储转移
                            self.buf.push(
                                pend["seq"], pend["scalars"], pend["a"],
                                r,
                                next_pend["seq"], next_pend["scalars"],
                                done=False
                            )
                            self._log(3, f"[Replay] Push transition (late FB): gop_id={gop_id} -> {next_gop_id}")
                            self.pending.pop(gop_id)
                            self._last_gop_id = None
                
                # 训练
                self.total_steps += 1
                if self.cfg.mode == "train" and self.total_steps >= self.cfg.start_steps and len(self.buf) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        b = self.buf.sample(self.cfg.batch_size, self.cfg.device)
                        loss_q, eps = self.agent.train_step(b)
                        self.epoch_train_count += 1
                        
                        if self.writer and (self.total_steps % self.cfg.tb_log_interval) == 0:
                            self.writer.add_scalar('Loss/Q', loss_q, self.total_steps)
                        
                        if (self.total_steps % 50) == 0:
                            self._log(2, f"[Train] step={self.total_steps} Lq={loss_q:.4f}")
                
                self._last_action = float(pend["action_qp"])
                
                # 删除处理过的 FB
                try:
                    os.remove(fb_path)
                    self._log(3, f"[GOP][FB] ⑤ 删除反馈 -> {fb_path}")
                except Exception as e:
                    self._log(2, f"[GOP][WARN] 删除 FB 失败: {e}")
                
                progressed = True
            
            # === 处理 RQ ===
            rq_files = _scan_gop_rq_files(rl_dir)
            if rq_files:
                rq_path = rq_files[0]
                
                # 先尝试读取文件，获取 gop_id，判断是否应该处理
                try:
                    rq = safe_read_json(rq_path)
                    if rq_path in rq_read_failures:
                        rq_read_failures.pop(rq_path, None)
                except Exception as e:
                    rq_read_failures[rq_path] += 1
                    fail_cnt = rq_read_failures[rq_path]
                    if fail_cnt <= 3:
                        self._log(2, f"[GOP][WARN] bad rq (retry #{fail_cnt}): {rq_path}: {e}")
                    time.sleep(self.cfg.poll_ms / 1000.0)
                    continue
                
                # 提取 gop_id
                gop_id = int(rq.get("gop_id", -1))
                
                # === 防止重复处理：检查此 RQ 是否已经处理过 ===
                if gop_id in self.processed_rq_ids:
                    self._log(3, f"[GOP][SKIP] GOP {gop_id} 的 RQ 已处理过，跳过")
                    try:
                        os.remove(rq_path)  # 删除重复的RQ文件
                    except Exception:
                        pass
                    continue
                
                # === 严格顺序处理：检查上一个 GOP 的 FB 是否已处理 ===
                # 如果未处理，跳过此 RQ，等下一轮循环再处理（不标记为已处理）
                if gop_id > 0:
                    prev_gop_id = gop_id - 1
                    if prev_gop_id in self.pending and "reward" not in self.pending[prev_gop_id]:
                        # 上一个 GOP 的 FB 还没处理，跳过此 RQ
                        
                        # 死锁检测：记录开始等待的时间
                        current_time = time.time()
                        if prev_gop_id not in waiting_for_fb_since:
                            waiting_for_fb_since[prev_gop_id] = current_time
                            self._log(2, f"[GOP][WAIT] 等待 GOP {prev_gop_id} 的 FB，暂时跳过 GOP {gop_id} 的 RQ")
                            skipped_rq_ids.clear()  # 清空跳过记录
                        
                        # 检查是否超时
                        wait_duration = current_time - waiting_for_fb_since[prev_gop_id]
                        if wait_duration > fb_wait_timeout_sec:
                            error_msg = (
                                f"\n{'='*60}\n"
                                f"[GOP][FATAL] 等待 GOP {prev_gop_id} 的 FB 超时！\n"
                                f"已等待时间: {wait_duration:.1f} 秒 (超时阈值: {fb_wait_timeout_sec} 秒)\n"
                                f"当前 GOP {gop_id} 无法继续处理\n"
                                f"可能原因: 编码器卡住、崩溃或 FB 文件写入失败\n"
                                f"建议检查编码器日志: ./logs/encoder/\n"
                                f"程序将退出\n"
                                f"{'='*60}\n"
                            )
                            self._log(1, error_msg)
                            raise RuntimeError(f"等待 GOP {prev_gop_id} 的 FB 超时")
                        
                        # 记录已跳过的 RQ，避免重复打印
                        if gop_id not in skipped_rq_ids:
                            skipped_rq_ids.add(gop_id)
                        
                        # 不删除 RQ 文件，下一轮循环会重新扫描到
                        # 也不标记为已处理，允许重试
                        time.sleep(0.5)  # 等待 500ms，给 FB 处理时间
                        progressed = False  # 标记为未进展，继续循环
                        continue  # 跳过当前 RQ，继续主循环
                    else:
                        # 上一个 GOP 的 FB 已经处理完成，清除等待记录
                        if prev_gop_id in waiting_for_fb_since:
                            wait_duration = time.time() - waiting_for_fb_since[prev_gop_id]
                            self._log(2, f"[GOP][RESUME] GOP {prev_gop_id} 的 FB 已到达 (等待了 {wait_duration:.1f}s)，继续处理 GOP {gop_id}")
                            waiting_for_fb_since.pop(prev_gop_id)
                            skipped_rq_ids.clear()  # 清空跳过记录
                
                # === 只有真正处理RQ时才标记为已处理 ===
                self.processed_rq_ids.add(gop_id)
                
                # 构建状态
                g_state = dict(
                    score_ema=self.rw.score_ema.get(),
                    last_action=getattr(self, "_last_action", self.cfg.default_qp),
                )
                seq, scalars, gop_id, gop_size, target_bitrate, target_score = build_state_from_gop_rq(
                    self.cfg, rq, g_state
                )
                
                # === 鲁棒性检查：验证 last_score 一致性 ===
                # 只有当上一个 GOP (gop_id-1) 的 FB 已经收到时才验证
                if gop_id > 0:
                    prev_gop_id = gop_id - 1
                    # 检查上一个 GOP 是否已经收到 FB 并记录了 score
                    if prev_gop_id in self.pending and "fb_score" in self.pending[prev_gop_id]:
                        expected_score = self.pending[prev_gop_id]["fb_score"]
                        last_score_from_rq = float(rq.get("last_score", 0.0))
                        # 允许小的浮点误差（0.01）
                        score_diff = abs(last_score_from_rq - expected_score)
                        if score_diff > 0.01:
                            error_msg = (
                                f"[ERROR] GOP {gop_id} 的 last_score 不一致！\n"
                                f"  RQ 中的 last_score: {last_score_from_rq:.2f}\n"
                                f"  GOP {prev_gop_id} FB 的 score: {expected_score:.2f}\n"
                                f"  差值: {score_diff:.2f}\n"
                                f"  这表明编码器侧的 last_score 传递有误！"
                            )
                            self._log(1, error_msg)
                            raise ValueError(error_msg)
                
                if self.agent is None:
                    self._ensure_models(seq.shape, scalars.shape[0])
                
                self._gop_seen = max(self._gop_seen, gop_id + 1)
                is_first_gop = (gop_id == 0)
                
                self._log(2, f"[GOP][RQ] ① 接收请求 -> {rq_path} | gop_id={gop_id} size={gop_size} first={is_first_gop}")
                
                # 如果上一个 GOP 已经收到 FB，用当前 state 作为 s'，并 push 到 buffer
                if self._last_gop_id is not None and self._last_gop_id in self.pending:
                    prev = self.pending[self._last_gop_id]
                    if "reward" in prev:
                        self.buf.push(
                            prev["seq"], prev["scalars"], prev["a"],
                            prev["reward"],
                            seq, scalars,
                            prev["done"]
                        )
                        self._log(3, f"[Replay] Push transition: gop_id={self._last_gop_id} -> {gop_id}")
                        self.pending.pop(self._last_gop_id)
                        self._last_gop_id = None
                
                # 选择动作
                seq1 = torch.from_numpy(seq).unsqueeze(0).to(self.cfg.device).float()
                sca1 = torch.from_numpy(scalars).unsqueeze(0).to(self.cfg.device).float()
                
                discrete_values = getattr(self.agent, "discrete_action_values", None)
                if discrete_values is not None:
                    discrete_values = discrete_values.detach().cpu().numpy() if torch.is_tensor(discrete_values) else np.array(discrete_values)
                else:
                    action_min = float(self.cfg.action_min)
                    action_max = float(self.cfg.action_max)
                    action_step = max(1, int(self.cfg.action_step))
                    discrete_values = np.array(list(range(int(action_min), int(action_max) + 1, action_step)), dtype=np.float32)
                
                num_actions = len(discrete_values)
                
                # 推理或训练模式
                if self.cfg.mode == "infer":
                    with torch.no_grad():
                        a_idx_t, _ = self.agent.act(seq1, sca1, deterministic=True)
                    a_idx = a_idx_t.squeeze(0).detach().cpu().numpy().astype(np.int32)
                    act_src = "policy_det"
                elif self.total_steps < self.cfg.start_steps:
                    a_idx = np.random.randint(0, num_actions, size=(1,), dtype=np.int32)
                    act_src = "explore"
                else:
                    with torch.no_grad():
                        a_idx_t, _ = self.agent.act(seq1, sca1, deterministic=False)
                    a_idx = a_idx_t.squeeze(0).detach().cpu().numpy().astype(np.int32)
                    act_src = "policy"
                
                # 获取单个 QP 值（取第一个动作维度）
                action_qp = int(discrete_values[int(a_idx[0])])
                action_qp = max(self.cfg.action_min, min(self.cfg.action_max, action_qp))
                
                self._log(2, f"[GOP][ACT] gop_id={gop_id} src={act_src} qp={action_qp}")
                
                # 写入 QP 文件（单个 QP 值）
                qp_path = rq_path.replace("_rq.json", "_qp.json")
                safe_write_json_atomic(qp_path, {"qp": action_qp})
                self._log(3, f"[GOP][QP] ② 写入决策 -> {qp_path} qp={action_qp}")
                
                # 暂存当前 GOP 状态
                self.pending[gop_id] = dict(
                    seq=seq,
                    scalars=scalars,
                    a=a_idx.copy(),
                    action_qp=action_qp,
                    gop_size=gop_size,
                    target_bitrate=target_bitrate,
                    target_score=target_score,
                    is_first_gop=is_first_gop,
                )
                self._last_gop_id = gop_id
                
                # 删除处理过的 RQ
                try:
                    os.remove(rq_path)
                    self._log(3, f"[GOP][RQ] ③ 删除请求 -> {rq_path}")
                except Exception as e:
                    self._log(2, f"[GOP][WARN] 删除 RQ 失败: {e}")
                
                progressed = True
            
            # === 处理 FB ===
            for fb_path in _scan_gop_fb_files(rl_dir):
                try:
                    fb = safe_read_json(fb_path)
                    if fb_path in self.fb_read_failures:
                        self.fb_read_failures.pop(fb_path)
                except Exception as e:
                    self.fb_read_failures[fb_path] += 1
                    retry_count = self.fb_read_failures[fb_path]
                    
                    if retry_count < self.fb_max_retries:
                        # 还有重试机会，等待后继续
                        if retry_count % 10 == 1:  # 每 10 次重试记录一次日志
                            self._log(2, f"[GOP][FB][WARN] FB 读取失败 (重试 {retry_count}/{self.fb_max_retries}): {fb_path}: {e}")
                        time.sleep(self.fb_retry_wait_ms / 1000.0)
                        continue
                    else:
                        # 达到最大重试次数，程序退出
                        error_msg = (
                            f"\n{'='*60}\n"
                            f"[GOP][FB][FATAL] FB 文件读取失败，已重试 {self.fb_max_retries} 次\n"
                            f"文件路径: {fb_path}\n"
                            f"错误信息: {e}\n"
                            f"程序将退出以避免数据不一致\n"
                            f"{'='*60}\n"
                        )
                        self._log(1, error_msg)
                        
                        # 抛出异常，让程序退出
                        raise RuntimeError(f"FB 文件读取失败: {fb_path}") from e
                
                gop_id = int(fb.get("gop_id", -1))
                if gop_id not in self.pending:
                    self._log(2, f"[GOP][WARN] fb for gop_id={gop_id} has no pending RQ")
                    try:
                        os.remove(fb_path)
                    except:
                        pass
                    continue
                
                pend = self.pending[gop_id]
                
                # 解析 FB 数据
                bitrate = float(fb.get("bitrate", 0.0))
                score = float(fb.get("score", 0.0))
                target_bitrate = float(pend.get("target_bitrate", 2000.0))
                target_score = float(pend.get("target_score", 40.0))
                gop_size = int(pend.get("gop_size", 225))
                is_first_gop = bool(pend.get("is_first_gop", False))
                
                # 计算 reward
                r = self.rw.step(
                    bitrate=bitrate,
                    score=score,
                    target_bitrate=target_bitrate,
                    target_score=target_score,
                    gop_size=gop_size,
                    is_first_gop=is_first_gop,
                )
                
                # 判断是否为 episode 结束（可通过 is_last 字段或其他逻辑）
                is_last_gop = bool(fb.get("is_last", fb.get("is_last_gop", False)))
                
                self._log(2, 
                    f"[GOP][FB] ④ 接收反馈 -> {fb_path} | gop_id={gop_id} "
                    f"bitrate={bitrate:.1f}/{target_bitrate:.1f} score={score:.2f}/{target_score:.2f} reward={r:.4f}"
                )
                
                # 记录此 FB 的 score 到 pending，用于下一个 RQ 的验证
                pend["fb_score"] = score
                
                # 记录 reward 和 done
                pend["reward"] = r
                pend["done"] = is_last_gop
                
                # 如果是最后一个 GOP，直接 push（终止步）
                if is_last_gop:
                    # 计算 episode bonus（全局优化信号）
                    episode_bonus = self.rw.compute_episode_bonus()
                    
                    # 将 episode bonus 加到最后一个 GOP 的 reward
                    r_final = r + episode_bonus * 0.3  # 权重 0.3
                    
                    seq = pend["seq"]
                    sca = pend["scalars"]
                    a = pend["a"]
                    seq2 = np.zeros_like(seq)
                    sca2 = np.zeros_like(sca)
                    self.buf.push(seq, sca, a, r_final, seq2, sca2, done=True)
                    self._log(3, f"[Replay] Push terminal: gop_id={gop_id} r_gop={r:.4f} bonus={episode_bonus:.4f} r_total={r_final:.4f}")
                    self.pending.pop(gop_id)
                    self._last_gop_id = None
                    
                    # Episode 结束统计
                    info = self.rw.on_episode_end()
                    self.epoch_episodes += 1
                    self.epoch_total_reward += info['episode_return']
                    self.epoch_total_frames += info['total_frames']
                    
                    self._log(1, f"\n{'='*60}")
                    self._log(1, f"[EPISODE END] Epoch #{self.current_epoch}")
                    self._log(1, f"  GOP 数量: {info['gop_count']}")
                    self._log(1, f"  总帧数: {info['total_frames']}")
                    self._log(1, f"  Episode 回报: {info['episode_return']:+.4f}")
                    self._log(1, f"  平均码率: {info['avg_bitrate']:.2f} ({info['bitrate_saved_pct']:+.1f}%)")
                    self._log(1, f"  平均质量: {info['avg_score']:.2f} ({info['score_diff_pct']:+.1f}%)")
                    self._log(1, f"{'='*60}\n")
                else:
                    # 非终止 GOP：标记为 last_gop_id，等待下一个 RQ 到来时存储转移
                    self._last_gop_id = gop_id
                    
                    # === 新增：检查是否有后续 GOP 已经在 pending 中 ===
                    # 如果 GOP 顺序是：RQ2 -> RQ3 -> FB2 -> FB3
                    # 那么收到 FB2 时，GOP3 已经在 pending 中，可以立即存储转移
                    next_gop_id = gop_id + 1
                    if next_gop_id in self.pending:
                        next_pend = self.pending[next_gop_id]
                        if "seq" in next_pend and "scalars" in next_pend:
                            # 下一个 GOP 的状态已经构建好，可以存储转移
                            self.buf.push(
                                pend["seq"], pend["scalars"], pend["a"],
                                r,
                                next_pend["seq"], next_pend["scalars"],
                                done=False
                            )
                            self._log(3, f"[Replay] Push transition (late FB): gop_id={gop_id} -> {next_gop_id}")
                            self.pending.pop(gop_id)
                            self._last_gop_id = None
                
                # 训练
                self.total_steps += 1
                if self.cfg.mode == "train" and self.total_steps >= self.cfg.start_steps and len(self.buf) >= self.cfg.batch_size:
                    for _ in range(self.cfg.updates_per_step):
                        b = self.buf.sample(self.cfg.batch_size, self.cfg.device)
                        loss_q, eps = self.agent.train_step(b)
                        self.epoch_train_count += 1
                        
                        if self.writer and (self.total_steps % self.cfg.tb_log_interval) == 0:
                            self.writer.add_scalar('Loss/Q', loss_q, self.total_steps)
                        
                        if (self.total_steps % 50) == 0:
                            self._log(2, f"[Train] step={self.total_steps} Lq={loss_q:.4f}")
                
                self._last_action = float(pend["action_qp"])
                
                # 删除处理过的 FB
                try:
                    os.remove(fb_path)
                    self._log(3, f"[GOP][FB] ⑤ 删除反馈 -> {fb_path}")
                except Exception as e:
                    self._log(2, f"[GOP][WARN] 删除 FB 失败: {e}")
                
                progressed = True
            
            if not progressed:
                idle_loops += 1
                if idle_loops * self.cfg.poll_ms >= 1000:
                    self._log(3, f"[GOP][WAIT] 等待 RQ/FB (pending={len(self.pending)})")
                    idle_loops = 0
                    if stop_evt.is_set():
                        break
                time.sleep(self.cfg.poll_ms / 1000.0)
            else:
                idle_loops = 0
        
        # 清理
        self._log(1, f"[Run] GOP RL loop exited.")
        self._cleanup_pending()
    
    def _cleanup_pending(self):
        """
        清理残留的 pending 项
        
        如果编码器退出时仍有未完成的 GOP（已收到 FB 但未构成完整转移），
        将它们作为终止状态存入 Replay Buffer，避免丢失训练数据。
        """
        if len(self.pending) == 0:
            return
        
        pending_ids = sorted(self.pending.keys())
        self._log(1, f"[Run][CLEANUP] 处理 {len(pending_ids)} 个剩余 GOP: {pending_ids}")
        
        for gop_id in pending_ids:
            pend = self.pending[gop_id]
            
            # 只处理已经收到 reward 的 GOP（表示 FB 已处理）
            if "reward" in pend:
                seq = pend["seq"]
                sca = pend["scalars"]
                a = pend["a"]
                r = pend["reward"]
                
                # 使用零状态作为终止状态
                seq2 = np.zeros_like(seq)
                sca2 = np.zeros_like(sca)
                
                # 存入 Replay Buffer 作为终止状态
                self.buf.push(seq, sca, a, r, seq2, sca2, done=True)
                self._log(2, f"[Cleanup] 保存 GOP {gop_id} 为终止状态 (reward={r:.4f})")
                
                # 统计为完成的 episode
                if not pend.get("done", False):
                    # 如果这个 GOP 还没有触发 episode 结束统计，现在触发
                    info = self.rw.on_episode_end()
                    self.epoch_episodes += 1
                    self._log(2, f"[Cleanup] Episode 结束统计: return={info.get('episode_return', 0):.2f}")
            else:
                # 没有 reward 的 GOP（只有 RQ 没有 FB）直接丢弃
                self._log(2, f"[Cleanup] 丢弃未完成 GOP {gop_id} (无 FB)")
        
        self.pending.clear()
        self._log(1, f"[Run][CLEANUP] 清理完成")
    
    def print_epoch_summary(self, epoch_id: int, epoch_total: int, interrupted: bool = False):
        """打印 epoch 统计"""
        self._log(1, f"\n{'#'*60}")
        status = "（已中断）" if interrupted else ""
        self._log(1, f"# EPOCH #{epoch_id}/{epoch_total} 统计{status}")
        self._log(1, f"{'#'*60}")
        
        if self.epoch_episodes == 0:
            self._log(1, "  本 Epoch 未完成任何 Episode")
        else:
            avg_reward = self.epoch_total_reward / self.epoch_episodes
            self._log(1, f"  完成 Episodes: {self.epoch_episodes}")
            self._log(1, f"  总训练步数: {self.total_steps}")
            self._log(1, f"  平均 Episode 回报: {avg_reward:+.4f}")
        
        self._log(1, f"{'#'*60}\n")
        
        # 重置统计
        self.epoch_episodes = 0
        self.epoch_total_reward = 0.0
        self.epoch_total_bits = 0.0
        self.epoch_total_score = 0.0
        self.epoch_total_frames = 0
        self.epoch_train_count = 0
