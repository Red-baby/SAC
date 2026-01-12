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
        self.fb_max_retries: int = 5
        
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
        
        while not stop_evt.is_set():
            progressed = False
            
            # === 处理 RQ ===
            rq_files = _scan_gop_rq_files(rl_dir)
            if rq_files:
                rq_path = rq_files[0]
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
                
                # 构建状态
                g_state = dict(
                    score_ema=self.rw.score_ema.get(),
                    last_action=getattr(self, "_last_action", self.cfg.default_qp),
                )
                seq, scalars, gop_id, gop_size, target_bitrate, target_score = build_state_from_gop_rq(
                    self.cfg, rq, g_state
                )
                
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
                    if self.fb_read_failures[fb_path] < self.fb_max_retries:
                        continue
                    else:
                        self._log(1, f"[GOP][FB][ERROR] bad fb after retries: {fb_path}")
                        try:
                            os.remove(fb_path)
                            self.fb_read_failures.pop(fb_path, None)
                        except:
                            pass
                        continue
                
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
