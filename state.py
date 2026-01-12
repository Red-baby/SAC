# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple, List
from utils import pad_or_trim, circular_pad, log_process


def _smart_downsample(seq: np.ndarray, temporal: np.ndarray, target_T: int) -> np.ndarray:
    """
    智能下采样：保留 I/P 帧 (temporal_level <= 1)，对 B 帧区间使用平均池化
    
    策略：
    1. 完整保留 temporal_level=0 (I帧) 和 temporal_level=1 (P帧)
    2. 对连续的 B 帧区间 (temporal_level >= 2) 使用平均池化
    3. 如果结果仍超过 target_T，对 B 帧区间进一步池化
    
    Args:
        seq: [C, T] 序列特征
        temporal: [T] 时域等级数组（未归一化）
        target_T: 目标序列长度
        
    Returns:
        downsampled: [C, T'] 下采样后的序列，T' <= target_T
    """
    C, T = seq.shape
    if T <= target_T:
        return seq
    
    # 找出所有关键帧（I/P 帧）的位置
    key_frame_mask = temporal <= 1.0  # temporal_level 0 或 1
    key_frame_indices = np.where(key_frame_mask)[0]
    
    # 如果关键帧数量已经超过目标，只保留关键帧并均匀采样
    if len(key_frame_indices) >= target_T:
        sample_indices = np.linspace(0, len(key_frame_indices) - 1, target_T, dtype=np.int32)
        return seq[:, key_frame_indices[sample_indices]]
    
    # 构建结果列表
    result_frames: List[np.ndarray] = []
    result_count = 0
    
    # 计算剩余可用于 B 帧的位置数
    remaining_slots = target_T - len(key_frame_indices)
    
    # 将序列分割成：关键帧 + B帧区间
    i = 0
    b_frame_segments: List[Tuple[int, int]] = []  # (start, end) 区间
    
    while i < T:
        if key_frame_mask[i]:
            # 关键帧：直接保留
            result_frames.append(seq[:, i:i+1])
            result_count += 1
            i += 1
        else:
            # B 帧区间：找到连续的 B 帧
            start = i
            while i < T and not key_frame_mask[i]:
                i += 1
            end = i
            b_frame_segments.append((start, end))
    
    # 对 B 帧区间进行池化
    total_b_frames = sum(end - start for start, end in b_frame_segments)
    
    if total_b_frames > 0 and remaining_slots > 0:
        # 计算每个 B 帧区间应该保留多少帧
        pooled_b_frames: List[np.ndarray] = []
        
        for start, end in b_frame_segments:
            segment_len = end - start
            # 按比例分配 slots
            segment_slots = max(1, int(remaining_slots * segment_len / total_b_frames))
            
            if segment_slots >= segment_len:
                # 不需要池化，全部保留
                pooled_b_frames.append(seq[:, start:end])
            else:
                # 需要池化：将区间划分成 segment_slots 份，每份取平均
                pooled = np.zeros((C, segment_slots), dtype=np.float32)
                indices = np.linspace(start, end, segment_slots + 1, dtype=np.int32)
                for j in range(segment_slots):
                    pooled[:, j] = np.mean(seq[:, indices[j]:indices[j+1]], axis=1)
                pooled_b_frames.append(pooled)
        
        # 重新构建结果：按原始顺序交错排列关键帧和池化后的 B 帧
        result_frames = []
        b_seg_idx = 0
        i = 0
        while i < T:
            if key_frame_mask[i]:
                result_frames.append(seq[:, i:i+1])
                i += 1
            else:
                # 添加池化后的 B 帧区间
                if b_seg_idx < len(pooled_b_frames):
                    result_frames.append(pooled_b_frames[b_seg_idx])
                    b_seg_idx += 1
                # 跳过原始的 B 帧区间
                while i < T and not key_frame_mask[i]:
                    i += 1
    
    # 合并所有帧
    if len(result_frames) == 0:
        # 极端情况：没有任何帧，返回均匀采样
        indices = np.linspace(0, T - 1, target_T, dtype=np.int32)
        return seq[:, indices]
    
    result = np.concatenate(result_frames, axis=1)
    
    # 如果结果仍然超过目标长度，进行最终的均匀截断
    if result.shape[1] > target_T:
        indices = np.linspace(0, result.shape[1] - 1, target_T, dtype=np.int32)
        result = result[:, indices]
    
    return result


def build_state_from_gop_rq(cfg, rq: dict, g_state: dict) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """
    从 GOP 级别的 RQ 构建状态（用于新的 GOP 级别处理模式）
    
    RQ 格式:
    {
        "gop_id": 0,                  # 当前 GOP ID
        "rl_gop_size": 225,           # GOP 帧数
        "encoded_frames": 0,          # 已编码帧数
        "bitrate_ration": 0.0,        # 已编码帧的码率与目标比值
        "encoded_score": 0.0,         # 已编码帧的平均质量
        "encoded_comp": 0.0,          # 已编码帧的平均复杂度
        "last_bitrate_ratio": 0.0,    # 上一个 GOP 的码率比值
        "last_score": 0.0,            # 上一个 GOP 的平均质量
        "last_comp": 0.0,             # 上一个 GOP 的平均复杂度
        "lastqpbase": 0,              # 上一个 GOP 的 qpbase
        "target_score": 40.5,         # 目标质量
        "target_bitrate": 2125,       # 目标比特率
        "qps": [...],                 # pass1 每帧 QP
        "poise": [...],               # 每帧复杂度指标
        "comp": [...],                # 每帧复杂度指标
        "score_1pass": [...],         # pass1 每帧质量
        "bits_1pass": [...],          # pass1 每帧比特
        "temporal_level": [...]       # 每帧时域等级
    }
    
    Returns:
        seq: [C=6, T] 序列特征 [poise, comp_log, bits_log, score_1pass_norm, qps_norm, temporal_norm]
        scalars: [11] 标量特征 (包含 is_first_gop)
        gop_id: GOP ID
        gop_size: GOP 大小
        target_bitrate: 目标比特率
        target_score: 目标质量
    """
    # 获取基本参数
    gop_size_std = int(getattr(cfg, "gop_size_standard", 225))
    gop_id = int(rq.get("gop_id", 0))
    gop_size = int(rq.get("rl_gop_size", gop_size_std))
    gop_size = max(1, gop_size)
    T = gop_size_std  # 固定序列长度
    
    # 目标值
    target_score = float(rq.get("target_score", 40.0))
    target_bitrate = float(rq.get("target_bitrate", 2000.0))
    
    # === 序列特征处理 ===
    # 使用循环 padding 填充不足的 GOP
    poise_raw = rq.get("poise", [])
    comp_raw = rq.get("comp", [])
    bits_1pass_raw = rq.get("bits_1pass", [])
    score_1pass_raw = rq.get("score_1pass", [])
    qps_raw = rq.get("qps", [])
    temporal_raw = rq.get("temporal_level", [])
    
    # 循环 padding 到标准 GOP 大小
    poise = circular_pad(poise_raw, T, 1.5).astype(np.float32)
    comp = circular_pad(comp_raw, T, 0.0).astype(np.float32)
    bits_1pass = circular_pad(bits_1pass_raw, T, 0.0).astype(np.float32)
    score_1pass = circular_pad(score_1pass_raw, T, 40.0).astype(np.float32)
    qps = circular_pad(qps_raw, T, 127.0).astype(np.float32)
    temporal = circular_pad(temporal_raw, T, 6.0).astype(np.float32)
    
    # 对大数值进行 log 处理
    # comp: 可能有非常大的值（如 800000），使用 log1p 处理
    comp_log = np.log1p(np.maximum(comp, 0.0)).astype(np.float32)
    # 归一化到合理范围 (log1p(800000) ≈ 13.6)
    comp_log = (comp_log / 15.0).astype(np.float32)
    
    # bits_1pass: 可能有非常大的值，使用 log1p 处理
    bits_log = np.log1p(np.maximum(bits_1pass, 0.0)).astype(np.float32)
    # 归一化到合理范围 (log1p(500000) ≈ 13.1)
    bits_log = (bits_log / 15.0).astype(np.float32)
    
    # score_1pass: 质量分数通常在 30-60 范围，除以 100 归一化
    score_1pass_norm = (score_1pass / 100.0).astype(np.float32)
    
    # qps: QP 值通常在 30-230 范围，除以 256 归一化
    qps_norm = (qps / 256.0).astype(np.float32)
    
    # temporal_level: 时域等级通常在 0-6 范围，除以 6 归一化
    temporal_norm = (temporal / 6.0).astype(np.float32)
    
    # poise: 通常在 1.0-2.0 范围，直接使用或轻微归一化
    poise_norm = (poise / 2.0).astype(np.float32)
    
    # 构建序列特征 [C=6, T]
    seq = np.stack([
        poise_norm,       # 复杂度指标 1
        comp_log,         # 复杂度指标 2（log 处理后）
        bits_log,         # pass1 比特（log 处理后）
        score_1pass_norm, # pass1 质量（归一化）
        qps_norm,         # pass1 QP（归一化）
        temporal_norm     # 时域等级（归一化）
    ], axis=0).astype(np.float32)
    
    # === 智能序列下采样 ===
    # 保留 I/P 帧 (temporal_level <= 1)，对 B 帧区间使用平均池化
    enable_downsample = bool(getattr(cfg, "enable_smart_downsample", True))
    target_T = int(getattr(cfg, "seq_target_T", 64))
    if enable_downsample and seq.shape[1] > target_T:
        seq = _smart_downsample(seq, temporal, target_T)


    
    # === 标量特征处理 ===
    default_qp = float(getattr(cfg, "default_qp", 127))
    
    # 获取当前 GOP 的编码状态
    encoded_frames = int(rq.get("encoded_frames", 0))
    bitrate_ratio = float(rq.get("bitrate_ration", 0.0))  # 注意拼写是 "ration" 不是 "ratio"
    encoded_score = float(rq.get("encoded_score", 0.0))
    encoded_comp = float(rq.get("encoded_comp", 0.0))
    
    # 获取上一个 GOP 的状态
    last_bitrate_ratio = float(rq.get("last_bitrate_ratio", 0.0))
    last_score = float(rq.get("last_score", 0.0))
    last_comp = float(rq.get("last_comp", 0.0))
    lastqpbase = float(rq.get("lastqpbase", 0))
    
    # === 第一个 GOP 特殊处理 ===
    is_first_gop = (gop_id == 0)
    
    if is_first_gop:
        # 第一个 GOP 没有前面的编码结果，使用合理的默认值
        last_bitrate_ratio = 1.0  # 假设上一个 GOP 刚好达到目标码率
        last_score = target_score  # 假设上一个 GOP 质量等于目标
        lastqpbase = default_qp    # 使用默认 QP
        
        # 计算平均复杂度作为 last_comp 的默认值
        if encoded_comp > 0:
            last_comp = encoded_comp
        elif len(comp_raw) > 0:
            last_comp = float(np.mean(comp_raw))
        else:
            last_comp = 100000.0  # 默认复杂度
    
    # 计算归一化的标量特征
    gop_progress = float(encoded_frames) / float(max(gop_size, 1))
    
    # 对复杂度使用 log 处理
    encoded_comp_log = np.log1p(max(encoded_comp, 0.0)) / 15.0 if encoded_comp > 0 else 0.0
    last_comp_log = np.log1p(max(last_comp, 0.0)) / 15.0
    
    # 构建标量特征 [11 维] - 添加 is_first_gop 明确标记
    scalars = np.array([
        gop_progress,                          # 编码进度 [0, 1]
        bitrate_ratio,                         # 当前码率比值 [0, 2+]
        encoded_score / 100.0,                 # 已编码质量 [0, 1]
        encoded_comp_log,                      # 已编码复杂度 (log) [0, 1]
        last_bitrate_ratio,                    # 上一 GOP 码率比值 [0, 2+]
        last_score / 100.0,                    # 上一 GOP 质量 [0, 1]
        last_comp_log,                         # 上一 GOP 复杂度 (log) [0, 1]
        lastqpbase / 256.0,                    # 上一 GOP QP [0, 1]
        target_score / 100.0,                  # 目标质量 [0, 1]
        np.log1p(target_bitrate) / 10.0,       # 目标码率 (log) [0, 1.5]
        float(is_first_gop),                   # 是否为第一个 GOP [0, 1] - 明确信号
    ], dtype=np.float32)
    
    return seq, scalars, gop_id, gop_size, target_bitrate, target_score


def build_state_from_rq(cfg, rq: dict, g_state: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    """
    原始的 Mini-GOP 级别状态构建函数（保持向后兼容）
    
    Returns:
      seq: [C=6, T] with order [poise, comp, rdcost, score_target, bit_target, qp/256]
      scalars: [9]  = [d_score_alloc, d_score_ratio, d_score_gop_alloc,
                        d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
                        mg_pos_abs, score_ema/100, last_action/(action_max-action_min)]
      qps: [mg_size] 当前 minigop 内每一帧的 qp（原始值，未归一化）
      temporal_level: [mg_size] 当前 minigop 内每一帧的时域等级 [1,2,3,4,6]
      mg_id, mg_size, bits_alloc(gop), score_alloc(gop)
    """
    T = int(getattr(cfg, "frames_per_mg", 16))
    poise = pad_or_trim(rq.get("poise", []), T, 0.0).astype(np.float32)
    comp  = pad_or_trim(rq.get("comp",  []), T, 0.0).astype(np.float32)
    rdc   = pad_or_trim(rq.get("rdcost",[]), T, 0.0).astype(np.float32)
    score_tgt = pad_or_trim(rq.get("score", []), T, 0.0).astype(np.float32)  # score_target
    if len(rq.get("score_target", [])) > 0:
        score_tgt = pad_or_trim(rq.get("score_target", []), T, 0.0).astype(np.float32)
    bit_tgt   = pad_or_trim(rq.get("bits",   []), T, 0.0).astype(np.float32) # bits_target
    if len(rq.get("bit_target", [])) > 0:
        bit_tgt = pad_or_trim(rq.get("bit_target", []), T, 0.0).astype(np.float32)

    comp  = log_process(comp,  getattr(cfg, "apply_log_comp", True),   getattr(cfg, "robust_scale_seq", True), getattr(cfg, "robust_clip", 5.0))
    rdc   = log_process(rdc,   getattr(cfg, "apply_log_rdcost", True), getattr(cfg, "robust_scale_seq", True), getattr(cfg, "robust_clip", 5.0))
    bit_tgt = log_process(bit_tgt, getattr(cfg, "apply_log_bit_target", True), getattr(cfg, "robust_scale_seq", True), getattr(cfg, "robust_clip", 5.0))
    score_tgt = (score_tgt / 100.0).astype(np.float32)

    # 读取 qps（当前 minigop 内每一帧的 qp）
    qps_raw = rq.get("qps", [])
    mg_id = int(rq.get("mg_id", 0))
    mg_size = int(rq.get("mg_size", T))
    mg_size = max(1, mg_size)
    
    # 确保 qps 长度为 T（与 seq 对齐），不足则用最后一个值填充，超出则截断
    if len(qps_raw) == 0:
        # 如果没有 qps，尝试从 baseqp 或 base_q 获取（向后兼容）
        baseqp_fallback = float(rq.get("baseqp", rq.get("base_q", 0.0)))
        qps = np.full(T, baseqp_fallback, dtype=np.float32)
    else:
        qps = pad_or_trim(qps_raw, T, qps_raw[-1] if len(qps_raw) > 0 else 0.0).astype(np.float32)
    
    # 归一化 qps（除以 256.0，与原来的 baseqp/256.0 保持一致的范围）
    qps_norm = (qps / 256.0).astype(np.float32)
    
    # 将 qps 作为序列特征添加到 seq 中（第 6 个通道）
    seq = np.stack([poise, comp, rdc, score_tgt, bit_tgt, qps_norm], axis=0).astype(np.float32)
    
    d_score_ratio = float(rq.get("score_ratio", rq.get("d_score_ratio", 1.0)))
    d_bits_ratio  = float(rq.get("bits_ratio",  rq.get("d_bits_ratio", 1.0)))
    d_score_alloc = float(rq.get("score_alloc", rq.get("d_score_alloc", 0.0)))
    d_score_gop_alloc = float(rq.get("score_gop_alloc", rq.get("d_score_gop_alloc", d_score_alloc)))
    i_bits_alloc = float(rq.get("bits_alloc", rq.get("i_bits_alloc", 0.0)))
    i_bits_gop_alloc = float(rq.get("bits_gop_alloc", rq.get("i_bits_gop_alloc", i_bits_alloc)))

    mg_pos_abs = float(max(0, mg_id))
    action_min = float(getattr(cfg, "action_min", 0.0))
    action_max = float(getattr(cfg, "action_max", action_min + 1.0))
    action_range = max(1.0, action_max - action_min)
    last_action = float(g_state.get("last_action", action_min))
    last_action_norm = (last_action - action_min) / action_range

    # scalars 不再包含 qp（已作为序列特征），保持 9 维
    scalars = np.array([
        d_score_alloc, d_score_ratio, d_score_gop_alloc,
        d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
        mg_pos_abs,    g_state.get("score_ema",0.0)/100.0, last_action_norm
    ], dtype=np.float32)
    
    # 返回原始 qps（未归一化，长度为 mg_size，用于后续处理）
    if len(qps_raw) > 0:
        qps_original = pad_or_trim(qps_raw, mg_size, qps_raw[-1] if len(qps_raw) > 0 else 0.0).astype(np.float32)
    else:
        baseqp_fallback = float(rq.get("baseqp", rq.get("base_q", 0.0)))
        qps_original = np.full(mg_size, baseqp_fallback, dtype=np.float32)
    
    # 读取 temporal_level（时域等级）
    temporal_level_raw = rq.get("temporal_level", [])
    if len(temporal_level_raw) > 0:
        # 确保长度为 mg_size，不足则用最后一个值填充，超出则截断
        temporal_level = pad_or_trim(temporal_level_raw, mg_size, temporal_level_raw[-1] if len(temporal_level_raw) > 0 else 6).astype(np.int32)
    else:
        # 如果没有提供，默认填充为 6
        temporal_level = np.full(mg_size, 6, dtype=np.int32)

    bits_alloc = i_bits_gop_alloc if i_bits_gop_alloc > 0 else i_bits_alloc
    score_alloc = d_score_gop_alloc if d_score_gop_alloc > 0 else d_score_alloc

    return seq, scalars, qps_original, temporal_level, mg_id, mg_size, bits_alloc, score_alloc
