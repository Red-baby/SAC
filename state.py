# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple
from utils import pad_or_trim, log_process

def build_state_from_rq(cfg, rq: dict, g_state: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    """
    Returns:
      seq: [C=6, T] with order [poise, comp, rdcost, score_target, bit_target, q_val/256]
      scalars: [9]  = [d_score_alloc, d_score_ratio, d_score_gop_alloc,
                        d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
                        mg_pos_abs, score_ema/100, last_delta/delta_qp_max]
      q_vals: [mg_size] 当前 minigop 内每一帧的 q_val（原始值，未归一化）
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

    # 读取 q_vals（当前 minigop 内每一帧的 q_val）
    q_vals_raw = rq.get("q_vals", [])
    mg_id = int(rq.get("mg_id", 0))
    mg_size = int(rq.get("mg_size", T))
    mg_size = max(1, mg_size)
    
    # 确保 q_vals 长度为 T（与 seq 对齐），不足则用最后一个值填充，超出则截断
    if len(q_vals_raw) == 0:
        # 如果没有 q_vals，尝试从 baseqp 或 base_q 获取（向后兼容）
        baseqp_fallback = float(rq.get("baseqp", rq.get("base_q", 0.0)))
        q_vals = np.full(T, baseqp_fallback, dtype=np.float32)
    else:
        q_vals = pad_or_trim(q_vals_raw, T, q_vals_raw[-1] if len(q_vals_raw) > 0 else 0.0).astype(np.float32)
    
    # 归一化 q_vals（除以 256.0，与原来的 baseqp/256.0 保持一致的范围）
    q_vals_norm = (q_vals / 256.0).astype(np.float32)
    
    # 将 q_vals 作为序列特征添加到 seq 中（第 6 个通道）
    seq = np.stack([poise, comp, rdc, score_tgt, bit_tgt, q_vals_norm], axis=0).astype(np.float32)
    
    d_score_ratio = float(rq.get("score_ratio", rq.get("d_score_ratio", 1.0)))
    d_bits_ratio  = float(rq.get("bits_ratio",  rq.get("d_bits_ratio", 1.0)))
    d_score_alloc = float(rq.get("score_alloc", rq.get("d_score_alloc", 0.0)))
    d_score_gop_alloc = float(rq.get("score_gop_alloc", rq.get("d_score_gop_alloc", d_score_alloc)))
    i_bits_alloc = float(rq.get("bits_alloc", rq.get("i_bits_alloc", 0.0)))
    i_bits_gop_alloc = float(rq.get("bits_gop_alloc", rq.get("i_bits_gop_alloc", i_bits_alloc)))

    mg_pos_abs = float(max(0, mg_id))

    # scalars 不再包含 q_val（已作为序列特征），保持 9 维
    scalars = np.array([
        d_score_alloc, d_score_ratio, d_score_gop_alloc,
        d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
        mg_pos_abs,    g_state.get("score_ema",0.0)/100.0, g_state.get("last_delta",0.0)/max(1.0, getattr(cfg, "delta_qp_max", 10))
    ], dtype=np.float32)
    
    # 返回原始 q_vals（未归一化，长度为 mg_size，用于后续处理）
    if len(q_vals_raw) > 0:
        q_vals_original = pad_or_trim(q_vals_raw, mg_size, q_vals_raw[-1] if len(q_vals_raw) > 0 else 0.0).astype(np.float32)
    else:
        baseqp_fallback = float(rq.get("baseqp", rq.get("base_q", 0.0)))
        q_vals_original = np.full(mg_size, baseqp_fallback, dtype=np.float32)

    bits_alloc = i_bits_gop_alloc if i_bits_gop_alloc > 0 else i_bits_alloc
    score_alloc = d_score_gop_alloc if d_score_gop_alloc > 0 else d_score_alloc

    return seq, scalars, q_vals_original, mg_id, mg_size, bits_alloc, score_alloc
