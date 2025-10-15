# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple
from utils import pad_or_trim, log_process

def build_state_from_rq(cfg, rq: dict, g_state: dict) -> Tuple[np.ndarray, np.ndarray, int, int, int, float, float]:
    """
    Returns:
      seq: [C=5, T] with order [poise, comp, rdcost, score_target, bit_target]
      scalars: [10]  = [d_score_alloc, d_score_ratio, d_score_gop_alloc,
                         d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
                         baseqp/256, mg_pos_abs, score_ema/100, last_delta/delta_qp_max]
      baseqp, mg_id, mg_size, bits_alloc(gop), score_alloc(gop)
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

    seq = np.stack([poise, comp, rdc, score_tgt, bit_tgt], axis=0).astype(np.float32)

    baseqp = int(rq.get("baseqp", rq.get("base_q", 0)))
    d_score_ratio = float(rq.get("score_ratio", rq.get("d_score_ratio", 1.0)))
    d_bits_ratio  = float(rq.get("bits_ratio",  rq.get("d_bits_ratio", 1.0)))
    d_score_alloc = float(rq.get("score_alloc", rq.get("d_score_alloc", 0.0)))
    d_score_gop_alloc = float(rq.get("score_gop_alloc", rq.get("d_score_gop_alloc", d_score_alloc)))
    i_bits_alloc = float(rq.get("bits_alloc", rq.get("i_bits_alloc", 0.0)))
    i_bits_gop_alloc = float(rq.get("bits_gop_alloc", rq.get("i_bits_gop_alloc", i_bits_alloc)))
    mg_id = int(rq.get("mg_id", 0))
    mg_size = int(rq.get("mg_size", T))
    mg_size = max(1, mg_size)

    mg_pos_abs = float(max(0, mg_id))

    scalars = np.array([
        d_score_alloc, d_score_ratio, d_score_gop_alloc,
        d_bits_ratio,  i_bits_alloc,  i_bits_gop_alloc,
        baseqp/256.0,  mg_pos_abs,    g_state.get("score_ema",0.0)/100.0, g_state.get("last_delta",0.0)/max(1.0, getattr(cfg, "delta_qp_max", 10))
    ], dtype=np.float32)

    bits_alloc = i_bits_gop_alloc if i_bits_gop_alloc > 0 else i_bits_alloc
    score_alloc = d_score_gop_alloc if d_score_gop_alloc > 0 else d_score_alloc

    return seq, scalars, baseqp, mg_id, mg_size, bits_alloc, score_alloc
