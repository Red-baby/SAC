# -*- coding: utf-8 -*-
import os, json, time, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def now_ms() -> int:
    return int(time.time() * 1000)

def safe_read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_write_json_atomic(path: str, obj):
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)

def pad_or_trim(arr, T, fill=0.0):
    arr = list(arr) if arr is not None else []
    if len(arr) >= T:
        return np.array(arr[:T], dtype=np.float32)
    out = np.full((T,), fill, dtype=np.float32)
    if len(arr) > 0:
        out[:len(arr)] = np.array(arr, dtype=np.float32)
    return out

def robust_standardize(x: np.ndarray, clip: float = 5.0):
    x = x.astype(np.float32)
    med = np.median(x)
    q75 = np.percentile(x, 75.0); q25 = np.percentile(x, 25.0)
    iqr = max(q75 - q25, 1e-6)
    y = (x - med) / iqr
    if clip is not None and clip > 0:
        y = np.clip(y, -clip, +clip)
    return y.astype(np.float32)

def log_process(arr, do_log=True, robust=True, clip=5.0):
    x = np.asarray(arr, dtype=np.float32)
    if do_log:
        x = np.log1p(np.maximum(x, 0.0))
    if robust:
        x = robust_standardize(x, clip=clip)
    return x

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta; self.val = None
    def update(self, x: float):
        if self.val is None: self.val = float(x)
        else: self.val = self.beta*self.val + (1-self.beta)*float(x)
        return self.val
    def get(self): return 0.0 if self.val is None else float(self.val)
