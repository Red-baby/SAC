# -*- coding: utf-8 -*-
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, state_seq_shape, state_scalar_dim: int):
        self.capacity = int(capacity)
        C, T = state_seq_shape
        self._seq = np.zeros((capacity, C, T), dtype=np.float32)
        self._sca = np.zeros((capacity, state_scalar_dim), dtype=np.float32)
        self._a   = np.zeros((capacity, 1), dtype=np.float32)
        self._r   = np.zeros((capacity, 1), dtype=np.float32)
        self._seq2= np.zeros((capacity, C, T), dtype=np.float32)
        self._sca2= np.zeros((capacity, state_scalar_dim), dtype=np.float32)
        self._d   = np.zeros((capacity, 1), dtype=np.float32)
        self._n = 0; self._p = 0

    def __len__(self): return self._n

    def push(self, seq, sca, a, r, seq2, sca2, done):
        i = self._p
        self._seq[i]  = seq;    self._sca[i]  = sca
        self._a[i]    = a;      self._r[i]    = float(r)
        self._seq2[i] = seq2;   self._sca2[i] = sca2
        self._d[i]    = float(done)
        self._p = (self._p + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self._n, size=(batch_size,))
        seq  = torch.from_numpy(self._seq[idx]).to(device)
        sca  = torch.from_numpy(self._sca[idx]).to(device)
        a    = torch.from_numpy(self._a[idx]).to(device)
        r    = torch.from_numpy(self._r[idx]).to(device)
        seq2 = torch.from_numpy(self._seq2[idx]).to(device)
        sca2 = torch.from_numpy(self._sca2[idx]).to(device)
        d    = torch.from_numpy(self._d[idx]).to(device)
        return seq, sca, a, r, seq2, sca2, d

    # --- Checkpoint helpers -------------------------------------------------
    def export_state(self):
        """导出 replay buffer 状态用于保存 checkpoint。"""
        if self._n == 0:
            return None
        return {
            "seq": self._seq[:self._n].copy(),
            "sca": self._sca[:self._n].copy(),
            "a": self._a[:self._n].copy(),
            "r": self._r[:self._n].copy(),
            "seq2": self._seq2[:self._n].copy(),
            "sca2": self._sca2[:self._n].copy(),
            "d": self._d[:self._n].copy(),
            "ptr": int(self._p),
            "size": int(self._n),
            "capacity": int(self.capacity),
        }

    def load_state(self, state: dict):
        """从 checkpoint 恢复 replay buffer 状态。"""
        if not state:
            return
        size = int(state.get("size", 0))
        if size <= 0:
            self._n = 0
            self._p = 0
            return
        if size > self.capacity:
            raise ValueError(f"replay buffer size {size} > capacity {self.capacity}")

        def _copy(dst, src_key):
            src = state.get(src_key, None)
            if src is None:
                raise KeyError(f"missing replay buffer field '{src_key}' in checkpoint")
            dst[:size] = src[:size]

        _copy(self._seq, "seq")
        _copy(self._sca, "sca")
        _copy(self._a, "a")
        _copy(self._r, "r")
        _copy(self._seq2, "seq2")
        _copy(self._sca2, "sca2")
        _copy(self._d, "d")

        self._n = size
        ptr = int(state.get("ptr", size % self.capacity))
        self._p = max(0, min(ptr, self.capacity - 1))
