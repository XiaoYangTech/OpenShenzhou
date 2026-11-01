#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""timestretch.py —— 保音高的时长伸缩（WSOLA），用于复刻引擎 RateAdjustment

引擎行为（实测，见 rea_work/findings_engine.md §34）：
- 语音包 INI 的 VoiceSetting.RateAdjustment=100 时，模型输出即最终音频（逐采样一致）；
- RateAdjustment≠100 时，引擎输出时长 ≈ 模型时长 × 100 / RateAdjustment，
  且音高不变（F0 实测 200.8→201.7 Hz）、包络保持（相关 0.998）——
  即引擎做的是"保音高时间伸缩"（不是重采样、也不是模型内语速预测）。
- 本模块用 WSOLA 近似该伸缩：时长/音高与引擎一致，波形仍有差异
  （引擎具体算法尚未逆向出来；该差异已在 README「已知差异」中说明）。
"""
import numpy as np

FRAME = 960      # 40 ms 分析窗（24 kHz）
SYN_HOP = 240    # 10 ms 合成步长
TOL = 1440       # WSOLA 相似度搜索半径（60 ms）


def wsola(x, ratio, frame=FRAME, syn_hop=SYN_HOP, tol=TOL):
    """把音频 x 伸缩到 len(x)*ratio（ratio<1 变短、>1 变长），保持音高。"""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0 or abs(ratio - 1.0) < 1e-6:
        return x.copy()
    target = int(round(n * ratio))
    if target <= frame:
        idx = np.linspace(0, n - 1, target)
        return np.interp(idx, np.arange(n), x).astype(np.float32)
    ana_hop = max(1, int(round(syn_hop * ratio)))
    win = np.hanning(frame).astype(np.float32)
    out = np.zeros(target + frame, dtype=np.float32)
    wsum = np.zeros(target + frame, dtype=np.float32)
    pos = 0.0
    k = 0
    prev_tail = None
    while True:
        a = int(round(pos))
        if a + frame > n:
            break
        o = k * syn_hop
        if o >= target:
            break
        start = a
        if prev_tail is not None and len(prev_tail) >= 8:
            lo = max(0, a - tol)
            hi = min(n - frame, a + tol)
            if hi > lo:
                p = prev_tail.astype(np.float32)
                npn = float(np.dot(p, p))
                if npn > 1e-9:
                    def _score(cand):
                        v = x[cand:cand + len(p)]
                        return float(np.dot(v, p)) / (np.sqrt(float(np.dot(v, v)) * npn) + 1e-9)
                    # 粗搜（步长 4）+ 细搜（±4 步长 1）
                    best = -2.0
                    coarse = list(range(lo, hi, 4))
                    for cand in coarse:
                        c = _score(cand)
                        if c > best:
                            best = c
                            start = cand
                    for cand in range(max(lo, start - 3), min(hi, start + 4)):
                        c = _score(cand)
                        if c > best:
                            best = c
                            start = cand
        seg = x[start:start + frame]
        out[o:o + frame] += seg * win
        wsum[o:o + frame] += win
        prev_tail = x[start + syn_hop:start + frame]
        pos += ana_hop
        k += 1
    mask = wsum > 1e-6
    out[mask] /= wsum[mask]
    return out[:target]
