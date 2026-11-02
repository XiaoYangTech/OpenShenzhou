# -*- coding: utf-8 -*-
"""pitchshift.py —— 引擎 SSML `<pitch>` 链路的精确复刻（int16 域，逐采样一致）

逆向自引擎 DLL（v0.2.10）：
  FUN_1800b0784  音频处理器主调度
     · dVar7 = f20/f28  (f20 总速率因子，f28 音高因子)
     · 若 dVar7 ∈ [0.99999, 1.00001]：直通（把 pending 原样追加）
       否则 chulam FUN_1800b0514：PSOLA 以 r=dVar7 做时长伸缩
     · mode==0 时若 f28 != 1：FUN_1800b0264 以 f28 做 12 抽头 int16 多相重采样
  FUN_1800b0264  精确状态机：edi=24000、esi=(int)(24000/f28)，两者一起右移至 ≤0x4000；
                每输出样本用 50 相位 × 12 抽头的 int16 系数表（VA 0x18044b3f0）
                做「(step-rem)/step 与 rem/step 线性插值」的定点滤波，累加溢出饱和。
  FUN_1800afb08  flush：补 2*738 个 0 后继续处理，最终输出截断到
                obj54 + int((i50*f28/f20 + i58)/(f28*f2c) + 0.5)

对引擎进程内 24kHz PCM 验证：pitch=+25% / +50% / -50% 三种情况 **逐采样 maxdiff=0**。
"""
import base64

import numpy as np

from .rateadjust import (
    STEP, MIN_P, MAX_P, _ONE, _TWO, _HALF,
    _detect_period, _crossfade,
)

_PITCH_TAB_B64 = (
    "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"
)

_pitch_tab_cache = None


def _pitch_tab():
    global _pitch_tab_cache
    if _pitch_tab_cache is None:
        raw = base64.b64decode(_PITCH_TAB_B64)
        _pitch_tab_cache = np.frombuffer(raw, dtype='<i2').astype(np.int64)
    return _pitch_tab_cache


def _wrap32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _trunc_div(num, den):
    return num // den if num >= 0 else -((-num) // den)


class _PsolaStream:
    """FUN_1800b0514 的流式版本（不截断；按块喂入，输出追加到 self.out）"""

    def __init__(self, rate):
        self.rf = np.float32(rate)
        self.buf = []
        self.pos = 0
        self.carry = 0
        self.hist = [0, 0]
        self.out = []

    def append(self, samples):
        self.buf.extend(int(v) for v in samples)

    def process(self):
        rf = self.rf
        blen = len(self.buf)
        while self.pos + STEP <= blen:
            if self.carry < 1:
                p = _detect_period(self.buf, self.pos, self.hist)
                if p <= 0:
                    break
                if rf > _ONE:
                    f10 = _ONE / (rf - _ONE)
                    if rf < _TWO:
                        self.carry = int((_TWO - rf) * np.float32(p) * f10)
                        olen = p
                    else:
                        olen = int(f10 * np.float32(p))
                        self.carry = 0
                    if self.pos + p + max(p, olen) > blen:
                        break
                    a = self.buf[self.pos:self.pos + p]
                    b = self.buf[self.pos + p:self.pos + p + max(p, olen)]
                    self.out.extend(_crossfade(a, b, olen))
                    self.pos += olen + p
                else:
                    f10 = _ONE / (_ONE - rf)
                    if rf >= _HALF:
                        self.carry = int((rf + rf - _ONE) * np.float32(p) * f10)
                        olen = p
                    else:
                        olen = int(np.float32(p) * rf * f10)
                    if self.pos + 2 * p > blen:
                        break
                    a = self.buf[self.pos:self.pos + p]
                    b = self.buf[self.pos + p:self.pos + 2 * p]
                    self.out.extend(a)
                    self.out.extend(_crossfade(b, a, olen))
                    self.pos += olen
            else:
                m = min(STEP, self.carry)
                if self.pos + m > blen:
                    break
                self.out.extend(self.buf[self.pos:self.pos + m])
                self.pos += m
                self.carry -= m
        if self.pos > 0:
            del self.buf[:self.pos]
            self.pos = 0


class _Resampler264:
    """FUN_1800b0264：精确状态机（edi=输入率、esi=输出率、12 抽头、50 相位）"""

    def __init__(self, f28, sr=24000):
        self.edi = sr
        self.esi = int(np.float32(sr) / np.float32(f28))
        while self.esi > 0x4000 or self.edi > 0x4000:
            self.esi >>= 1
            self.edi >>= 1
        self.inbuf = []
        self.phase = 0
        self.outcnt = 0
        self.out = []

    def append(self, samples):
        self.inbuf.extend(int(v) for v in samples)

    def process(self):
        tab = _pitch_tab()
        m = len(self.inbuf)
        if m - 12 <= 0:
            return
        limit = m - 12
        consumed = 0
        while True:
            if (self.phase + 1) * self.esi > self.outcnt * self.edi:
                diff = (self.phase + 1) * self.esi - self.edi * self.outcnt
                phase_num = (diff - 1) * 50
                k = phase_num // self.esi
                rem = phase_num - k * self.esi
                base = consumed
                acc = 0
                ov = 0
                for t in range(12):
                    c0 = int(tab[t * 50 + k])
                    c1 = int(tab[t * 50 + k + 1])
                    num = ((self.esi - rem) * c0 + rem * c1) * 2
                    coeff = _trunc_div(num, self.esi)
                    prod = _wrap32(coeff * self.inbuf[base + t])
                    prev = acc
                    acc = _wrap32(prev + prod)
                    sp = 1 if prod >= 0 else -1
                    sprev = 1 if prev >= 0 else -1
                    if sp == sprev and (1 if acc >= 0 else -1) != sprev:
                        ov += sprev
                if ov > 0:
                    v = 32767
                elif ov < 0:
                    v = -32768
                else:
                    v = (acc >> 16) & 0xFFFF
                    v = v - 0x10000 if v >= 0x8000 else v
                self.out.append(v)
                self.outcnt += 1
            else:
                self.phase += 1
                consumed += 1
                if self.phase == self.edi:
                    self.phase = 0
                    self.outcnt = 0
                if consumed >= limit:
                    break
        del self.inbuf[:consumed]


def pitch_shift_int16(x, f28, f20=1.0, f2c=1.0, sr=24000):
    """引擎音高处理（int16 域）：返回 np.int16。

    x: 自然音频 int16；f28: 音高因子；f20: 总速率因子；f2c: 引擎 +0x2c 字段（通常 1.0）。
    处理顺序与 b0784 一致：PSOLA(f20/f28) → b0264(f28) → afb08 flush 截断。
    """
    x = np.asarray(x, dtype=np.int64)
    f20 = np.float32(f20)
    f28 = np.float32(f28)
    f2c = np.float32(f2c)
    dvar7 = float(np.float32(f20 / f28))
    append_path = 0.99999 <= dvar7 <= 1.00001
    ps = None
    if append_path:
        p1 = [int(v) for v in x]
        pending = []
    else:
        ps = _PsolaStream(dvar7)
        ps.append(x)
        ps.process()
        p1 = ps.out
        ps.out = []
        pending = ps.buf
    rs = _Resampler264(f28, sr)
    rs.append(p1)
    rs.process()
    q1 = rs.out
    rs.out = []
    if not append_path:
        i50 = len(pending)
        i58 = len(rs.inbuf)
        # afb08: target = obj54 + int((i50*f28/f20 + i58)/(f28*f2c) + 0.5)
        num = np.float32(i50) * f28 / f20 + np.float32(i58)
        den = f28 * f2c
        target = int(np.float32(num / den) + _HALF)
        ps.buf.extend([0] * (2 * STEP))
        ps.process()
        p2 = ps.out
        rs.append(p2)
        rs.process()
        q2 = rs.out
        q2 = q2[:max(0, target)]
    else:
        q2 = []
    out = np.array(q1 + q2, dtype=np.int64)
    return np.clip(out, -32768, 32767).astype(np.int16)


def adjust_pitch_float(audio, f28, f20=1.0, f2c=1.0):
    """float32 [-1,1] 音频 → 引擎音高处理后的 float32（int16 域精确复刻）"""
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    x = np.trunc(a * np.float32(32767.0)).astype(np.int32)
    y = pitch_shift_int16(x, f28, f20=f20, f2c=f2c)
    return y.astype(np.float32) / np.float32(32767.0)
