# -*- coding: utf-8 -*-
"""rateadjust.py —— 引擎 RateAdjustment 的精确复刻（TD-PSOLA 式基音同步重叠拼接）

逆向自引擎 DLL（v0.2.10，SHA256 83a48ebc…）：
  FUN_1800b0514      核心处理循环：按基音周期复制/丢弃 + 线性交叠
  FUN_1800afcd8/afbd4 SAD 归一化基音周期检测（含历史平滑）
  FUN_1800afea0      线性交叠：dst[i] = (A[i]*(n-i) + B[i]*i)/n（int16 整数除法）
  FUN_1800b0784      分块调度：比率 1.0 直通，否则进入 FUN_1800b0514

实测参数（24 kHz 单声道）：min_period=60、max_period=369、step=738。
处理在 int16 域进行（引擎声码器输出 int16），因此本模块：

    adjust_float(audio_float32, rate) -> float32

* rate: RateAdjustment/100（1.1 表示 110% 语速）
* rate > 1（压缩）：每个基音周期输出 1 个周期与下一段交叠，间隔性复制 (2r-1)/(r-1) 个周期
* rate < 1（拉伸）：先复制一个周期，再按 (2r-1)/(1-r) 个周期做交叠，随后原样复制

该实现对 1.5（ja 90 → 0.9、es/en-GB/en-IN 95 → 0.95、de 110 → 1.1、测试 1.5）
均与引擎输出逐采样一致（corr=1.0、maxdiff=0，见 rea_work/findings_engine.md §43）。
"""
import numpy as np

MIN_P = 60
MAX_P = 369
STEP = 738

_ONE = np.float32(1.0)
_TWO = np.float32(2.0)
_HALF = np.float32(0.5)


def _detect_period(x, pos, hist):
    """FUN_1800afbd4 + FUN_1800afcd8：SAD 归一化基音检测（int16，含平滑）"""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.int32)
    n = len(x)
    sad_best = 0
    best_l = 0
    u3 = 0
    u4 = 0xFF
    for lag in range(MIN_P, MAX_P + 1):
        if pos + 2 * lag > n:
            break
        sad = int(np.abs(x[pos:pos + lag] - x[pos + lag:pos + 2 * lag]).sum(dtype=np.int64))
        if best_l == 0 or sad * best_l < lag * sad_best:
            sad_best = sad
            best_l = lag
        cand = lag
        if sad * u4 <= lag * u3:
            cand = u4
            sad = u3
        u3 = sad
        u4 = cand
    if best_l == 0:
        return 0
    res8 = sad_best // best_l
    res20 = (u3 // u4) if u4 else 0
    period = best_l
    prev_p, prev_res = hist
    if res8 != 0 and prev_p != 0:
        if not (res8 * 3 < res20 or res8 * 2 <= prev_res * 3):
            period = prev_p
    hist[0] = best_l
    hist[1] = res8
    return period


def _crossfade(a, b, count):
    """FUN_1800afea0：dst[i] = (a[i]*(count-i) + b[i]*i)/count，C 整数除法（向零截断）"""
    count = int(count)
    if count <= 0:
        return np.zeros(0, dtype=np.int32)
    a = np.asarray(a[:count], dtype=np.int64)
    b = np.asarray(b[:count], dtype=np.int64)
    w = np.arange(count, dtype=np.int64)
    num = a * (count - w) + b * w
    val = np.where(num >= 0, num // count, -((-num) // count))
    return np.clip(val, -32768, 32767).astype(np.int32)


def engine_rate_factor(pct):
    """引擎速率因子：float32(percent * 0.01f)——注意不是 percent/100.0 的双精度除法。

    实测：RateAdjustment=85 → f20=0.8499999642372131（= float32(85*0.01f)），
    而 float32(0.85) 是 0.85000002384；两者会让 carry 差 1 导致整段相位漂移。
    """
    return float(np.float32(np.float32(pct) * np.float32(0.01)))


def adjust_int16(x, rate):
    """引擎 RateAdjustment 的 int16 复刻；返回 np.int16"""
    buf = [int(v) for v in np.asarray(x, dtype=np.int32)]
    out = []
    pos = 0
    carry = 0
    hist = [0, 0]
    r = float(rate)
    rf = np.float32(rate)

    def process():
        nonlocal pos, carry
        blen = len(buf)
        while pos + STEP <= blen:
            if carry < 1:
                p = _detect_period(buf, pos, hist)
                if p <= 0:
                    break
                if rf > _ONE:  # 压缩
                    f10 = _ONE / (rf - _ONE)
                    if rf < _TWO:
                        carry = int((_TWO - rf) * np.float32(p) * f10)
                        olen = p
                    else:
                        olen = int(f10 * np.float32(p))
                        carry = 0
                    if pos + p + max(p, olen) > blen:
                        break
                    a = buf[pos:pos + p]
                    b = buf[pos + p:pos + p + max(p, olen)]
                    out.extend(_crossfade(a, b, olen))
                    pos += olen + p
                else:  # 拉伸
                    f10 = _ONE / (_ONE - rf)
                    if rf >= _HALF:
                        carry = int((rf + rf - _ONE) * np.float32(p) * f10)
                        olen = p
                    else:
                        olen = int(np.float32(p) * rf * f10)
                    if pos + 2 * p > blen:
                        break
                    a = buf[pos:pos + p]
                    b = buf[pos + p:pos + 2 * p]
                    out.extend(a)
                    out.extend(_crossfade(b, a, olen))
                    pos += olen
            else:
                m = min(STEP, carry)
                if pos + m > blen:
                    break
                out.extend(buf[pos:pos + m])
                pos += m
                carry -= m
        if pos > 0:
            del buf[:pos]
            pos = 0

    process()
    before = len(out)
    pending = len(buf)
    # 收尾：补 2*STEP 个 0 后再处理，输出截断到 round(pending/rate)
    buf.extend([0] * (2 * STEP))
    process()
    n_keep = int(pending / r + 0.5)
    return np.clip(np.array(out[:before + n_keep], dtype=np.int64), -32768, 32767).astype(np.int16)


def adjust_float(audio, rate):
    """float32 [-1,1] 音频 → 引擎语速变换后的 float32（int16 域精确复刻）"""
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    x = np.trunc(a * np.float32(32767.0)).astype(np.int32)
    return adjust_int16(x, rate).astype(np.float32) / np.float32(32767.0)
