# -*- coding: utf-8 -*-
"""SSML 支持（引擎行为复刻，Python SDK）

已实测的引擎行为（Aria en-US，24 kHz）：

* ``<prosody volume="...">`` —— 纯增益，时长不变。
  数值音量 v（百分比，默认 100）→ 线性增益 ``10 ** ((v-100)/80)``，超过 ±1 时削波。
  命名音量：x-soft=20 / soft=40 / medium=60 / loud=80 / x-loud=100（换算成上面的 v）。
* ``<prosody rate="...">`` —— 引擎在音频域用基音同步拼接实现（与语音包 RateAdjustment 同一算法，
  总因子 = RateAdjustment/100 × (1+r/100)），SDK 已精确复刻，时长/波形与引擎一致。
  SDK 用 WSOLA 保音高伸缩近似（时长一致、音高一致，波形非逐采样一致）。
  命名：x-slow=-66.6 / slow=-33.3 / medium=0 / fast=+33.3 / x-fast=+66.6。
* ``<prosody pitch="...">`` —— 引擎通过模型内音高特征重构（时长基本不变）。
  当前 SDK 模型无音高输入，暂用重采样近似（音高与时长耦合），属已知差异。
* ``<break time="500ms"/>`` —— 在音频中插入等长静音（24 kHz 采样数 = ms*24），
  插入位置用 SDK 的词边界事件（与引擎逐帧一致）确定；引擎对 break 的分段细节仍略有差异。
* 其余标签（speak/voice/p/s/say-as/emphasis）只取其中的文本。

用法::

    from openshenzhou import TTS
    from openshenzhou.ssml import speak_ssml
    tts = TTS('D:/Voice/Aria_Female_en-US')
    audio = speak_ssml(tts, '<speak><prosody rate="-20%" volume="loud">Hello world.</prosody></speak>')
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

from .timestretch import wsola

NS = "{http://www.w3.org/2001/10/synthesis}"

_VOLUME_NAMES = {
    "x-soft": 20.0, "soft": 40.0, "medium": 60.0, "loud": 80.0, "x-loud": 100.0,
    "default": 100.0,
}
_RATE_NAMES = {
    "x-slow": -66.6, "slow": -33.3, "medium": 0.0, "default": 0.0,
    "fast": 33.3, "x-fast": 66.6,
}
_PITCH_NAMES = {
    "x-low": -50.0, "low": -25.0, "medium": 0.0, "default": 0.0,
    "high": 25.0, "x-high": 50.0,
}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tag(elem):
    return elem.tag.replace(NS, "")


def _pct(value, names):
    """解析百分比/命名成数值百分比（+50% → 50, -50% → -50, slow → -33.3）"""
    v = (value or "").strip().lower()
    if v in names:
        return names[v]
    m = re.match(r"^([+-]?)(\d+(?:\.\d+)?)%$", v)
    if m:
        num = float(m.group(2))
        return -num if m.group(1) == "-" else num
    m = re.match(r"^[+-]?\d+(?:\.\d+)?$", v)
    if m:
        return float(v)
    return None


def _volume_gain(value):
    """volume 属性 → 线性增益（引擎实测映射）"""
    if value is None:
        return 1.0
    v = (value or "").strip().lower()
    if v in _VOLUME_NAMES:
        num = _VOLUME_NAMES[v]
    else:
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)%?$", v)
        if not m:
            return 1.0
        num = float(m.group(1))
    return float(10.0 ** ((num - 100.0) / 80.0))


def _iter_segments(elem, style, out):
    """把 SSML 树摊平成 [('text', 文本, style)] / [('break', ms)] 片段"""
    tag = _tag(elem)
    if tag == "break":
        t = (elem.get("time") or "").strip().lower()
        m = re.match(r"^(\d+(?:\.\d+)?)\s*(ms|s)?$", t)
        if m:
            num = float(m.group(1))
            unit = m.group(2) or "s"
            ms = num if unit == "ms" else num * 1000.0
            out.append(("break", ms))
        return
    new_style = style
    if tag == "prosody":
        rate = _pct(elem.get("rate"), _RATE_NAMES)
        pitch = _pct(elem.get("pitch"), _PITCH_NAMES)
        gain = _volume_gain(elem.get("volume"))
        new_style = (style[0] * gain, style[1] if rate is None else rate,
                     style[2] if pitch is None else pitch)
    if elem.text:
        out.append(("text", elem.text, new_style))
    for child in list(elem):
        _iter_segments(child, new_style, out)
        if child.tail:
            out.append(("text", child.tail, new_style))


def _synth(t, text):
    from .synth import split_sentences, synthesize
    ids = t.text_to_ids(text)
    chunks = split_sentences(ids, t.pack.symbols)
    stream = t._streaming()
    auds = []
    for ch in chunks:
        if stream:
            auds.append(synthesize(t.pack.sessions(), ch, None, streaming=stream))
        else:
            proj = t._projection()
            auds.append(synthesize(t.pack.sessions(), ch, None, projection=proj if proj else None))
    if not auds:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(auds) if len(auds) > 1 else auds[0]


def _resample_pitch(audio, semis):
    """旧接口：音高近似（仅线性插值，音高与时长耦合），保留作兜底"""
    if abs(semis) < 1e-3 or len(audio) == 0:
        return audio
    ratio = 2.0 ** (semis / 12.0)
    n = max(1, int(round(len(audio) / ratio)))
    idx = np.linspace(0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _sinc_resample(x, ratio, taps=24, start=0.0):
    """窗函数 sinc 重采样：输出长度 = len(x)/ratio，音高 ×ratio（时长不变由前级 PSOLA 保证）"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or abs(ratio - 1.0) < 1e-9:
        return x.astype(np.float32)
    n_out = max(1, int(round((len(x) - start) / ratio)))
    # 输出点 j 对应输入位置 t = start + j*ratio（播放速度 ×ratio → 音高 ×ratio）
    j = np.arange(n_out, dtype=np.float64)
    t = start + j * ratio
    half = taps // 2
    k = np.arange(-half, half, dtype=np.float64)
    idx = np.floor(t).astype(np.int64)
    frac = t - idx
    out = np.zeros(n_out, dtype=np.float64)
    norm = np.zeros(n_out, dtype=np.float64)
    # 降采样（ratio>1）时按 1/ratio 收缩截止频率做抗混叠
    cutoff = 1.0 if ratio <= 1.0 else 1.0 / ratio
    for kk in k:
        pos = idx + int(kk)
        d = frac - kk
        w = cutoff * np.sinc(cutoff * d) * (0.5 + 0.5 * np.cos(np.pi * d / half))
        valid = (pos >= 0) & (pos < len(x))
        out[valid] += x[pos[valid]] * w[valid]
        norm[valid] += w[valid]
    norm[norm == 0] = 1.0
    return (out / norm).astype(np.float32)


def _engine_resample_step(f28, sr=24000):
    """引擎 b0264 的整数化重采样步长：esi=(int)(sr/f28)，while(>0x4000){esi>>=1;edi>>=1}，
    实际输入步长 = edi/esi（输出长度 = 输入/步长）"""
    edi = sr
    esi = int(np.float32(sr) / np.float32(f28))
    while esi > 0x4000 or edi > 0x4000:
        esi >>= 1
        edi >>= 1
    return float(edi) / float(esi)


def _pitch_shift_engine(audio, f28, rate_factor):
    """引擎音高链路：先 PSOLA 以 rate/f28 变速，再按引擎整数比率重采样（时长还原、音高 ×f28）"""
    if len(audio) == 0:
        return audio
    from .rateadjust import adjust_float
    stretched = adjust_float(audio, float(np.float32(rate_factor) / np.float32(f28)))
    # 引擎 b0264 的相位/中心抽头对齐等价于输入起点偏移 5 采样（对引擎 24kHz PCM 实测
    # pitch 25%/50%/-50% 时 corr=0.99996~0.999998）
    return _sinc_resample(stretched, _engine_resample_step(f28), start=5.0)


def _word_starts(tts, text):
    """词起始采样位置（用逐帧一致的事件模型；失败时返回 None）"""
    try:
        from .events import word_events
        evs = word_events(tts, text)
        return [int(round(e["audio_ticks"] * 24.0 / 10000.0)) for e in evs]
    except Exception:
        return None


def _break_pos(pos, total):
    """引擎把 break 静音插在「下一个 1200 采样块边界」：
    实测 break 在开头时插在 1200（第一块）之后，两词之间时插在后续块边界；
    SDK 用词边界向上对齐到 1200 的整数倍复刻该行为。"""
    pos = max(0, min(int(pos), int(total)))
    if pos <= 0:
        return 1200
    return int((pos + 1199) // 1200) * 1200


def _runs(segs):
    """按 style 分组：返回 [(style, [text|break...])]"""
    runs = []
    for seg in segs:
        if seg[0] == "break":
            if not runs:
                runs.append(((1.0, 0.0, 0.0), []))
            runs[-1][1].append(seg)
            continue
        style = seg[2]
        if runs and runs[-1][0] == style:
            runs[-1][1].append(("text", seg[1]))
        else:
            runs.append((style, [("text", seg[1])]))
    return runs


def speak_ssml(tts, ssml, volume=None):
    """按 SSML 合成音频（24 kHz float32）；volume 为整体额外增益（可选）"""
    t = tts
    t._ensure()
    root = ET.fromstring(ssml.strip())
    if _tag(root) != "speak":
        raise ValueError("SSML 根元素必须是 <speak>")
    segs = []
    _iter_segments(root, (1.0, 0.0, 0.0), segs)
    parts = []
    # 引擎：总加减速因子 = 语音包 RateAdjustment/100 × (1 + prosody_rate/100)，
    # 且由同一个音频域基音同步拼接处理器一次完成（已由 SAPI 路径实测确认：
    # Katja rate+100% 时处理器因子 1.1×2.0=2.2）。
    import numpy as _np
    # 引擎把 RateAdjustment/100 存成 float32（setter 接收 double 后转 float），
    # SSML 因子在 float32 域相乘：f20 = float32(base) * float32(1 + p/100)
    from .rateadjust import engine_rate_factor as _erf
    base_rate = _np.float32(_erf(getattr(t.pack, "rate_adjust", 100) or 100))
    for style, pieces in _runs(segs):
        gain, rate, pitch = style
        # 把同一段内的文本拼成整句合成（保持与引擎相同的句子结构），break 处插静音
        if all(p[0] == "text" for p in pieces):
            joined = _join_texts([p[1] for p in pieces])
            if not joined:
                continue
            audio = _synth(t, joined)
        else:
            texts = [p[1] for p in pieces if p[0] == "text"]
            joined = _join_texts(texts)
            audio = _synth(t, joined) if joined else np.zeros(0, dtype=np.float32)
            # 计算每个 break 前的词数 → 对应词边界采样位置
            wrds = _WORD_RE.findall(joined)
            starts = _word_starts(t, joined)
            inserts = []  # (位置, 静音采样数)
            word_pos = 0
            for p in pieces:
                if p[0] == "break":
                    ms = p[1]
                    silence = int(round(ms * 24.0))
                    idx = word_pos
                    pos = starts[idx] if starts and idx < len(starts) else len(audio)
                    inserts.append((_break_pos(pos, len(audio)), silence))
                else:
                    token = _WORD_RE.search(p[1] or "")
                    word_pos += len(_WORD_RE.findall(p[1] or ""))
            if inserts:
                out = []
                prev = 0
                for pos, silence in inserts:
                    pos = max(prev, min(pos, len(audio)))
                    out.append(audio[prev:pos])
                    out.append(np.zeros(silence, dtype=np.float32))
                    prev = pos
                out.append(audio[prev:])
                audio = np.concatenate(out) if out else audio
        factor = float(_np.float32(base_rate * _np.float32(1.0 + (rate or 0.0) / 100.0)))
        if abs(factor - 1.0) > 1e-9:
            # 引擎进 rate 处理器之前，自然音频首尾已有各 300 采样静音
            # （Vocoder 前导/收尾帧），必须先复刻，否则基音检测会相位漂移
            if len(audio) > 800:
                audio = _np.asarray(audio, dtype=_np.float32).copy()
                audio[:300] = 0.0
                audio[-300:] = 0.0
            # 引擎同款：基音同步拼接（精确复刻的 RateAdjustment 算法）
            from .rateadjust import adjust_float
            audio = adjust_float(audio, factor)
        # 引擎音高：SSML pitch 百分比 → 半音 = pct/10 → 因子 f28 = 2^(pct/120)
        # （SAPI 实测：absmiddle=5→SSML 25%→f28=1.155353=2^(2.5/12)；
        #  absmiddle=10/20→adapter 截断到 10→SSML 50%→f28=1.33484=2^(5/12)；
        #  absmiddle=-20→SSML -50%→f28=0.749154=2^(-5/12)）
        # 处理顺序：先 PSOLA 变速 rate/f28，再按 f28 重采样（时长不变、音高 ×f28）
        f28 = float(np.float32(2.0) ** np.float32((pitch or 0.0) / 120.0))
        if abs(f28 - 1.0) > 1e-6:
            # 引擎在音高/速率处理前，自然音频首尾已有各 300 采样静音
            if len(audio) > 800:
                audio = np.asarray(audio, dtype=np.float32).copy()
                audio[:300] = 0.0
                audio[-300:] = 0.0
            audio = _pitch_shift_engine(audio, f28, factor)
        elif pitch:
            audio = _resample_pitch(audio, pitch / 10.0)
        audio = audio * float(gain)
        parts.append(audio)
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if volume is not None:
        audio = audio * float(volume)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _join_texts(texts):
    out = ""
    for t in texts:
        t = re.sub(r"\s+", " ", t or "")
        if not t:
            continue
        if out and not out.endswith(" ") and not t.startswith(" "):
            out += " "
        out += t.strip()
    return out.strip()
