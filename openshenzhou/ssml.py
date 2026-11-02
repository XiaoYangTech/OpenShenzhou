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
    """引擎音高链路精确复刻：PSOLA(r=f20/f28) → b0264 12 抽头定点重采样 → afb08 flush 截断

    对引擎进程内 24kHz PCM 验证：pitch=+25%/+50%/-50% 逐采样 maxdiff=0（见 findings §61）。
    """
    if len(audio) == 0:
        return audio
    from .pitchshift import adjust_pitch_float
    return adjust_pitch_float(audio, f28, f20=rate_factor)


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


def _engine_durations(tts, ids):
    """引擎 encoder 的逐 token mel 帧数（SSML break 的插入位置由它决定）"""
    enc = tts.pack.sessions()[0]
    x = np.array([ids], dtype=np.int32)
    sp = np.full((1, len(ids)), 1.0, np.float32)
    _di, d = enc.run(None, {"inputs": x, "speed_ratios": sp})
    return np.asarray(d).reshape(-1).astype(np.int64)


def _synth_ids(tts, ids):
    """按 token 序列合成（用于把 break 变成 br3 暂停 token 的单句）"""
    from .synth import synthesize
    stream = tts._streaming()
    if stream:
        return synthesize(tts.pack.sessions(), ids, None, streaming=stream)
    proj = tts._projection()
    return synthesize(tts.pack.sessions(), ids, None, projection=proj if proj else None)


def _synth_break_pieces(tts, pieces):
    """含 <break> 的 run。

    引擎实测行为（findings §62）：
      · 句中 break 在 token 序列里被替换为 ``br3``（暂停 token），整句一次合成；
        实测音频总长 = 300 × encoder 总帧数，与引擎完全一致；
      · 静音插入位置 = 300 × 到该 br3 为止的 encoder 帧数累计（多数用例与引擎逐采样一致，
        个别用例差 2~3 帧，见 §62.3）；
      · 句首 break：静音插在第一块 1200 采样之后（与引擎逐采样一致）；
      · 句边界/句间 break：静音插在前一句音频之后 1200 采样处（引擎实测纯插入）；
      · 句尾/文本末尾 break：引擎不插静音，忽略。
    """
    sym = tts.pack.symbols
    # 1) 切成文本片段 + break 位置
    frags = [[]]
    break_pos = []  # [片段下标, ms]
    for p in pieces:
        if p[0] == "break":
            break_pos.append([len(frags) - 1, p[1]])
            frags.append([])
        else:
            frags[-1].append(p[1])
    # 合并相邻 break
    merged = {}
    for pos, ms in break_pos:
        merged[pos] = merged.get(pos, 0) + ms
    frag_text = [_join_texts(x) for x in frags]
    full_text = _join_texts([f for f in frag_text if f.strip()])
    if not full_text:
        return np.zeros(0, dtype=np.float32)
    ids_full = tts.text_to_ids(full_text)
    from .synth import split_sentences
    chunks = split_sentences(ids_full, sym)
    lens = [len(c) for c in chunks]
    offs = np.cumsum([0] + lens[:-1]).tolist()

    # 2) 定位每个 break：分类 + chunk/局部 token 下标
    breaks = []  # (kind, chunk_idx, local_idx, ms)
    for pos, ms in sorted(merged.items()):
        pre = _join_texts([f for f in frag_text[:pos + 1] if f.strip()])
        post = _join_texts([f for f in frag_text[pos + 1:] if f.strip()])
        if not pre.strip():
            breaks.append(("lead", 0, 0, ms))
            continue
        if not post.strip():
            breaks.append(("trail", len(chunks) - 1, lens[-1] - 1, ms))
            continue
        ids_pre = tts.text_to_ids(pre)
        k = 0
        lim = min(len(ids_pre) - 2, len(ids_full) - 2)
        while k < lim and ids_pre[2 + k] == ids_full[2 + k]:
            k += 1
        bidx = 2 + k
        if bidx >= len(ids_full) or ids_full[bidx] == sym.get("~"):
            breaks.append(("trail", len(chunks) - 1, lens[-1] - 1, ms))
            continue
        if ids_full[bidx] == sym.get("-bos-") or bidx in offs:
            # 句边界：插到下一句（本 break 所在 chunk 的下一块）1200 处
            nxt = 0
            for i, o in enumerate(offs):
                if o <= bidx:
                    nxt = i
            breaks.append(("sent", nxt, 0, ms))
            continue
        ci = 0
        for i, o in enumerate(offs):
            if o <= bidx < o + lens[i]:
                ci = i
                break
        li = bidx - offs[ci]
        breaks.append(("mid", ci, li, ms))

    # 3) 逐 chunk 合成/插静音
    out_parts = []
    for ci, ch in enumerate(chunks):
        ids_mod = list(ch)
        cuts = []
        for kind, bci, li, ms in breaks:
            if kind == "mid" and bci == ci:
                ids_mod[li] = sym["br3"]
                cuts.append((li, int(round(ms * 24.0))))
        if not cuts:
            audio = _synth_ids(tts, ch)
        else:
            durs = _engine_durations(tts, ids_mod)
            cum = np.cumsum(durs)
            audio = _synth_ids(tts, ids_mod)
            ins = []
            for li, sil in cuts:
                cut = int(cum[li]) * 300
                ins.append((cut, sil))
            ins.sort()
            parts = []
            prev = 0
            for cut, sil in ins:
                cut = max(prev, min(cut, len(audio)))
                parts.append(audio[prev:cut])
                parts.append(np.zeros(sil, dtype=np.float32))
                prev = cut
            parts.append(audio[prev:])
            audio = np.concatenate(parts) if parts else audio
        # 句首 / 句边界 break：插到 1200 采样处（句首插在首块，句间插在下一块）
        for kind, bci, li, ms in breaks:
            if kind in ("lead", "sent") and bci == ci:
                sil = int(round(ms * 24.0))
                local = min(1200, len(audio))
                audio = np.concatenate([audio[:local], np.zeros(sil, dtype=np.float32),
                                        audio[local:]])
        out_parts.append(audio)
    return np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.float32)



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
            audio = _synth_break_pieces(t, pieces)
        # 引擎总速率因子：f20 = float32( base * float32(1 + float32(pct) * float32(0.01)) )
        # （实测：Aria +60% → 1.5999999046；+20% → 1.2000000477；-20% → 0.8000000119；
        #  -33% → 0.6700000167；与 (1+pct/100.0) 的 float32 结果有 1 ulp 差异）
        _pct = _np.float32(rate or 0.0)
        factor = float(_np.float32(_np.float32(base_rate) * _np.float32(
            1.0 + _np.float32(_pct * _np.float32(0.01)))))
        # 引擎音高：f28 = 2^(pitch百分比/120)
        f28 = float(np.float32(2.0) ** np.float32((pitch or 0.0) / 120.0))
        if abs(f28 - 1.0) > 1e-6:
            # 引擎在有音高时走**同一个音频处理器**：PSOLA(r=f20/f28) → 定点重采样(f28)，
            # 因此 rate 与 pitch 必须由同一条链路一次完成（不能先 rate 再 pitch）。
            if len(audio) > 800:
                audio = np.asarray(audio, dtype=np.float32).copy()
                audio[:300] = 0.0
                audio[-300:] = 0.0
            audio = _pitch_shift_engine(audio, f28, factor)
        elif abs(factor - 1.0) > 1e-9:
            # 纯 RateAdjustment：引擎走 b0514（PSOLA，r=f20），再经 afb08 flush 截断
            if len(audio) > 800:
                audio = _np.asarray(audio, dtype=_np.float32).copy()
                audio[:300] = 0.0
                audio[-300:] = 0.0
            from .rateadjust import adjust_float
            audio = adjust_float(audio, factor)
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
