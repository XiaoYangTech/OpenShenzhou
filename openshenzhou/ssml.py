# -*- coding: utf-8 -*-
"""SSML 支持（引擎行为复刻，Python SDK）

已实测的引擎行为（Aria en-US，24 kHz）：

* ``<prosody volume="...">`` —— 纯增益，时长不变。
  数值音量 v（百分比，默认 100）→ 线性增益 ``10 ** ((v-100)/80)``，超过 ±1 时削波。
  命名音量：x-soft=20 / soft=40 / medium=60 / loud=80 / x-loud=100（换算成上面的 v）。
* ``<prosody rate="...">`` —— 时长 = 自然时长 / (1 + r/100)，音高不变（引擎为重构语音）。
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
    """音高近似：按半音重采样（音高与时长耦合，仅近似）"""
    if abs(semis) < 1e-3 or len(audio) == 0:
        return audio
    ratio = 2.0 ** (semis / 12.0)
    n = max(1, int(round(len(audio) / ratio)))
    idx = np.linspace(0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _word_starts(tts, text):
    """词起始采样位置（用逐帧一致的事件模型；失败时返回 None）"""
    try:
        from .events import word_events
        evs = word_events(tts, text)
        return [int(round(e["audio"] * 24.0 / 10000.0)) for e in evs]
    except Exception:
        return None


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
                    inserts.append((pos, silence))
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
        if rate:
            # 引擎：新时长 = 原始时长 / (1 + r/100)
            audio = wsola(audio, 1.0 / (1.0 + rate / 100.0))
        if pitch:
            audio = _resample_pitch(audio, pitch / 10.0)  # 100% ≈ 10 半音（近似）
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
