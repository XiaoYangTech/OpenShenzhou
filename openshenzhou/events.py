#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""events.py —— 复刻引擎 WordBoundary 事件（audio 偏移 / 文本 offset / 长度 / 词 / 类型）

引擎行为（实测）：
- 每个词一个事件，type=0；duration 字段恒为 0
- audio 偏移单位 100ns：audio_ticks = 帧号 × 125000（12.5ms/帧）
- 帧号 = 前面各句总帧数 + 本句内该词首音素之前的累计帧数
  （首词包含 -bos-/br0；非首词包含前词末尾的 br1）
- 词划分 = 源文本 token 化（英文按空白/标点切分，保留缩写点、数字、金额、时间、百分号等）

用法：
    tts = TTS(voice_dir)
    for e in tts.word_events("Hello world. How are you?"):
        print(e["audio_ticks"], e["word"])
"""
import re

import numpy as np

FRAME_TICKS = 125000  # 12.5ms/帧 = 125000 × 100ns

ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
          "co", "no", "dept", "est", "fig", "gen", "gov", "sen", "rep", "col", "capt",
          "lt", "sgt", "rev", "hon", "pres", "approx", "e.g", "i.e",
          "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
          "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"}

_TOKEN_RE = re.compile(
    r"[$€£¥]?\d[\d,]*(?:[.:\-/]\d+)*%?"            # 金额 / 数字 / 时间 / 日期 / 百分号
    r"|[A-Za-z0-9]+(?:[,.'’\-/:][A-Za-z0-9]+)*\.?"  # 普通单词（内部标点、缩写点）
)

_STRUCT = {"-bos-", "br0", "br1", "br2", "br3", "br4", "~"}


def _is_struct(name):
    return name in _STRUCT or name.startswith("punc")


def tokenize(text):
    """源文本 → [(start, end, word)]，word 为事件文本。"""
    out = []
    i = 0
    n = len(text)
    while i < n:
        m = _TOKEN_RE.match(text, i)
        if not m:
            i += 1
            continue
        raw = m.group(0)
        start, end = m.start(), m.end()
        word = raw
        if word.endswith("."):
            core = word[:-1]
            keep = (re.fullmatch(r"(?:[A-Za-z]\.){2,}[A-Za-z]?\.?", word) is not None
                    or core.lower() in ABBREV)
            if not keep:
                word = core
                end -= 1
        out.append((start, end, word))
        i = end if end > i else start + 1
    return out


def word_events(tts, text, speed=1.0):
    """返回引擎风格的词边界事件列表。"""
    sym = tts.pack.symbols
    rev = {v: k for k, v in sym.items()}
    if tts._sessions is None:
        tts._sessions = tts.pack.sessions()
    enc = tts._sessions[0]
    from .synth import split_sentences

    ids = tts.text_to_ids(text)
    chunks = split_sentences(ids, sym)
    tokens = tokenize(text)

    # 句子归属：词间空隙出现句末标点（后接空白/结尾）即进入下一句
    tok_sent = []
    tok_first = []
    sent_i = 0
    prev_end = 0
    for ts, te, w in tokens:
        gap = text[prev_end:ts]
        # 省略号是句内停顿，不切句；单个句末标点才切句
        if not re.search(r"\.{2,}|…", gap) and re.search(r"[.!?](?=\s|$)", gap):
            sent_i += 1
        tok_sent.append(sent_i)
        tok_first.append(1 if (not tok_first or tok_sent[-2] != sent_i) else 0)
        prev_end = te

    events = []
    base_frames = 0
    for ci, ch in enumerate(chunks):
        x = np.array([ch], dtype=np.int32)
        sp = np.full((1, len(ch)), float(speed), dtype=np.float32)
        _di, dur = enc.run(None, {"inputs": x, "speed_ratios": sp})
        cum = np.cumsum(dur[0]).astype(np.int64)
        content_positions = [i for i, t in enumerate(ch) if not _is_struct(rev.get(t, ""))]
        for tok_i, (ts, te, w) in enumerate(tokens):
            if tok_sent[tok_i] != ci:
                continue
            if tok_first[tok_i]:
                p_content = 0
            else:
                prefix = text[:ts].rstrip()
                p_ids = tts.text_to_ids(prefix)
                p_chunks = split_sentences(p_ids, sym)
                p_content = sum(1 for t in p_chunks[-1] if not _is_struct(rev.get(t, "")))
            if p_content >= len(content_positions):
                continue
            idx_first = content_positions[p_content]
            frame = base_frames + (int(cum[idx_first - 1]) if idx_first > 0 else 0)
            events.append({
                "audio_ticks": frame * FRAME_TICKS,
                "duration_ticks": 0,
                "text_offset": ts,
                "text_length": te - ts,
                "word": w,
                "type": 0,
            })
        base_frames += int(np.sum(dur))
    return events
