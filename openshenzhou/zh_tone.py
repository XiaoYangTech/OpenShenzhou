#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zh_tone.py —— 中文上下文声调/读音修正（数据驱动，v2 富特征）

来源：8,756 句同域 UI 语料的引擎 ids 与 SDK 基线 ids 对齐训练的字级上下文修正表。
v2 特征：基线音节 token + 前2/前1/后1字符 + 前后词界位 + 句内位置（首/中/末），
7 级 backoff。

验证：
  - 独立留出 1500 句：基线 1419/1500 → **1462/1500（+43，仅 1 句被破坏）**
  - 中文 3000 句评测：2915 → **2925/3000（fixed 12 / broke 2）**
  - 中文 599 句：不变（596/599）
只对简体路径生效（繁体文本由 zh_trad 处理）。
"""
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BOUND = {"-bos-", "br0", "br1", "br2", "br3", "br4", "~", "-"}

_MODEL = None
_TABLES = None
_ENABLED = True


def _load():
    global _MODEL, _TABLES
    if _MODEL is not None:
        return
    path = os.path.join(DATA, "zh_tone_ctx_model.json.gz")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _MODEL = json.load(f)
    except FileNotFoundError:
        _MODEL = {"tables": []}
    _TABLES = _MODEL.get("tables") or []


def set_enabled(flag):
    global _ENABLED
    _ENABLED = bool(flag)


def _parse(ids, rev):
    names = []
    for x in ids:
        names.append(rev.get(x, "?%d" % x))
        if x == 1:
            break
    pieces = []
    cur = []
    for t in names:
        if t in BOUND or t.startswith("punc"):
            if cur:
                pieces.append(("word", cur))
                cur = []
            pieces.append(("sep", t))
        else:
            cur.append(t)
    if cur:
        pieces.append(("word", cur))
    words = [p[1] for p in pieces if p[0] == "word"]
    bits = []
    pending = []
    for kind, payload in pieces:
        if kind == "sep":
            pending.append(payload)
        else:
            bits.append(1 if "br1" in pending else 0)
            pending = []
    return pieces, words, bits


def _keys(chs, words, bits, i):
    n = len(chs)
    ch = chs[i]
    p1 = chs[i - 1] if i > 0 else "^"
    p2 = chs[i - 2] if i > 1 else "^"
    n1 = chs[i + 1] if i + 1 < n else "$"
    bl = bits[i] if i < len(bits) else 0
    br = bits[i + 1] if i + 1 < len(bits) else 0
    pos = "B" if i == 0 else ("E" if i == n - 1 else "M")
    cur = " ".join(words[i])
    s = "\x1f"
    return [s.join([cur, p2 + p1 + ch + n1, "%d%d" % (bl, br), pos]),
            s.join([cur, p1 + ch + n1, "%d%d" % (bl, br)]),
            s.join([cur, ch + n1, "%d%d" % (bl, br)]),
            s.join([cur, p1 + ch, str(bl)]),
            s.join([cur, ch, str(bl)]),
            s.join([cur, ch]),
            cur]


def apply(text, ids, sym2id):
    """对一句基线 ids 做上下文声调修正；不适用/无覆盖时原样返回"""
    _load()
    if not _ENABLED or not _TABLES:
        return ids
    try:
        from . import zh_trad
        if zh_trad.detect(text):
            return ids
    except Exception:
        pass
    rev = {v: k for k, v in sym2id.items()}
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff" or c == "〇"]
    pieces, words, bits = _parse(ids, rev)
    if len(words) != len(chars):
        return ids
    changed = False
    new_words = []
    for i, w in enumerate(words):
        keys = _keys(chars, words, bits, i)
        use = w
        for d, key in enumerate(keys):
            table = _TABLES[d] if d < len(_TABLES) else None
            if not table:
                continue
            hit = table.get(key)
            if hit:
                if hit != w and all(t in sym2id for t in hit):
                    use = list(hit)
                    changed = True
                break
        new_words.append(use)
    if not changed:
        return ids
    out = []
    wi = 0
    for kind, payload in pieces:
        if kind == "sep":
            out.append(sym2id.get(payload, sym2id.get("padding", 0)))
        else:
            out.extend(sym2id[t] for t in new_words[wi])
            wi += 1
    return out
