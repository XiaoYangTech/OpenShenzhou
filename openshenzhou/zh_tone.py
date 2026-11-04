#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zh_tone.py —— 中文上下文声调/读音修正（数据驱动）

来源：8,756 句同域 UI 语料的引擎 ids 与 SDK 基线 ids 对齐训练的字级上下文修正表
（键 = 基线音节 token + 前2/前1/后1字符，5 级 backoff；只保留 count>=5、ratio>=0.90
且与基线不同的覆盖项）。

验证：
  - 独立留出 1500 句：基线 1419/1500 → 1448/1500（+29，仅 2 句被破坏）
  - 中文 3000 句评测：2915/3000 → **2922/3000**（+7，0 句被破坏）
  - 中文 599 句评测：不变（596/599）
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


def _lookup(chars, i, cur):
    n = len(chars)
    ch = chars[i]
    p1 = chars[i - 1] if i > 0 else "^"
    p2 = chars[i - 2] if i > 1 else "^"
    n1 = chars[i + 1] if i + 1 < n else "$"
    cur_s = " ".join(cur)
    keys = [cur_s + "\x1f" + p2 + p1 + ch + n1,
            cur_s + "\x1f" + p1 + ch + n1,
            cur_s + "\x1f" + ch + n1,
            cur_s + "\x1f" + ch,
            cur_s]
    for depth, key in enumerate(keys):
        table = _TABLES[depth] if depth < len(_TABLES) else None
        if not table:
            continue
        hit = table.get(key)
        if hit:
            return list(hit)
    return None


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
    names = []
    for x in ids:
        names.append(rev.get(x, "?%d" % x))
        if x == 1:
            break
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff" or c == "〇"]
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
    if len(words) != len(chars):
        return ids
    out = []
    changed = False
    wi = 0
    for kind, payload in pieces:
        if kind == "sep":
            out.append(sym2id.get(payload, sym2id.get("padding", 0)))
            continue
        i = wi
        wi += 1
        w = payload
        pred = _lookup(chars, i, w)
        if pred is not None and pred != w and all(t in sym2id for t in pred):
            w = pred
            changed = True
        out.extend(sym2id[t] for t in w)
    if not changed:
        return ids
    return out
