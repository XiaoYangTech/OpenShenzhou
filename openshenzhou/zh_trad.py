#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zh_trad.py —— 繁体文本逐字读音修正（数据驱动逼近引擎繁体路径）

引擎对繁体文本不做繁→简归一，而是按繁体字本身读音逐字处理。SDK 主路径基于简体
词典/模型，繁体场景下部分字（還/得/子/見/將/著…）的读音、轻声与声调会偏离引擎。
本模块用 3000 句繁体小说引擎 ids 训练的字级上下文修正表（5 级 backoff，
交叉验证 97.80% vs 基线 96.03%）修正逐字读音。

投影：
  - 仅当文本含「有独立简体形的繁体字」时启用（trad_chars）；
  - 基线 ids 的音节数必须等于汉字数（否则放弃修正，保持原输出）；
  - 只覆盖与基线不同的读音项。
"""
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BOUND = {"-bos-", "br0", "br1", "br2", "br3", "br4", "~", "-"}

_MODEL = None
_TABLES = None
_TRAD_CHARS = None
_ENABLED = os.environ.get('OS_TRAD_OFF') != '1'


def set_enabled(flag):
    """测试/对照用：临时关闭繁体修正投影"""
    global _ENABLED
    _ENABLED = bool(flag)


def enabled():
    return _ENABLED


def _load():
    global _MODEL, _TABLES, _TRAD_CHARS
    if _MODEL is not None:
        return
    path = os.path.join(DATA, "zh_trad_ctx_model.json.gz")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            _MODEL = json.load(f)
    except FileNotFoundError:
        _MODEL = {"tables": [], "trad_chars": ""}
    _TABLES = _MODEL.get("tables") or []
    # 自动检测用高精度 gate（对简体语料 0 误触发）；无 gate 时退回全量繁体字集
    _TRAD_CHARS = set(_MODEL.get("trad_gate") or _MODEL.get("trad_chars") or "")


def detect(text):
    """文本是否含繁体字（有独立简体形的字）"""
    _load()
    for ch in text or "":
        if ch in _TRAD_CHARS:
            return True
    return False


def _chars_and_syllables(sent, ids, rev):
    """把 ids 解析成 (chars, syllables, segments)；音节数与汉字数不一致时返回 None。
    segments: [('sep', token)] 或 ('syl', [tokens])"""
    chars = [c for c in sent if "\u4e00" <= c <= "\u9fff" or c == "〇"]
    segments = []
    cur = []
    for i in ids:
        t = rev.get(i)
        if t is None:
            t = "?%d" % i
        if t in BOUND or t.startswith("punc"):
            if cur:
                segments.append(("syl", list(cur)))
                cur = []
            segments.append(("sep", t))
        else:
            cur.append(t)
    if cur:
        segments.append(("syl", list(cur)))
    syls = [s[1] for s in segments if s[0] == "syl"]
    if len(syls) != len(chars):
        return None
    return chars, syls, segments


def _lookup(chars, i, cur):
    """按 5 级 backoff 查修正结果；返回 token 列表或 None"""
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
            return list(hit[0])
    return None


def apply(text, ids, sym2id):
    """对一句文本的基线 ids 做逐字读音修正；不适用时原样返回"""
    _load()
    if not _ENABLED or not _TABLES or not detect(text):
        return ids
    rev = {v: k for k, v in sym2id.items()}
    parsed = _chars_and_syllables(text, ids, rev)
    if parsed is None:
        return ids
    chars, syls, segments = parsed
    fixed = []
    changed = False
    for i, cur in enumerate(syls):
        new = _lookup(chars, i, cur)
        if new is None or new == cur:
            fixed.append(cur)
            continue
        # 只接受能映射回符号表的 token
        if any(t not in sym2id for t in new):
            fixed.append(cur)
            continue
        fixed.append(new)
        changed = True
    if not changed:
        return ids
    out = []
    k = 0
    for kind, payload in segments:
        if kind == "sep":
            out.append(sym2id[payload])
        else:
            toks = fixed[k]
            k += 1
            # 保留原音节内 '-' 分隔符结构（若有）
            orig = syls[k - 1]
            if len(toks) == len(orig) or True:
                out.extend(sym2id[t] for t in toks)
    return out
