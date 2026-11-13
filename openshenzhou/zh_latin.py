#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zh_latin.py —— 中文语音中的拉丁字母/英文词读音（引擎实测复刻）

引擎行为（Xiaoxiao 等中文语音实测）：
  - 单个拉丁字母：中文式字母名（A→ey1、B→bi i_h i_l、C→s i_h i_h …）；
  - 多个字母连写（缩写/编号）：英文式字母名逐字母读，词内用 '-' 分隔
    （AB→ey1 - b iy1、CPU→s iy2 - ph iy - yi uw1）；
  - 英文单词：按英文读音读（Hello→hh eh - l ow1、Windows→wu ih1 n - dd ow zz）；
  - 字母/英文词与中文之间用 br1 词界分隔。

数据 data/zh_latin.json 由 tools/build_zh_latin_model.py 从
data/zh_latin_engine.json（引擎探针）生成：单字母表、字母串表、英文词读音表、
英文音素→中文音素 token 映射。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_MODEL = None
_ENABLED = True


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = os.path.join(DATA, "zh_latin.json")
    try:
        _MODEL = json.load(open(path, encoding="utf-8"))
    except Exception:
        _MODEL = {}
    return _MODEL


def set_enabled(flag):
    global _ENABLED
    _ENABLED = bool(flag)


def _split_groups(tokens):
    """把含 '-' 的 token 列表按 '-' 切分为音节组"""
    groups = []
    cur = []
    for t in tokens:
        if t == "-":
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        groups.append(cur)
    return groups


def _word_from_phones(word, sym2id, model):
    """英文词：英文词典音素 → 中文 token；不可用返回 None"""
    try:
        from . import g2p_en
        lex = g2p_en.load_lexicon()
    except Exception:
        return None
    phones = lex.get(word)
    if not phones:
        return None
    pmap = model.get("phone_map") or {}
    toks = []
    for p in phones:
        if p == "-":
            toks.append("-")
            continue
        t = pmap.get(p)
        if t is None:
            t = p if p in sym2id else ("zzh" if p == "zh" else None)
        if t is None or t not in sym2id:
            return None
        toks.append(t)
    return _split_groups(toks)


def groups(run, sym2id):
    """拉丁串 → 音节组（每组为 token 名列表）；不可处理返回 None"""
    model = _load()
    if not _ENABLED or not model or not run:
        return None
    letters = model.get("letters") or {}
    single = model.get("single") or {}
    words = model.get("words") or {}
    if len(run) == 1:
        key = run.upper()
        g = single.get(key)
        if g:
            return _split_groups(g)
        return None
    if "-" in run:
        parts = [x for x in run.split("-") if x]
        if not parts:
            return None
        joined = "".join(parts).lower()
        if joined in words:
            return _split_groups(words[joined])
        w = _word_from_phones(joined, sym2id, model)
        if w:
            return w
        # 逐段用字母串读法，段间 '-'
        toks = []
        for part in parts:
            sub = []
            for ch in part:
                g = letters.get(ch.upper())
                if g is None:
                    return None
                sub.extend(g)
            if not sub:
                return None
            if toks:
                toks.append("-")
            toks.extend(sub)
        return _split_groups(toks)
    key = run.lower()
    if key in words:
        return _split_groups(words[key])
    # 大写缩写（无元音，如 MP/PDF/XML）按字母读（引擎实测：MP3 → M P）
    if run.isupper() and not any(c in "AEIOU" for c in run):
        w = None
    else:
        w = _word_from_phones(key, sym2id, model)
    if w:
        return w
    # 逐字母（英文式字母名）
    toks = []
    for ch in run:
        g = letters.get(ch.upper())
        if g is None:
            return None
        if toks:
            toks.append("-")
        toks.extend(g)
    return _split_groups(toks)


def available():
    model = _load()
    return bool(model.get("single"))
