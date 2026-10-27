#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""symbols.py —— 标定数据（符号名）→ 当前音色包 id 的解析

同一语言的不同音色（如 Xiaoxiao / Yunxi）phones.txt 的 id 映射不同，
所以 lexicon / kana / syllables 以符号名保存，运行时再映射成本音色包的 id。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def _alt_names(x):
    yield x.rstrip("0123456789")
    yield x.replace("hh", "h")
    yield x.replace("pso", "p")
    yield x.replace("ss", "s")
    yield x.replace("dd", "d")
    yield x.replace("gg", "g")
    yield x.replace("bb", "b")
    yield x.replace("jj", "j")


def resolve_seq(seq, sym):
    out = []
    for x in seq:
        if isinstance(x, int):
            out.append(x)
            continue
        if x in sym:
            out.append(sym[x])
            continue
        hit = None
        for alt in _alt_names(x):
            if alt and alt in sym:
                hit = sym[alt]
                break
        out.append(hit if hit is not None else sym.get("padding", 0))
    return out


def resolve_table(table, sym):
    """{key: [符号名]} → {key: [id]}"""
    if not table:
        return table
    first = next(iter(table.values()), None)
    if not first or isinstance(first[0], int):
        return table
    key = (id(table), id(sym))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    out = {k: resolve_seq(v, sym) for k, v in table.items()}
    if len(_cache) > 16:
        _cache.clear()
    _cache[key] = out
    return out


def resolve_lexicon(lexicon, sym):
    if not lexicon:
        return lexicon or {}
    first = next(iter(lexicon.values()), None)
    if not first or isinstance(first[0], int):
        return lexicon
    key = (id(lexicon), id(sym))
    hit = _cache.get(key)
    if hit is not None and hit[0] is lexicon and hit[1] is sym:
        return hit[2]
    out = {k: resolve_seq(v, sym) for k, v in lexicon.items()}
    if len(_cache) > 16:
        _cache.clear()
    # 同时持有 lexicon/sym 引用，避免对象被回收后 id 复用导致串用缓存
    _cache[key] = (lexicon, sym, out)
    return out


def load_json_prefer_names(path_base):
    """优先读 <base>.names.json，退回归一化后的 <base>.json"""
    names = path_base + ".names.json"
    if os.path.exists(names):
        return json.load(open(names, encoding="utf-8"))
    p = path_base + ".json"
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def is_name_table(table):
    if not table:
        return False
    first = next(iter(table.values()), None)
    return bool(first) and isinstance(first[0], str)
