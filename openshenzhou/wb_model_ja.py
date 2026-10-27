#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wb_model_ja.py —— 日语官方断词模型（CBlstmWordBreaker）的纯本地复刻

已验证：本模块输出的逐字类别与引擎进程内 CBlstmWordBreaker 完全一致
（5 句 / 54/54 字符 100% 匹配）。

字符→嵌入行号来自反编译（lang=0x411 分支，函数 FUN_1802548ec / FUN_180254c78）。
类别语义与中文同源：1/2 = 词内延续，0/3 = 词起始标记；
词界 = 在 class∈{0,3} 的字符之前切，再按日语词表合并。

用法::

    from openshenzhou import wb_model_ja
    segs = wb_model_ja.segments(list("東京タワーに行きます"))
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data", "wb_ja_bilstm_weights.npz")
LEXICON = os.path.join(ROOT, "data", "lexicon_ja.names.json")
WINDOW = 20
MAX_MERGE_LEN = 12

_cache = {}


def _load():
    if "w" in _cache:
        return _cache["w"]
    d = np.load(NPZ)
    w = dict(
        emb=d["emb"].astype(np.float32),
        w_fwd=d["w_fwd"].astype(np.float32),
        b_fwd=d["b_fwd"].astype(np.float32),
        w_bwd=d["w_bwd"].astype(np.float32),
        b_bwd=d["b_bwd"].astype(np.float32),
        w_out=d["w_out"].astype(np.float32),
        b_out=d["b_out"].astype(np.float32),
    )
    if w["w_fwd"].shape[0] == 256:
        w["w_fwd"] = w["w_fwd"].T
        w["w_bwd"] = w["w_bwd"].T
    if w["w_out"].shape[0] == 4:
        w["w_out"] = w["w_out"].T
    _cache["w"] = w
    return w


def _load_words():
    if "lex" in _cache:
        return _cache["lex"]
    try:
        words = set(json.load(open(LEXICON, encoding="utf-8")).keys())
    except Exception:
        words = set()
    _cache["lex"] = words
    return words


def char_row(ch):
    """日语字符 → 嵌入行号（反编译公式，ja-JP）。"""
    c = ord(ch) & 0xFFFF
    if 0x3004 < c < 0x9FCD or c >= 0x9FCD:
        if c < 0x3008:
            return 6 + (c - 0x3005)
        if c < 0x30A0:
            return 6 + (c - 0x303D)
        if c < 0x30FB:
            return 6 + (c - 0x303E)
        if c < 0x3100:
            return 6 + (c - 0x303F)
        if c < 0x3200:
            return 6 + (c - 0x312F)
        if c < 0x9FCD:
            return 6 + (c - 0x4D2F)
        return 6 + (c - 0xACC8)
    if "0" <= ch <= "9":
        return 2
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return 3
    return 4


def _run_lstm(X, W, b, reverse=False):
    Wx, Wh = W[:64], W[64:]
    bb = b.reshape(4, 64).reshape(-1)
    n = len(X)
    h = np.zeros(64, dtype=np.float32)
    c = np.zeros(64, dtype=np.float32)
    H = np.zeros((n, 64), dtype=np.float32)
    idxs = range(n - 1, -1, -1) if reverse else range(n)
    for t in idxs:
        pre = (X[t] @ Wx + h @ Wh + bb).reshape(4, 64)
        i = 1.0 / (1.0 + np.exp(-np.clip(pre[0], -30, 30)))
        f = 1.0 / (1.0 + np.exp(-np.clip(pre[1], -30, 30)))
        g = np.tanh(pre[2])
        c = f * c + i * g
        o = 1.0 / (1.0 + np.exp(-np.clip(pre[3], -30, 30)))
        h = o * np.tanh(c)
        H[t] = h
    return H


def classes(chars):
    w = _load()
    n = len(chars)
    if n == 0:
        return []
    seq = [w["emb"][0]] + [w["emb"][char_row(c)] for c in chars] + [w["emb"][1]]
    while len(seq) % WINDOW:
        seq.append(w["emb"][1])
    F = np.stack(seq)
    T = len(F)
    Hf = np.zeros((T, 64), dtype=np.float32)
    Hb = np.zeros((T, 64), dtype=np.float32)
    for s in range(0, T, WINDOW):
        X = F[s:s + WINDOW]
        m = len(X)
        Hf[s:s + m] = _run_lstm(X, w["w_fwd"], w["b_fwd"], False)
        Hb[s:s + m] = _run_lstm(X, w["w_bwd"], w["b_bwd"], True)
    H = np.concatenate([Hf, Hb], 1)
    logits = H @ w["w_out"] + w["b_out"]
    return [int(x) for x in logits.argmax(1)[1:1 + n]]


def _segments_from_classes(cls):
    segs = []
    st = 0
    for j in range(1, len(cls)):
        if cls[j] in (0, 3):
            segs.append((st, j - st))
            st = j
    segs.append((st, len(cls) - st))
    return segs


HIRA = set("ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん")


def _merge_by_lexicon(text, segs, words, max_len=MAX_MERGE_LEN):
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(segs):
            best = None
            for k in range(len(segs) - 1, i, -1):
                end = segs[k][0] + segs[k][1]
                cand = text[segs[i][0]:end]
                if len(cand) > max_len or cand not in words:
                    continue
                # 引擎不会把纯假名短语合并成一个词（ですね/ください 等），
                # 只对汉字/片假名复合词做合并
                if any(ch in HIRA for ch in cand):
                    continue
                best = k
                break
            if best is not None:
                out.append((segs[i][0], segs[best][0] + segs[best][1] - segs[i][0]))
                i = best + 1
                changed = True
            else:
                out.append(segs[i])
                i += 1
        segs = out
    return segs


def segments(chars):
    text = "".join(chars)
    cls = classes(chars)
    segs = _segments_from_classes(cls)
    return _merge_by_lexicon(text, segs, _load_words())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for text in sys.argv[1:] or ["東京都に行きます"]:
        print(text, "->", [text[s:s + L] for s, L in segments(list(text))])
