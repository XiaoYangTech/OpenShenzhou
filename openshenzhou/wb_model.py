#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wb_model.py —— 官方中文断词模型（CBlstmWordBreaker）的纯本地复刻

已逐字节验证：本模块输出的逐字类别与引擎进程内 CBlstmWordBreaker 完全一致
（40 句/325 字 100% 匹配）。类别语义经统计确定：
    1/2 = 词内延续，0/3 = 词起始标记
因此「在 class∈{0,3} 的字符之前切」得到神经词界，再按官方词表做相邻词最长合并。

用法::

    from openshenzhou import wb_model
    bounds = wb_model.word_bounds(list("未预配上下文"))   # [False, True, False, True, ...]
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data", "wb_bilstm_weights.npz")
OFFICIAL_WORDS = os.path.join(ROOT, "data", "official_words_zh.json")
MERGE_EVIDENCE = os.path.join(ROOT, "data", "zh_merge_evidence.json")
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
    """合并候选词表：优先用「引擎语料实际产出的词」（data/zh_engine_words.json），
    否则退回官方词表。引擎只在自身词典词上做合并，官方词表范围过宽。"""
    if "lex" in _cache:
        return _cache["lex"]
    words = None
    if os.environ.get("OS_ENGINE_WORDS", "0") == "1":
        try:
            p = os.path.join(ROOT, "data", "zh_engine_words.json")
            d = json.load(open(p, encoding="utf-8"))
            if d:
                words = set(d.keys())
        except Exception:
            words = None
    if words is None:
        try:
            words = set(json.load(open(OFFICIAL_WORDS, encoding="utf-8")))
        except Exception:
            words = set()
    _cache["lex"] = words
    return words


def char_row(ch):
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FCC:
        return 6 + (o - 0x4E00)
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
    """逐字类别（0=B 1=M 2=E 3=S 语义，但按统计规则使用）。"""
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
        Hf[s:s + m] = _run_lstm(X, w["w_fwd"], w["b_fwd"], reverse=False)
        Hb[s:s + m] = _run_lstm(X, w["w_bwd"], w["b_bwd"], reverse=True)
    H = np.concatenate([Hf, Hb], 1)
    logits = H @ w["w_out"] + w["b_out"]
    cls = logits.argmax(1)
    return [int(x) for x in cls[1:1 + n]]


def _segments_from_classes(cls):
    """在 class∈{0,3} 的字符之前切"""
    segs = []
    st = 0
    for j in range(1, len(cls)):
        if cls[j] in (0, 3):
            segs.append((st, j - st))
            st = j
    segs.append((st, len(cls) - st))
    return segs


MERGE_RATE_THRESHOLD = float(os.environ.get("OS_MERGE_TH", "0.1"))
MERGE_RATE_DEFAULT = float(os.environ.get("OS_MERGE_DEFAULT", "0.5"))
_MERGE_DEFAULT_ENV = os.environ.get("OS_MERGE_DEFAULT")


def _load_merge_evidence():
    """官方词表里「有独立词证据」的子集：jieba 词典/引擎词表/引擎语料词频任一说它是词。

    用于合并率缺失时的兜底判定：引擎只在候选确实是词时才合并（如 为时已晚、新生），
    而不会合并非词短语（如 来做、的需要）。
    """
    if "ev" in _cache:
        return _cache["ev"]
    try:
        ev = set(json.load(open(MERGE_EVIDENCE, encoding="utf-8")))
    except Exception:
        ev = None
    _cache["ev"] = ev
    return ev


_PROBE = None


def _load_probe_words():
    """引擎词表探针结果：word → token 数（1=引擎整体认作一个词）。
    来自 tools/probe_zh_words_batch.py（把候选词当句子发给引擎，读 encoder 探针）。"""
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    if os.environ.get("OS_ZH_PROBE", "1") == "0":
        _PROBE = {}
        return _PROBE
    try:
        d = json.load(open(os.path.join(ROOT, "data", "zh_engine_probe.json"), encoding="utf-8"))
        _PROBE = d if isinstance(d, dict) else {}
    except Exception:
        _PROBE = {}
    return _PROBE


def _load_merge_params():
    if "rates" in _cache and "threshold" in _cache:
        return _cache["rates"], _cache["threshold"]
    try:
        d = json.load(open(os.path.join(ROOT, "data", "zh_merge_rates.json"), encoding="utf-8"))
        rates = d.get("rates", {})
        th = d.get("threshold")
    except Exception:
        rates, th = {}, None
    _cache["rates"] = rates
    _cache["threshold"] = th
    return rates, th


def _load_merge_rates():
    return _load_merge_params()[0]


def _merge_by_lexicon(text, segs, words, max_len=MAX_MERGE_LEN):
    """相邻词合并：候选词在官方词表中，且「引擎合并率」≥ 阈值才合并。

    合并率来自引擎语料统计（tools/tune_merge.py 生成）：
    引擎对哪些词会合并、哪些会拆开，直接反映其 Trie 后处理行为。
    """
    rates, file_th = _load_merge_params()
    env_th = os.environ.get("OS_MERGE_TH")
    if env_th not in (None, ""):
        th = float(env_th)
    elif file_th is not None:
        th = float(file_th)
    else:
        th = MERGE_RATE_THRESHOLD
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        probe = _load_probe_words()
        while i < len(segs):
            best = None
            for k in range(len(segs) - 1, i, -1):
                end = segs[k][0] + segs[k][1]
                cand = text[segs[i][0]:end]
                if len(cand) > max_len:
                    continue
                if cand not in words and cand not in probe:
                    continue
                probe = _load_probe_words()
                pv = probe.get(cand)
                if pv is not None:
                    ok = (pv == 1)
                    if ok:
                        best = k
                        break
                    continue
                rate = rates.get(cand)
                if rate is None:
                    if _MERGE_DEFAULT_ENV is not None:
                        ok = MERGE_RATE_DEFAULT >= th
                    else:
                        ev = _load_merge_evidence()
                        ok = (cand in ev) if ev is not None else True
                else:
                    ok = rate >= th
                if ok:
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
    """字符列表 → [(start, len)] 词序列"""
    text = "".join(chars)
    cls = classes(chars)
    segs = _segments_from_classes(cls)
    return _merge_by_lexicon(text, segs, _load_words())


def bounds_from_words(chars, words):
    """由词序列（字符串列表）生成长度 n-1 的边界标志列表。"""
    n = len(chars)
    if n <= 1:
        return []
    bounds = [False] * (n - 1)
    pos = 0
    for w in words:
        pos += len(w)
        if 0 < pos < n:
            bounds[pos - 1] = True
    return bounds


def word_bounds(chars):
    """返回长度 n-1 的布尔列表：第 k 个 True 表示第 k 个字之后存在词边界。"""
    n = len(chars)
    if n <= 1:
        return []
    return bounds_from_words(chars, [''.join(chars[s:s + L]) for s, L in segments(chars)])


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for text in sys.argv[1:] or ["未预配上下文", "本地回显关闭"]:
        segs = segments(list(text))
        print(text, "->", [text[s:s + L] for s, L in segs])
