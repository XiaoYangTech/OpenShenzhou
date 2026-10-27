#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmodel.py —— 中文分词边界模型（哈希特征 + 逻辑回归，词典分词做基底）

思路：以词典最长匹配分词为「基底」，模型学习对它做**修正**（合并/拆开）。
特征里带有基底边界、左右词长、词表命中情况，因此至少不会比基底差。

    m = SegModel.load()
    labs = m.predict(chars)      # len(chars)-1 个 0/1，1 = 断词（br1）
"""
import json
import os
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUNCT = set("，。！？、；：,.;!?…—－《》「」『』（）()【】 \t")


def load_lex_words():
    p = os.path.join(ROOT, "data", "lexicon_zh.names.json")
    try:
        return set(json.load(open(p, encoding="utf-8")).keys())
    except Exception:
        return set()


def load_official_words():
    p = os.path.join(ROOT, "data", "official_words_zh.json")
    try:
        return set(json.load(open(p, encoding="utf-8")))
    except Exception:
        return set()


_WORDSET = None


def wordset():
    global _WORDSET
    if _WORDSET is None:
        p = os.path.join(ROOT, "data", "zh_seg_wordset.json")
        try:
            _WORDSET = set(json.load(open(p, encoding="utf-8")))
        except Exception:
            _WORDSET = load_lex_words() | load_official_words()
    return _WORDSET


_ENGINE_WORDS = None


def engine_words():
    """引擎自己的切词词表（word -> 频次），由 tools/build_engine_words.py 生成"""
    global _ENGINE_WORDS
    if _ENGINE_WORDS is None:
        path = os.path.join(ROOT, "data", "zh_engine_words.json")
        try:
            _ENGINE_WORDS = json.load(open(path, encoding="utf-8"))
        except Exception:
            _ENGINE_WORDS = {}
    return _ENGINE_WORDS


def _freq_bucket(c):
    if c <= 0:
        return 0
    if c == 1:
        return 1
    if c <= 5:
        return 2
    if c <= 20:
        return 3
    return 4


def baseline_words(chars, maxlen=6):
    """词典最长匹配（和 g2p_zh.segment_words 一致），返回词的字符区间列表"""
    ws = wordset()
    out = []
    i = 0
    n = len(chars)
    while i < n:
        hit = None
        for L in range(min(maxlen, n - i), 0, -1):
            cand = "".join(chars[i:i + L])
            if cand in ws:
                hit = L
                break
        if hit is None:
            hit = 1
        out.append((i, i + hit))
        i += hit
    return out


def baseline_labels(nchars, words):
    labs = [0] * max(0, nchars - 1)
    for si in range(1, len(words)):
        pos = words[si][0] - 1
        if 0 <= pos < len(labs):
            labs[pos] = 1
    return labs


class SegModel:
    def __init__(self, W, dim, max_word_len=4):
        self.W = W
        self.dim = dim
        self.lex = load_lex_words()
        self.max_word_len = max_word_len

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(ROOT, "data", "zh_seg_model.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return cls(d["W"].astype(np.float32), int(d["dim"]))

    # ---------------- 特征 ----------------
    def featurize(self, chars, i, base_labs, words):
        n = len(chars)

        def ch(k):
            if k < 0:
                return "^"
            if k >= n:
                return "$"
            return chars[k]

        f = []
        for d in range(-2, 4):
            f.append("c%d=%s" % (d, ch(i + d)))
        f.append("b0=%s%s" % (ch(i), ch(i + 1)))
        f.append("b-1=%s%s" % (ch(i - 1), ch(i)))
        f.append("b1=%s%s" % (ch(i + 1), ch(i + 2)))
        f.append("t-1=%s%s%s" % (ch(i - 1), ch(i), ch(i + 1)))
        f.append("t0=%s%s%s" % (ch(i), ch(i + 1), ch(i + 2)))
        # 词表命中
        for L in range(1, self.max_word_len + 1):
            for s in range(1, L):
                start = i - s + 1
                end = start + L
                if start < 0 or end > n:
                    continue
                w = "".join(chars[start:end])
                if w in self.lex:
                    f.append("w%d_%d=%s" % (L, s, w))
        # 基底分词特征
        bl = base_labs[i] if 0 <= i < len(base_labs) else 0
        f.append("base=%d" % bl)
        left = None
        right = None
        for (s0, e0) in words:
            if s0 < i + 1 <= e0:
                left = (s0, e0)
            if s0 <= i + 1 < e0:
                right = (s0, e0)
        if left is None:
            left = (i, i + 1)
        if right is None:
            right = (i + 1, min(n, i + 2))
        lw = "".join(chars[left[0]:left[1]])
        rw = "".join(chars[right[0]:right[1]])
        f.append("lwlen=%d" % len(lw))
        f.append("rwlen=%d" % len(rw))
        f.append("lw=%s" % lw)
        f.append("rw=%s" % rw)
        f.append("pair=%s|%s" % (lw, rw))
        f.append("pos_in_lw=%d" % (i + 1 - left[0]))
        f.append("pos=%d" % min(3, int(4.0 * i / max(1, n))))
        f.append("len=%d" % min(5, n // 5))
        if i == 0:
            f.append("at_start")
        if i == n - 2:
            f.append("at_end")
        return f

    def scores(self, chars, base_labs=None, words=None):
        """返回每个 gap 的 logit；base_labs/words 缺省时现场算"""
        n = len(chars)
        if n < 2:
            return np.zeros(0, dtype=np.float32)
        if words is None:
            words = baseline_words(chars)
        if base_labs is None:
            base_labs = baseline_labels(n, words)
        out = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            feats = self.featurize(chars, i, base_labs, words)
            idx = [zlib.crc32(s.encode("utf-8")) % self.dim for s in feats]
            out[i] = float(self.W[idx].sum())
        return out

    def predict(self, chars, base_labs=None, words=None):
        return (self.scores(chars, base_labs, words) > 0).astype(np.int8)
