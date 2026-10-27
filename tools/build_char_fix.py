#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_char_fix.py —— 从引擎语料统计「单字读音」修正表

输出 data/zh_char_fix.json：
  {
    "uni": {字: [符号,符号,符号]},                    # 该字在语料中压倒性多数读音
    "tri": {"前字|字|后字": [符号,...]},               # 上下文决定的多音字读音
  }
用法：SDK 中单字成词时优先查此表（在旧 ctx / 字表之前）。
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
OUT = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(ENGINE, "tools"))

import wb_infer3 as M  # noqa: E402

PUNCT = set("，。！？、；：,.;!?…—－《》「」『』（）()【】 \t")
TAGS = {17, 18, 25, 39, 68}


def load_symbols(path):
    sym = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        sym[n] = int(i)
    return sym


def main():
    sym_path = os.path.join(HERE, "models", "zh-CN", "lexicon", "phones.txt")
    sym = load_symbols(sym_path)
    id2sym = {v: k for k, v in sym.items()}
    uni = defaultdict(Counter)
    tri = defaultdict(Counter)
    bi_l = defaultdict(Counter)
    bi_r = defaultdict(Counter)
    n_sent = 0
    for fn in sorted(glob.glob(os.path.join(ENGINE, "data", "seg_*.jsonl"))):
        for line in open(fn, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r["text"]
            if not text or len(text) > 40:
                continue
            if any(c in PUNCT for c in text):
                continue
            if not all(0x4E00 <= ord(c) <= 0x9FCC for c in text):
                continue
            labs = M.engine_labels(r["ids"])
            if len(labs) != len(text):
                continue
            toks = [id2sym.get(x) for x in r["ids"]
                    if x not in TAGS and x not in (1, 2, 3)]
            toks = [t for t in toks if t and not t.startswith("punc")]
            if len(toks) % 3:
                continue
            syls = [tuple(toks[i:i + 3]) for i in range(0, len(toks), 3)]
            if len(syls) != len(text):
                continue
            n_sent += 1
            for i, ch in enumerate(text):
                rep = ".".join(syls[i])
                uni[ch][rep] += 1
                prev_c = text[i - 1] if i > 0 else "^"
                next_c = text[i + 1] if i + 1 < len(text) else "$"
                tri[(prev_c, ch, next_c)][rep] += 1
                bi_l[(prev_c, ch)][rep] += 1
                bi_r[(ch, next_c)][rep] += 1
    print("可用句", n_sent, "单字", len(uni))

    def top(cnt, min_count, ratio):
        total = sum(cnt.values())
        if total < min_count:
            return None
        rep, c = cnt.most_common(1)[0]
        if c / total < ratio:
            return None
        return rep

    uni_out = {}
    poly = set()
    for ch, cnt in uni.items():
        if len(cnt) < 2:
            continue  # 只处理多读音字
        poly.add(ch)
        total = sum(cnt.values())
        if total < 20:
            continue
        rep = top(cnt, 20, 0.90)
        if rep:
            uni_out[ch] = rep.split(".")
    tri_out = {}
    for (prev_c, ch, next_c), cnt in tri.items():
        if ch not in poly:
            continue
        rep = top(cnt, 3, 0.90)
        if rep:
            tri_out["%s|%s|%s" % (prev_c, ch, next_c)] = rep.split(".")
    bi_l_out = {}
    for (prev_c, ch), cnt in bi_l.items():
        if ch not in poly:
            continue
        rep = top(cnt, 8, 0.95)
        if rep:
            bi_l_out["%s|%s" % (prev_c, ch)] = rep.split(".")
    bi_r_out = {}
    for (ch, next_c), cnt in bi_r.items():
        if ch not in poly:
            continue
        rep = top(cnt, 8, 0.95)
        if rep:
            bi_r_out["%s|%s" % (ch, next_c)] = rep.split(".")
    data = {"uni": uni_out, "tri": tri_out, "bi_l": bi_l_out, "bi_r": bi_r_out}
    json.dump(data, open(os.path.join(OUT, "zh_char_fix.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print("多音字 %d, uni %d, tri %d, bi_l %d, bi_r %d" % (len(poly),
        len(uni_out), len(tri_out), len(bi_l_out), len(bi_r_out)))


if __name__ == "__main__":
    main()
