#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_token_fix.py —— 从引擎语料构建「词→引擎实测读音」修正表

只保留：出现 ≥ min_count 次、单一读音占比 ≥ ratio 的引擎词 token。
用于修正多音字词（词典标注读音与引擎上下文读音不一致的情况）。

输出 data/zh_token_fix.json：{词: [[符号,符号,符号], ...]}
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


def eng_segs(lab):
    out = []
    i = 0
    while i < len(lab):
        if lab[i] == "S":
            out.append((i, 1))
            i += 1
            continue
        j = i
        while j < len(lab) and lab[j] in "BM":
            j += 1
        if j < len(lab) and lab[j] == "E":
            out.append((i, j - i + 1))
            i = j + 1
        else:
            out.append((i, j - i))
            i = j
    return out


def main():
    min_count = int(os.environ.get("OS_MIN_COUNT", "8"))
    ratio = float(os.environ.get("OS_RATIO", "0.95"))
    sym = load_symbols(os.path.join(HERE, "models", "zh-CN", "lexicon", "phones.txt"))
    id2sym = {v: k for k, v in sym.items()}
    tok = defaultdict(Counter)
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
            for st, L in eng_segs(labs):
                if st + L > len(text):
                    continue
                w = text[st:st + L]
                rep = tuple(syls[k] for k in range(st, st + L))
                tok[w][rep] += 1
    # 多音字集合：语料中同一位置出现 ≥2 种读音的字
    char_reads = defaultdict(Counter)
    for w, cnt in tok.items():
        for rep, c in cnt.items():
            if len(rep) != len(w):
                continue
            for k, ch in enumerate(w):
                char_reads[ch][rep[k]] += c
    poly = {ch for ch, rd in char_reads.items()
            if len(rd) >= 2 and sum(rd.values()) >= 20}
    print("多音字", len(poly))
    out = {}
    for w, cnt in tok.items():
        if not any(ch in poly for ch in w):
            continue
        total = sum(cnt.values())
        if total < min_count:
            continue
        rep, c = cnt.most_common(1)[0]
        if c / total < ratio:
            continue
        if len(rep) != len(w):
            continue
        if any(len(p) != 3 for p in rep):
            continue
        out[w] = [list(p) for p in rep]
    json.dump(out, open(os.path.join(OUT, "zh_token_fix.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print("词条", len(out), "（min_count=%d ratio=%.2f）" % (min_count, ratio))


if __name__ == "__main__":
    main()
