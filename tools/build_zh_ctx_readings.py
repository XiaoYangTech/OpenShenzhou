#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_ctx_readings.py —— 从引擎语料构建「词级上下文读音」表

对每个引擎词 token（由 B/M/E/S 标签切分）记录：
    key = 前词|词|后词
    value = 该词在语料中最常见的读音（逐音节 3 符号）
只保留出现次数 ≥ min_count 的上下文。

输出 data/zh_ctx_readings.json
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
            out.append((i, max(1, j - i)))
            i = j
    return out


def main():
    min_count = int(os.environ.get("OS_CTX_MIN", "2"))
    sym = load_symbols(os.path.join(HERE, "models", "zh-CN", "lexicon", "phones.txt"))
    id2sym = {v: k for k, v in sym.items()}
    stat = defaultdict(Counter)
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
            segs = eng_segs(labs)
            words = [text[s:s + L] for s, L in segs]
            reps = [tuple(syls[s:s + L]) for s, L in segs]
            for wi, w in enumerate(words):
                prev_w = words[wi - 1] if wi > 0 else "^"
                next_w = words[wi + 1] if wi + 1 < len(words) else "$"
                stat["%s|%s|%s" % (prev_w, w, next_w)][reps[wi]] += 1
    print("句子", n_sent, "上下文 key", len(stat))
    out = {}
    for key, cnt in stat.items():
        total = sum(cnt.values())
        if total < min_count:
            continue
        rep, c = cnt.most_common(1)[0]
        out[key] = [list(x) for x in rep]
    print("保留", len(out), "条 (min_count=%d)" % min_count)
    json.dump(out, open(os.path.join(OUT, "zh_ctx_readings.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    print("已写入 data/zh_ctx_readings.json")


if __name__ == "__main__":
    main()
