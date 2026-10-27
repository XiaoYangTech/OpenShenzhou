#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_char_rules.py —— 从引擎语料学习「多音字 → 上下文读音」高置信规则

对每个多音字，统计：
  * (前字, 字) → 读音比例
  * (字, 后字) → 读音比例
  * (前字, 字, 后字) → 读音比例
只保留 count ≥ min_count 且比例 ≥ ratio 的规则（极端确定的情形），
用于在 SDK 中对引擎读音做最终覆盖（例如「为」在「将」后读 wèi、「应」在「本」后读 yīng）。
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
    min_count = int(os.environ.get("OS_MIN_RULE", "20"))
    ratio = float(os.environ.get("OS_RATIO_RULE", "0.98"))
    sym = load_symbols(os.path.join(HERE, "models", "zh-CN", "lexicon", "phones.txt"))
    id2sym = {v: k for k, v in sym.items()}
    uni = defaultdict(Counter)
    prev_c = defaultdict(Counter)
    next_c = defaultdict(Counter)
    tri = defaultdict(Counter)
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
                rep = list(syls[i])
                uni[ch][tuple(rep)] += 1
                p = text[i - 1] if i > 0 else "^"
                q = text[i + 1] if i + 1 < len(text) else "$"
                prev_c[(p, ch)][tuple(rep)] += 1
                next_c[(ch, q)][tuple(rep)] += 1
                tri[(p, ch, q)][tuple(rep)] += 1
    print("句子", n_sent)

    def top(cnt):
        total = sum(cnt.values())
        if total < min_count:
            return None
        rep, c = cnt.most_common(1)[0]
        if c / total < ratio:
            return None
        return list(rep), total, round(c / total, 3)

    # 只保留多音字
    poly = {ch for ch, cnt in uni.items()
            if len(cnt) >= 2 and sum(cnt.values()) >= 20}
    print("多音字", len(poly))
    rules = {"tri": {}, "prev": {}, "next": {}}
    hits = 0
    for (p, ch, q), cnt in tri.items():
        if ch not in poly:
            continue
        r = top(cnt)
        if r:
            rules["tri"]["%s|%s|%s" % (p, ch, q)] = {"rep": list(r[0]), "n": r[1], "p": r[2]}
            hits += 1
    for (p, ch), cnt in prev_c.items():
        if ch not in poly:
            continue
        r = top(cnt)
        if r:
            rules["prev"]["%s|%s" % (p, ch)] = {"rep": list(r[0]), "n": r[1], "p": r[2]}
    for (ch, q), cnt in next_c.items():
        if ch not in poly:
            continue
        r = top(cnt)
        if r:
            rules["next"]["%s|%s" % (ch, q)] = {"rep": list(r[0]), "n": r[1], "p": r[2]}
    json.dump(rules, open(os.path.join(OUT, "zh_char_rules.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("规则: tri %d, prev %d, next %d" % (
        len(rules["tri"]), len(rules["prev"]), len(rules["next"])))


if __name__ == "__main__":
    main()
