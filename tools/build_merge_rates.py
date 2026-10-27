#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_merge_rates.py —— 用全部语料统计「引擎词合并率」，生成 data/zh_merge_rates.json

训练语料默认排除 eval_cap.jsonl（评测集）与 seg_B.jsonl（留出集）；
可用 --include-heldout 把留出集也加入（最终发布用）。
"""
import glob
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, os.path.join(ENGINE, "tools"))
sys.path.insert(0, HERE)

import wb_infer3 as M  # noqa: E402

PUNCT = set("，。！？、；：,.;!?…—－《》『』（）()【】 \t")
CACHE = os.path.join(ENGINE, "data", "wb_classes_cache.json")
ALPHA = 0.1


def load_rows(fn, limit=10 ** 9):
    out = []
    for line in open(fn, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        text = r["text"]
        if any(c in PUNCT for c in text):
            continue
        if not (4 <= len(text) <= 40):
            continue
        if not all(0x4E00 <= ord(c) <= 0x9FCC for c in text):
            continue
        labs = M.engine_labels(r["ids"])
        if len(labs) != len(text):
            continue
        out.append((text, labs))
        if len(out) >= limit:
            break
    return out


def eng_bounds(lab):
    b = set()
    for i, l in enumerate(lab):
        if l in "ES" and i + 1 < len(lab):
            b.add(i + 1)
    return b


def r1_segs(cls):
    segs = []
    st = 0
    for j in range(1, len(cls)):
        if cls[j] in (0, 3):
            segs.append((st, j - st))
            st = j
    segs.append((st, len(cls) - st))
    return segs


def main():
    include_heldout = "--include-heldout" in sys.argv
    Wd = M.load()
    try:
        cache = json.load(open(CACHE, encoding="utf-8"))
    except Exception:
        cache = {}

    def get_cls(text):
        cls = cache.get(text)
        if cls is None:
            pred = M.infer(Wd, text, "chunk")
            cls = [int(x) for x in pred[1:1 + len(text)]]
            cache[text] = cls
        return cls

    official = set(json.load(open(os.path.join(HERE, "data", "official_words_zh.json"), encoding="utf-8")))
    files = sorted(glob.glob(os.path.join(ENGINE, "data", "seg_*.jsonl")))
    use = []
    for fn in files:
        base = os.path.basename(fn)
        if base == "eval_cap.jsonl":
            continue
        if base == "seg_B.jsonl" and not include_heldout:
            continue
        use.append(fn)
    print("训练语料:", [os.path.basename(x) for x in use])
    m_cnt = Counter()
    s_cnt = Counter()
    n_sent = 0
    for fn in use:
        rows = load_rows(fn)
        for text, lab in rows:
            n_sent += 1
            cls = get_cls(text)
            segs = r1_segs(cls)
            eb = eng_bounds(lab)
            for i in range(len(segs)):
                for k in range(i + 1, min(len(segs), i + 5)):
                    end = segs[k][0] + segs[k][1]
                    cand = text[segs[i][0]:end]
                    if cand not in official:
                        continue
                    inner = [segs[j][0] for j in range(i + 1, k + 1)]
                    if all(b not in eb for b in inner):
                        m_cnt[cand] += 1
                    else:
                        s_cnt[cand] += 1
    print("句子", n_sent, "合并候选", len(m_cnt), "拆分候选", len(s_cnt))
    rates = {}
    for w in set(m_cnt) | set(s_cnt):
        m = m_cnt.get(w, 0)
        s = s_cnt.get(w, 0)
        denom = m + s + 2 * ALPHA
        rates[w] = round(((m + ALPHA) / denom) if denom > 0 else 0.0, 4)
    json.dump({"alpha": ALPHA, "rates": rates},
              open(os.path.join(HERE, "data", "zh_merge_rates.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("已保存 data/zh_merge_rates.json，词条", len(rates))


if __name__ == "__main__":
    main()
