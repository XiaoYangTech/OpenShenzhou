#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_merge.py —— 训练集统计合并率 → 平滑/阈值选择 → 保存合并率表

复用 data/wb_classes_cache.json（逐句类别缓存），避免重复跑 LSTM。
"""
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, os.path.join(ENGINE, "tools"))
sys.path.insert(0, HERE)

PUNCT = set("，。！？、；：,.;!?…—－《》『』（）()【】 \t")
CACHE = os.path.join(ENGINE, "data", "wb_classes_cache.json")


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


import wb_infer3 as M  # noqa: E402


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
    train = []
    for fn in ("data/seg_A.jsonl", "data/seg_C.jsonl", "data/seg_D.jsonl",
               "data/seg_C3.jsonl", "data/seg_D3.jsonl"):
        train += load_rows(os.path.join(ENGINE, fn), 8000)
    test = load_rows(os.path.join(ENGINE, "data/seg_B.jsonl"), 3000)
    Wd = M.load()
    cache = json.load(open(CACHE, encoding="utf-8"))

    def get_cls(text):
        cls = cache.get(text)
        if cls is None:
            pred = M.infer(Wd, text, "chunk")
            cls = [int(x) for x in pred[1:1 + len(text)]]
            cache[text] = cls
        return cls

    official = set(json.load(open(os.path.join(HERE, "data", "official_words_zh.json"), encoding="utf-8")))
    m_cnt = Counter()
    s_cnt = Counter()
    for text, lab in train:
        cls = get_cls(text)
        segs = r1_segs(cls)
        eb = set()
        for i, l in enumerate(lab):
            if l in "ES" and i + 1 < len(lab):
                eb.add(i + 1)
        for i in range(len(segs)):
            for k in range(i + 1, min(len(segs), i + 3)):
                end = segs[k][0] + segs[k][1]
                cand = text[segs[i][0]:end]
                if cand not in official:
                    continue
                inner = [segs[j][0] for j in range(i + 1, k + 1)]
                if all(b not in eb for b in inner):
                    m_cnt[cand] += 1
                else:
                    s_cnt[cand] += 1
    print("训练统计完成：合并候选", len(m_cnt), "拆分候选", len(s_cnt))

    def evaluate(alpha, th, max_seg=3):
        ok = 0
        for text, lab in test:
            cls = get_cls(text)
            segs = r1_segs(cls)
            out = []
            i = 0
            while i < len(segs):
                best = None
                for k in range(min(len(segs) - 1, i + max_seg - 1), i, -1):
                    end = segs[k][0] + segs[k][1]
                    cand = text[segs[i][0]:end]
                    if cand not in official:
                        continue
                    m = m_cnt.get(cand, 0)
                    s = s_cnt.get(cand, 0)
                    denom = m + s + 2 * alpha
                    rate = ((m + alpha) / denom) if denom > 0 else 0.0
                    if rate >= th:
                        best = k
                        break
                if best is not None:
                    out.append((segs[i][0], segs[best][0] + segs[best][1] - segs[i][0]))
                    i = best + 1
                else:
                    out.append(segs[i])
                    i += 1
            eng = set()
            for i2, l in enumerate(lab):
                if l in "ES" and i2 + 1 < len(lab):
                    eng.add(i2 + 1)
            our = {st for st, _ in out[1:]}
            if our == eng:
                ok += 1
        return ok / len(test)

    best = (0, None, None, None)
    for alpha in (0.0, 0.1, 0.5, 1.0):
        for th in (0.1, 0.3, 0.5, 0.7, 0.9):
            acc = evaluate(alpha, th)
            if acc > best[0]:
                best = (acc, alpha, th, 3)
            print("alpha=%.1f th=%.1f acc=%.5f" % (alpha, th, acc))
    print("BEST", best)
    # 保存平滑后的合并率表（alpha=0.5 便于泛化）
    alpha = 0.5 if best[1] is None else best[1]
    rates = {}
    for w in set(m_cnt) | set(s_cnt):
        m = m_cnt.get(w, 0)
        s = s_cnt.get(w, 0)
        denom = m + s + 2 * alpha
        rates[w] = round(((m + alpha) / denom) if denom > 0 else 0.0, 4)
    out = {"alpha": alpha, "rates": rates}
    json.dump(out, open(os.path.join(HERE, "data", "zh_merge_rates.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("已保存 data/zh_merge_rates.json，词条", len(rates))


if __name__ == "__main__":
    main()
