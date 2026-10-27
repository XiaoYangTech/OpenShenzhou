#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_probe_priority.py —— 生成中文「词表探针」优先候选词

来源（不使用 599 / 大规模评测集，保持评测独立）：
  1) data/zh_merge_rates.json 里已有合并率统计的全部候选词（引擎 UI 语料统计）
  2) data/eval/zh_seg_cap*.jsonl（新抓的 3000 句小说域引擎 golden）中，
     跨 wb 片段边界、且属于官方词表的候选词

输出 data/eval/zh_probe_priority.txt（每行一词，供 probe_zh_words_batch.py 批量探测）。
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
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402

CJK = lambda c: "\u4e00" <= c <= "\u9fff"


def load_sym(pkg):
    m = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        m[n] = int(i)
    return m


def decode_words(text, ids, sym):
    rev = {v: k for k, v in sym.items()}
    chars = [c for c in text if CJK(c)]
    labels = [0] * max(0, len(chars) - 1)
    pos = syl = 0
    started = False
    for x in ids:
        nm = rev.get(x)
        if nm is None:
            continue
        if nm in ("-bos-", "br0", "br2", "br3", "br4", "~") or nm.startswith("punc"):
            continue
        if nm == "br1":
            if started:
                pos += syl + 1
                if 0 < pos <= len(chars) and pos - 1 < len(labels):
                    labels[pos - 1] = 1
            syl = 0
            started = False
            continue
        if nm == "-":
            syl += 1
            started = True
        else:
            started = True
    if started:
        pos += syl + 1
    return (chars, labels) if pos == len(chars) else None


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
    from openshenzhou import wb_model
    import wb_infer3 as M

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = load_sym(pkg)
    official = set(json.load(open(os.path.join(HERE, "data", "official_words_zh.json"),
                                  encoding="utf-8")))
    Wd = M.load()
    try:
        cache = json.load(open(os.path.join(ENGINE, "data", "wb_classes_cache.json"),
                               encoding="utf-8"))
    except Exception:
        cache = {}

    def cls_of(text):
        c = cache.get(text)
        if c is None:
            pred = M.infer(Wd, text, "chunk")
            c = [int(x) for x in pred[1:1 + len(text)]]
            cache[text] = c
        return c

    cand = Counter()
    # 1) 现有合并率候选
    try:
        rates = json.load(open(os.path.join(HERE, "data", "zh_merge_rates.json"),
                               encoding="utf-8")).get("rates", {})
        for w in rates:
            if 2 <= len(w) <= 8 and all(CJK(c) for c in w):
                cand[w] += 1000
    except Exception:
        pass

    # 2) 小说域新 golden 的候选
    m_cnt, s_cnt = Counter(), Counter()
    n_ok = 0
    for f in sorted(glob.glob(os.path.join(HERE, "data", "eval", "zh_seg_cap*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            dec = decode_words(r["text"], r["ids"], sym)
            if not dec:
                continue
            chars, labels = dec
            n_ok += 1
            text = "".join(chars)
            cls = cls_of(text)
            segs = r1_segs(cls)
            eb = set(i for i, v in enumerate(labels) if v == 1)
            for i in range(len(segs)):
                for k in range(i + 1, min(len(segs), i + 4)):
                    end = segs[k][0] + segs[k][1]
                    w = text[segs[i][0]:end]
                    if not (2 <= len(w) <= 8) or not all(CJK(c) for c in w):
                        continue
                    if w not in official:
                        continue
                    inner = [segs[j][0] for j in range(i + 1, k + 1)]
                    if all((b - 1) not in eb for b in inner):
                        m_cnt[w] += 1
                    else:
                        s_cnt[w] += 1
    print("novel sentences", n_ok, "candidates", len(m_cnt), len(s_cnt))
    for w in set(m_cnt) | set(s_cnt):
        cand[w] += 1
    print("total priority candidates", len(cand))
    out = os.path.join(HERE, "data", "eval", "zh_probe_priority.txt")
    open(out, "w", encoding="utf-8").write("\n".join(sorted(cand)))
    print("saved", out)


if __name__ == "__main__":
    main()
