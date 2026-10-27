#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_broad_lex.py —— 从大规模引擎 golden 构建「词 → 引擎读音」词表

数据：tts-engine/data/seg_*.jsonl（排除 eval_cap.jsonl 与留出集 seg_B.jsonl）
输出：data/lexicon_zh_broad.names.json  {词: [音素名...]}（平铺，每 3 个音素一个音节）

用途：g2p_zh 的词典读音优先用引擎实测读音，覆盖宽领域词（多音字/轻声/变调）。
"""
import glob
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, os.path.join(HERE, "tools"))

from _paths import voice_pkg  # noqa: E402

CJK = lambda c: "\u4e00" <= c <= "\u9fff"
MIN_COUNT = 1


def main():
    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    rev = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        rev[int(i)] = n

    counts = {}  # word -> Counter(tuple(reading))
    kept_sent = 0
    skipped = 0
    files = sorted(glob.glob(os.path.join(ENGINE, "data", "seg_*.jsonl")))
    for fn in files:
        base = os.path.basename(fn)
        if base in ("eval_cap.jsonl", "seg_B.jsonl"):
            continue
        for line in open(fn, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r["text"]
            chars = [c for c in text if CJK(c)]
            if len(chars) < 2:
                continue
            # token 列表：flat phones 与 syllable 数
            toks = []
            cur = []
            for x in r["ids"]:
                nm = rev.get(x, "?")
                if nm in ("-bos-", "br0", "br2", "br3", "br4", "~") or nm.startswith("punc"):
                    continue
                if nm == "br1":
                    if cur:
                        toks.append(cur)
                    cur = []
                    continue
                if nm == "-":
                    continue
                cur.append(nm)
            if cur:
                toks.append(cur)
            pos = 0
            ok = True
            spans = []
            for t in toks:
                if len(t) % 3 != 0:
                    ok = False
                    break
                syl = len(t) // 3
                if pos + syl > len(chars):
                    ok = False
                    break
                word = "".join(chars[pos:pos + syl])
                spans.append((word, tuple(t)))
                pos += syl
            if not ok or pos != len(chars):
                skipped += 1
                continue
            kept_sent += 1
            for w, rd in spans:
                if len(w) > 8:
                    continue
                d = counts.setdefault(w, Counter())
                d[rd] += 1

    print("sentences kept", kept_sent, "skipped", skipped, "unique words", len(counts))
    out = {}
    entries = 0
    for w, cnt in counts.items():
        total = sum(cnt.values())
        if total < MIN_COUNT:
            continue
        rd, n = cnt.most_common(1)[0]
        out[w] = list(rd)
        entries += 1
    path = os.path.join(HERE, "data", "lexicon_zh_broad.names.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", path, "entries", entries, "size MB %.1f" % (os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
