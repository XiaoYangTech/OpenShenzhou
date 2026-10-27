#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_zh_large.py —— 中文大规模泛化评测（3000 句引擎实测 golden）

语料：data/eval/zh_large_eval.jsonl（tts-engine/data/seg_B.jsonl 留出抽样，
     与中文 599 评测集、合并率训练语料均不重叠）

用法::
    OS_VOICE_ROOT=... OS_SDK_ROOT=<python-sdk> python tools/eval_zh_large.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402


def main():
    from openshenzhou import g2p_zh as gz

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = gz.load_symbols(os.path.join(pkg, "phones.txt"))
    table = gz.load_table()
    lex = gz.load_lexicon()
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "eval",
                                                              "zh_large_eval.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    def first_sentence(ids):
        out = []
        for x in ids:
            out.append(x)
            if x == 1:
                break
        return out

    ok = n = 0
    for r in rows:
        pred = gz.text_to_ids(r["text"], table, sym, lex)
        if pred is None:
            continue
        n += 1
        if first_sentence(pred) == r["ids"]:
            ok += 1
    print("中文大规模逐 id 一致: %d/%d = %.2f%%" % (ok, n, 100.0 * ok / max(1, n)))
    return 0 if ok / max(1, n) >= 0.93 else 1


if __name__ == "__main__":
    sys.exit(main())
