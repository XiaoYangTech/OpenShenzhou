#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_first_sentence.py —— 用「引擎首句标签」评估前端

背景：采集探针 fast 模式在每句 encoder 后中止，因此多句文本只拿到首句标签；
但引擎本身就是"每句一次 encoder"，首句标签是完整可比的。
本脚本：把完整文本过一遍我们的前端，取它输出的第一句（到第一个 '~'），
与引擎首句标签逐 id 比较。

用法: python tools/eval_first_sentence.py data/eval_cap.jsonl
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", "OpenShenzhou"))
sys.path.insert(0, ROOT)

import openshenzhou.g2p_zh as gz  # noqa: E402

END = 1


def first_sentence(ids):
    out = []
    for x in ids:
        out.append(x)
        if x == END:
            break
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "eval", "zh_eval_599.jsonl")
    pkg = os.path.join(os.environ["TEMP"], "AllVoice", "Xiaoxiao_Female_zh-CN")
    sym = gz.load_symbols(os.path.join(pkg, "phones.txt"))
    table = gz.load_table()
    lex = gz.load_lexicon()
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = n = 0
    for r in rows:
        pred = gz.text_to_ids(r["text"], table, sym, lex)
        if pred is None:
            continue
        n += 1
        if first_sentence(pred) == r["ids"]:
            ok += 1
    print("首句逐 id 一致: %d/%d = %.1f%%" % (ok, n, 100.0 * ok / max(1, n)))


if __name__ == "__main__":
    main()
