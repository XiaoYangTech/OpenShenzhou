#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_engine_wb.py —— 复刻版 CWordBreaker（tools/engine_wb.py）与引擎真值词序列对比。

数据: data/eval/zh_cw_599.json（Frida hook 抓取的引擎 CWordBreaker 输出，597 句）
用法::
    OS_SDK_ROOT=<python-sdk> python tools/eval_engine_wb.py [--show N]
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

import engine_wb  # noqa: E402

PUNCT = set('。！？.!?，、；：,;:')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    cw_map = json.load(open(os.path.join(HERE, "data", "eval", "zh_cw_599.json"),
                            encoding="utf-8"))
    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "data", "eval", "zh_eval_599.jsonl"),
                 encoding="utf-8") if l.strip()]
    total = ok = 0
    shown = 0
    for r in rows:
        t = r["text"].strip()
        i = 0
        while i < len(t) and t[i] not in "。！？.!?":
            i += 1
        sent = t[:i + 1] if i < len(t) else t
        if sent not in cw_map:
            continue
        true_cw = [w for w in cw_map[sent] if w and not all(ch in PUNCT for ch in w)]
        spans = engine_wb.word_spans(sent)
        ours = []
        for a, L in spans:
            w = "".join(c for c in sent[a:a + L] if "\u4e00" <= c <= "\u9fff" or c == "〇")
            if w:
                ours.append(w)
        total += 1
        if ours == true_cw:
            ok += 1
        elif shown < args.show:
            shown += 1
            print("差异:", sent)
            print("  engine:", true_cw)
            print("  repl  :", ours)
    print("CWordBreaker 复刻一致: %d/%d = %.1f%%"
          % (ok, total, 100.0 * ok / total if total else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
