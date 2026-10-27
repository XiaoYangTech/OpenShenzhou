#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_single_char_lexicon.py —— 用语料多数读音修正单字词表项

只看「单字词」且语料证据充分（count>=30 且主读音占比>=0.9）的条目；
避免覆盖多音字在不同语境下的少数读音（由上下文表/CRF 处理）。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "data", "lexicon_zh.names.json")


def main():
    lex = json.load(open(PATH, encoding="utf-8"))
    rs = json.load(open(os.path.join(HERE, "data", "zh_char_reading_sets.json"),
                        encoding="utf-8"))
    changed = 0
    for ch, cnt in rs.items():
        if ch not in lex or len(ch) != 1:
            continue
        total = sum(cnt.values())
        if total < 30:
            continue
        best, n = max(cnt.items(), key=lambda kv: kv[1])
        if n / total < 0.9:
            continue
        rep = best.split(".")
        cur = lex[ch]
        if len(cur) == 3 and list(cur) != rep and isinstance(cur[0], str):
            lex[ch] = rep
            changed += 1
    json.dump(lex, open(PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print("updated single-char entries", changed)


if __name__ == "__main__":
    main()
