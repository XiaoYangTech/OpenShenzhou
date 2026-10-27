#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_pinyin_all.py —— 补全「汉字 → 拼音」表（data/pinyin_zh.json）

用途：Python 侧未知字回退用 pypinyin，Node 侧未知字回退查 pinyin_zh.json。
把 pypinyin 对所有 CJK 统一表意文字的默认读音补进表里，保证 Python/Node
对生僻字/乱码字回退一致（否则 Node 用 'a1' 兜底会与 Python 不同）。

用法::
    python tools/build_zh_pinyin_all.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402


def main():
    from openshenzhou import g2p_zh as gz
    from pypinyin import pinyin, Style

    p = os.path.join(HERE, "data", "pinyin_zh.json")
    d = json.load(open(p, encoding="utf-8"))
    before = len(d)
    ranges = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]
    added = 0
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            if ch in d:
                continue
            try:
                py = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
                if py and py[0]:
                    val = py[0][0]
                    if val:
                        d[ch] = val
                        added += 1
            except Exception:
                continue
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print("pinyin_zh: %d -> %d (+%d)" % (before, len(d), added))


if __name__ == "__main__":
    main()
