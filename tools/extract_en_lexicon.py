#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_en_lexicon.py —— 从 MSTTSLocEnUS.dat 提取引擎英文「词 → 音素串」词典

格式（实测）：u16 word_len + word(utf16) + u16 0 + u16 phon_len + phon(utf16) + u16 0
phon 形如 "s eh 1 t ih ng z"（音素 + 可选重音数字分开写）。
"""
import json
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAT = os.path.join(os.environ.get("OS_VOICE_ROOT", ""), "Aria_Female_en-US",
                   "MSTTSLocEnUS.dat")


def read_str(d, off, maxlen=64, ascii_only=False):
    if off + 4 > len(d):
        return None, None
    L = struct.unpack_from("<H", d, off)[0]
    if L < 1 or L > maxlen:
        return None, None
    p = off + 2
    chars = []
    for _ in range(L):
        if p + 2 > len(d):
            return None, None
        cp = struct.unpack_from("<H", d, p)[0]
        if ascii_only and not (32 <= cp < 127):
            return None, None
        chars.append(chr(cp))
        p += 2
    if p + 2 > len(d) or struct.unpack_from("<H", d, p)[0] != 0:
        return None, None
    return "".join(chars), p + 2


def main():
    d = open(DAT, "rb").read()
    n = len(d)
    words = {}
    # 扫描：u16 词长 + 纯小写字母词 + 0 + 音素串
    i = 0
    step = 1
    while i < n - 8:
        w, p2 = read_str(d, i, 40, ascii_only=True)
        if w and w.isalpha() and w.islower() and len(w) >= 2:
            ph, p3 = read_str(d, p2, 200, ascii_only=True)
            if ph and (" " in ph or ph.isalpha()):
                toks = ph.split()
                ok = toks and all(t.isdigit() or t.replace("-", "").isalpha() for t in toks)
                if ok:
                    if w not in words:
                        words[w] = ph
                    i = p3
                    continue
        i += step
    print("entries", len(words))
    out = os.path.join(HERE, "data", "en_engine_lexicon_dat.json")
    json.dump(words, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", out)
    for w in ("hello", "world", "computer", "settings", "and", "the", "one", "hundred"):
        print(w, words.get(w))


if __name__ == "__main__":
    main()
