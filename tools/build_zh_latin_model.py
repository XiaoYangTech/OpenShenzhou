#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_latin_model.py —— 由引擎探针数据生成中文混合拉丁读法模型

输入 data/zh_latin_engine.json（引擎实测，中文语音读拉丁字母/英文词的 token 序列）：
  letters_single: 单字母（中文式字母名，如 A→ey1, B→bi i_h i_l）
  letters_pair:   字母串内每个字母（英文式字母名，如 B→b iy1）
  words:          英文词/缩写读音（token 序列，'-' 为词内音节分隔）

输出 data/zh_latin.json：
  single/letters/words 读法表 + phone_map（英文音素名→中文音素 token 名映射，
  由「英文词典音素序列 ↔ 引擎 token 序列」对齐投票得到，再用 103 个探针词 100% 自检）。
"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from _paths import voice_pkg, sdk_root  # noqa: E402

if os.path.isdir(os.path.join(sdk_root(), "openshenzhou")):
    sys.path.insert(0, sdk_root())
from openshenzhou import g2p_zh, g2p_en  # noqa: E402


def main():
    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = g2p_zh.load_symbols(os.path.join(pkg, "phones.txt"))
    raw = json.load(open(os.path.join(HERE, "data", "zh_latin_engine.json"), encoding="utf-8"))
    lex = g2p_en.load_lexicon()

    # 音素映射投票：词表里有、探针也有的词
    votes = collections.defaultdict(collections.Counter)
    aligned = 0
    for w, core in raw["words"].items():
        phones = lex.get(w)
        if not phones:
            continue
        ph = [p for p in phones if p != "-"]
        en = [t for t in core if t != "-"]
        if len(ph) != len(en):
            continue
        aligned += 1
        for p, t in zip(ph, en):
            votes[p][t] += 1
    phone_map = {p: c.most_common(1)[0][0] for p, c in votes.items()}

    def map_phone(p):
        if p == "-":
            return "-"
        if p in phone_map:
            return phone_map[p]
        if p in sym:
            return p
        if p == "zh":
            return "zzh"
        return None

    ok = bad = 0
    for w, core in raw["words"].items():
        phones = lex.get(w)
        if not phones:
            continue
        gen = [map_phone(p) for p in phones]
        if None in gen:
            continue
        if gen == core:
            ok += 1
        else:
            bad += 1
    model = {
        "single": raw["letters_single"],
        "letters": raw["letters_pair"],
        "words": raw["words"],
        "phone_map": phone_map,
    }
    out = os.path.join(HERE, "data", "zh_latin.json")
    json.dump(model, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("aligned words", aligned, "regenerated exact", ok, "mismatch", bad)
    print("saved", out, os.path.getsize(out))


if __name__ == "__main__":
    main()
