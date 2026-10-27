#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_merge_evidence.py —— 生成「官方词表中有独立词证据」的子集

合并率表缺失时，引擎只在候选确实是词时合并（如 为时已晚、新生），不会合并非词短语
（如 来做、的需要）。证据来源（任一成立）：
  1) jieba 词典存在该词（可选依赖，未安装则跳过）
  2) 引擎 .dat 词表（tts-engine/data/engine_lexicon.json，逆向产物，可选）
  3) 引擎语料实际输出过的词（data/zh_engine_token_freq.json）

输出 data/zh_merge_evidence.json（排序后的词列表）。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def main():
    official = set(load_json(os.path.join(HERE, "data", "official_words_zh.json"), []))
    efreq = load_json(os.path.join(HERE, "data", "zh_engine_token_freq.json"), {})
    eng_lex = None
    p = os.path.join(HERE, "..", "tts-engine", "data", "engine_lexicon.json")
    if len(sys.argv) > 1:
        p = sys.argv[1]
    if os.path.exists(p):
        eng_lex = set(load_json(p, []))
    jf = {}
    try:
        import jieba
        jieba.initialize()
        jf = jieba.dt.FREQ
    except Exception:
        jf = {}
    ev = set()
    for w in official:
        if (jf and jf.get(w, 0) > 0) or (eng_lex is not None and w in eng_lex) \
                or efreq.get(w, 0) > 0:
            ev.add(w)
    out = os.path.join(HERE, "data", "zh_merge_evidence.json")
    json.dump(sorted(ev), open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("official", len(official), "evidence", len(ev), "->", out)


if __name__ == "__main__":
    main()
