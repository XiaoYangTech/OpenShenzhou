#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_engine_words.py —— 从引擎标定数据重建「引擎分词词表」及词频

引擎输出：-bos- br0 <词1 音节组> br1 <词2> ... br4
音节组内 '-' 分隔音节，每个音节对应一个汉字（中文场景）。
用 text 的汉字序列与 ids 的词/音节结构对齐，即可还原每个"韵律词"的字符串，
统计词频 → data/zh_engine_words.json（word -> count）。

这份词表反映的是**引擎自己的切词习惯**（含把"忘记"拆成"忘|记"这类），
和普通词典不同，作为断句模型特征很有价值。
"""
import json
import os
import sys
from collections import Counter
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
PUNCT = set("，。！？、；：,.;!?…—－《》「」『』（）()【】 \t")
BOS, BR4, BR0, BR1, BR3, BR2, SEP, END = 2, 17, 68, 25, 39, 18, 3, 1
TAGS = {BR1, BR4, BR0, BR3, BR2}
FILES = ["seg_A.jsonl", "seg_B.jsonl", "seg_A2.jsonl", "seg_C.jsonl", "seg_C2.jsonl",
         "seg_C3.jsonl", "seg_D.jsonl", "seg_D2.jsonl", "seg_D3.jsonl"]


def word_spans(ids):
    """返回每个词的 (音节数, 词内是否有分隔符) 列表；不是音节/标签的符号忽略。"""
    i = 0
    n = len(ids)
    while i < n and ids[i] != BOS:
        i += 1
    i += 1
    spans = []
    cur = 0          # 音节数
    seps = 0         # 当前词内 '-' 个数
    started = False
    while i < n:
        x = ids[i]
        if x == END:
            if started:
                spans.append(cur + seps)
            break
        if x in TAGS or x == BOS:
            if started:
                spans.append(cur + seps)
                cur = 0
                seps = 0
                started = False
        elif x == SEP:
            if started:
                seps += 1
        else:
            if not started:
                started = True
                cur = 1
        i += 1
    if started:
        spans.append(cur + seps)
    return [s for s in spans if s > 0]


def main():
    cnt = Counter()
    total = 0
    for f in FILES:
        p = os.path.join(HERE, "data", f)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            chars = [c for c in r["text"] if c not in PUNCT]
            sizes = word_spans(r["ids"])
            if sum(sizes) != len(chars) or len(chars) < 2:
                continue
            pos = 0
            for s in sizes:
                w = "".join(chars[pos:pos + s])
                pos += s
                if w:
                    cnt[w] += 1
                    total += 1
    print("引擎词表: %d 词, %d tokens" % (len(cnt), total))
    out = os.path.join(ROOT, "data", "zh_engine_words.json")
    # 限制词表大小（保留高频词），控制体积
    keep = {w: c for w, c in cnt.items() if c >= 2}
    print("保留词数( freq>=2 ): %d" % len(keep))
    json.dump(keep, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("写入", out)


if __name__ == "__main__":
    main()
