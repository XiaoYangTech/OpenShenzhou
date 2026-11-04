#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_zh_trad.py —— 繁体文本逐字读音评测（字符级）

数据：data/zh_trad_novel_ids_3000.json.gz（引擎对繁体小说的 ids 抓取，3000 句）
指标：字符级音节 token 完全一致率（忽略词/句边界差异），以及整句一致率。
"""
import argparse
import gzip
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

from openshenzhou import g2p_zh  # noqa: E402

BOUND = {"-bos-", "br0", "br1", "br2", "br3", "br4", "~", "-"}


def syllables(ids, rev):
    names = []
    for x in ids:
        names.append(rev.get(x, "?%d" % x))
        if x == 1:
            break
    out = []
    cur = []
    for t in names:
        if t in BOUND or t.startswith("punc"):
            if cur:
                out.append(tuple(cur))
                cur = []
        else:
            cur.append(t)
    if cur:
        out.append(tuple(cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Xiaoxiao_Female_zh-CN")
    ap.add_argument("--corpus", default=os.path.join(HERE, "data", "zh_trad_novel_ids_3000.json.gz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-trad", action="store_true", help="关闭繁体修正（基线对照）")
    args = ap.parse_args()

    pkg = voice_pkg(args.voice)
    table = g2p_zh.load_table()
    sym = g2p_zh.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_zh.load_lexicon()
    rev = {v: k for k, v in sym.items()}
    with gzip.open(args.corpus, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = dict(list(data.items())[:args.limit])

    if args.no_trad:
        import openshenzhou.zh_trad as zh_trad
        zh_trad.set_enabled(False)

    tot = ex = sent_ok = sent_all = 0
    for sent, gold_ids in data.items():
        pred = g2p_zh.text_to_ids(sent, table, sym, lex)
        gs = syllables(gold_ids, rev)
        ps = syllables(pred, rev)
        sent_all += 1
        if gs == ps:
            sent_ok += 1
        for k in range(min(len(gs), len(ps))):
            tot += 1
            if gs[k] == ps[k]:
                ex += 1
    print("繁体逐字音节一致: %d/%d = %.2f%%；整句一致: %d/%d = %.2f%%" % (
        ex, tot, 100.0 * ex / max(tot, 1), sent_ok, sent_all,
        100.0 * sent_ok / max(sent_all, 1)))


if __name__ == "__main__":
    main()
