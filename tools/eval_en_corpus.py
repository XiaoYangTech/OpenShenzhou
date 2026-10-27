#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_en_corpus.py —— 英文前端「逐 id 一致率」评测（引擎探针抓的 ids vs SDK）

用法::

    python tools/eval_en_corpus.py data/eval/en_engine_500.jsonl \\
        [--voice Aria_Female_en-US] [--limit N] [--show 10]
"""
import argparse
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

from openshenzhou import g2p_en  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--voice", default="Aria_Female_en-US")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    pkg = voice_pkg(args.voice)
    sym = g2p_en.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_en.load_lexicon()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    ok = n = 0
    bad = []
    for r in rows:
        text = r["text"]
        eng = r["ids"]
        pred = g2p_en.text_to_ids(text, sym, lex)
        # 引擎探针只抓首句（每句一次 encoder）：截到第一个 br4+punc，并补 END
        br4 = sym.get("br4")
        if br4 is not None and br4 in pred:
            k = pred.index(br4)
            k += 1
            if k < len(pred) and str(sym.get("~")) and pred[k] not in (sym.get("~"),):
                pass
            # 若下一个是 punc 符号则一并包含
            rev = {v: kk for kk, v in sym.items()}
            if k < len(pred) and rev.get(pred[k], "").startswith("punc"):
                k += 1
            pred = pred[:k] + [sym.get("~")]
        else:
            end = sym.get("~")
            if end in pred:
                pred = pred[:pred.index(end) + 1]
        n += 1
        if pred == eng:
            ok += 1
        else:
            bad.append((text, eng, pred))
    print("英文逐 id 一致: %d/%d = %.2f%%" % (ok, n, 100.0 * ok / max(1, n)))
    for text, eng, pred in bad[:args.show]:
        k = 0
        while k < min(len(eng), len(pred)) and eng[k] == pred[k]:
            k += 1
        print("\n[%s]" % text)
        print("  eng ", eng[max(0, k - 3):k + 8])
        print("  ours", pred[max(0, k - 3):k + 8])
        print("  first diff at", k, "len", len(eng), len(pred))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            for text, eng, pred in bad:
                f.write(json.dumps({"text": text, "engine": eng, "ours": pred},
                                   ensure_ascii=False) + "\n")
        print("saved", args.save, len(bad))


if __name__ == "__main__":
    main()
