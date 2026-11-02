#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_zh_cw_algorithm.py —— 验证「引擎后处理分词算法」在引擎真值 CWordBreaker 词序列上的还原度。

数据（均来自引擎实测，已随仓库提供）：
  data/zh_compound_dict.txt.gz   引擎 POSTWORDBREAKER 复合词表（108,577 条，
                                 条目形如 `弹,出,窗口`，逗号 = 内部成分切分）
  data/eval/zh_cw_599.json       引擎 CWordBreaker 真值输出（597 句，Frida hook 抓取）
  data/eval/zh_eval_599.jsonl    599 语料 + 引擎最终 id golden

算法（反编译 FUN_180240c18 / FUN_180240ef8 / FUN_180119a28 得到）：
  从当前词起，取至多 5 个 CWordBreaker 词，用逗号连接成候选串，在复合词表中查找；
  取最长且恰好落在词边界（即整词条目）的匹配，合并为一个词；否则保留原词。

用法::
    python tools/eval_zh_cw_algorithm.py [--show N]
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
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402

PUNCT = set('。！？.!?，、；：,;:')
MAX_WORDS = 5


def load_dict():
    path = os.path.join(HERE, "data", "zh_compound_dict.txt.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}


def merge_words(tokens, entry_set, max_words=MAX_WORDS):
    out = []
    i = 0
    while i < len(tokens):
        best = i
        for k in range(i + 1, min(len(tokens), i + max_words)):
            if ",".join(tokens[i:k + 1]) in entry_set:
                best = k
        if best > i:
            out.append("".join(tokens[i:best + 1]))
            i = best + 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def gold_words(sent, ids, rev):
    cn = [c for c in sent if "\u4e00" <= c <= "\u9fff" or c == "〇"]
    out = []
    content = 0
    pending = False
    st = 0
    for x in ids:
        nm = rev.get(x)
        if nm in ("-bos-", "br0"):
            continue
        if nm == "~":
            break
        if nm and nm.startswith("punc"):
            pending = True
            continue
        if nm in ("br1", "-"):
            pending = nm == "br1"
            continue
        if nm in ("br2", "br3", "br4"):
            pending = True
            continue
        if pending and content > 0:
            ci = content // 3
            out.append("".join(cn[st:ci]))
            st = ci
        content += 1
        pending = False
    out.append("".join(cn[st:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    from openshenzhou import g2p_zh as gz
    sym = gz.load_symbols(os.path.join(pkg, "phones.txt"))
    rev = {v: k for k, v in sym.items()}

    entries = load_dict()
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
        toks = [w for w in cw_map[sent]
                if w and not all(ch in PUNCT for ch in w)]
        gold = gold_words(sent, r["ids"], rev)
        if not toks or "".join(toks) != "".join(gold):
            continue
        total += 1
        pred = merge_words(toks, entries)
        if pred == gold:
            ok += 1
        elif shown < args.show:
            shown += 1
            print("差异:", sent)
            print("  CW  :", toks)
            print("  预测:", pred)
            print("  gold:", gold)
    print("引擎后处理算法还原: %d/%d = %.1f%%" % (ok, total, 100.0 * ok / total if total else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
