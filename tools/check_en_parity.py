#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_en_parity.py —— 英文 Python/Node 双端逐 id 一致性（682 句引擎语料）

语料：data/eval/en_engine_500.jsonl（自然句）、en_engine_tn.jsonl（TN 压力集）、
en_engine_heteronyms.jsonl（异读词句）。Python 侧用 SDK 的 g2p_en，Node 侧调用
node 的 TTS.textToIds（需先在 node/ 下 npm install）。

用法::
    OS_SDK_ROOT=/path/to/python-sdk [OS_NODE_ROOT=/path/to/work-node] python tools/check_en_parity.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402

CORPORA = [
    ("自然句", "en_engine_500.jsonl"),
    ("TN", "en_engine_tn.jsonl"),
    ("异读词", "en_engine_heteronyms.jsonl"),
    ("holdout", "en_engine_holdout500.jsonl"),
    ("holdout2", "en_engine_holdout2_500.jsonl"),
]


def node_root():
    cand = os.environ.get("OS_NODE_ROOT")
    if cand and os.path.isdir(os.path.join(cand, "node")):
        return cand
    if os.path.isdir(os.path.join(HERE, "node")):
        return HERE
    alt = os.path.join(os.path.dirname(HERE), "work-node")
    if os.path.isdir(os.path.join(alt, "node")):
        return alt
    return None


def main():
    from openshenzhou import g2p_en

    pkg = voice_pkg("Aria_Female_en-US")
    sym = g2p_en.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_en.load_lexicon()
    rev = {v: k for k, v in sym.items()}
    root = node_root()
    if root is None:
        print("未找到 node 目录（可设 OS_NODE_ROOT）")
        return 1
    dump_js = os.path.join(HERE, "tools", "node_dump_ids.js")
    total = same = 0
    for name, fname in CORPORA:
        path = os.path.join(HERE, "data", "eval", fname)
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        out_json = os.path.join(os.environ.get("TEMP", HERE), "en_parity_dump.json")
        p = subprocess.run(["node", dump_js, root,
                            pkg, path, out_json],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print("node 运行失败:", (p.stdout or "") + (p.stderr or ""))
            return 1
        node_ids = {r["text"]: r["node"] for r in json.load(open(out_json, encoding="utf-8"))}

        def slice_first(ids):
            br4 = sym.get("br4")
            if br4 is not None and br4 in ids:
                k = ids.index(br4) + 1
                while k < len(ids) and (str(rev.get(ids[k], "")).startswith("punc")
                                        or rev.get(ids[k]) == "br0"):
                    k += 1
                return ids[:k] + [sym.get("~")]
            return ids

        ok = 0
        for r in rows:
            py = slice_first(g2p_en.text_to_ids(r["text"], sym, lex))
            nd = slice_first(node_ids.get(r["text"], []))
            if py == nd:
                ok += 1
        print("英文 %-6s Python≡Node: %d/%d" % (name, ok, len(rows)))
        same += ok
        total += len(rows)
    print("合计: %d/%d = %.2f%%" % (same, total, 100.0 * same / max(1, total)))
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
