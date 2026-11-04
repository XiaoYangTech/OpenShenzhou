#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_zh_trad_parity.py —— 繁体文本 Python≡Node 逐 id 一致性检查

从 data/zh_trad_novel_ids_3000.json.gz 取前 N 句，分别用 Python SDK 与
Node SDK 生成 ids 并比较。
"""
import argparse
import gzip
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from _paths import voice_pkg, sdk_root  # noqa: E402

if os.path.isdir(os.path.join(sdk_root(), "openshenzhou")):
    sys.path.insert(0, sdk_root())

from openshenzhou import g2p_zh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Xiaoxiao_Female_zh-CN")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    pkg = voice_pkg(args.voice)
    with gzip.open(os.path.join(HERE, "data", "zh_trad_novel_ids_3000.json.gz"),
                   "rt", encoding="utf-8") as f:
        data = json.load(f)
    sents = list(data)[:args.limit]

    table = g2p_zh.load_table()
    sym = g2p_zh.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_zh.load_lexicon()
    py = [g2p_zh.text_to_ids(s, table, sym, lex) for s in sents]

    sent_path = os.path.join(os.environ.get("TEMP", "."), "zh_trad_parity_sents.json")
    with open(sent_path, "w", encoding="utf-8") as f:
        json.dump(sents, f, ensure_ascii=False)
    js = os.path.join(HERE, "tools", "node_zh_trad_ids.js")
    env = dict(os.environ)
    env.setdefault("OS_NODE_ROOT", os.environ.get("OS_NODE_ROOT") or os.path.join(HERE, "..", "work-node"))
    node = subprocess.run(["node", js, pkg, sent_path], capture_output=True, env=env)
    if node.returncode != 0:
        print("Node 运行失败:", node.stderr.decode("utf-8", "replace")[-800:])
        return 1
    nd = json.loads(node.stdout.decode("utf-8"))
    ok = sum(1 for a, b in zip(py, nd) if a == b)
    print("繁体 Python≡Node: %d/%d" % (ok, len(sents)))
    return 0 if ok == len(sents) else 1


if __name__ == "__main__":
    sys.exit(main())
