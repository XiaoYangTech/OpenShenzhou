#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_zh_parity_batch.py —— 中文 Python/Node 双端逐 id 一致性（批量）

语料：data/eval/zh_eval_599.jsonl（599 句）+ zh_large_eval.jsonl 抽样（默认 300 句）
Node 侧用 tools/node_zh_ids_batch.js 一次性导出；Python 侧直接调用 SDK 前端。

用法::
    OS_VOICE_ROOT=... [OS_SDK_ROOT=...] [OS_NODE_ROOT=...] python tools/check_zh_parity_batch.py
"""
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402


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
    from openshenzhou import g2p_zh as gz

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = gz.load_symbols(os.path.join(pkg, "phones.txt"))
    table = gz.load_table()
    lex = gz.load_lexicon()
    root = node_root()
    if root is None:
        print("未找到 node 目录（可设 OS_NODE_ROOT）")
        return 1

    corpora = [("599", os.path.join(HERE, "data", "eval", "zh_eval_599.jsonl"), 599),
               ("large", os.path.join(HERE, "data", "eval", "zh_large_eval.jsonl"), 300)]
    total = same = 0
    for name, path, limit in corpora:
        rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
        tmp = os.path.join(os.environ.get("TEMP", HERE), "zh_%s_parity.jsonl" % name)
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
        out_json = os.path.join(os.environ.get("TEMP", HERE), "zh_%s_node_ids.json" % name)
        p = subprocess.run(["node", os.path.join(HERE, "tools", "node_zh_ids_batch.js"),
                            root, pkg, tmp, out_json],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print("node 运行失败:", (p.stdout or "") + (p.stderr or ""))
            return 1
        node_ids = [r["ids"] for r in json.load(open(out_json, encoding="utf-8"))]
        ok = 0
        for r, nd in zip(rows, node_ids):
            py = gz.text_to_ids(r["text"], table, sym, lex)
            if py is None:
                continue
            if py == nd:
                ok += 1
        print("中文 %-5s Python≡Node: %d/%d" % (name, ok, len(rows)))
        same += ok
        total += len(rows)
    print("合计: %d/%d = %.2f%%" % (same, total, 100.0 * same / max(1, total)))
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
