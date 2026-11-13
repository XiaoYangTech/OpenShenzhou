#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_zh_mixed.py —— 中文混合拉丁读法评测（引擎探针集）+ Python≡Node

语料 data/eval/zh_mixed_ids.jsonl：引擎实测（Xiaoxiao）对以下文本的读音 ids：
  - 中文文本中的单个拉丁字母（A、B、C 题目标注、第A项、A和B、A. 第一项…）
  - 字母串（AB、ABC、CPU、USB、PDF、iPhone、MP3…）
  - 英文词（Hello、Windows、Office、Excel…）
  - 标点/括号/引号/点号（：、.（句内/句尾）、（）【】、…、；、“”…）

用法::
    OS_VOICE_ROOT=... [OS_SDK_ROOT=...] [OS_NODE_ROOT=...] python tools/eval_zh_mixed.py [--show N]
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

CORPUS = os.path.join(HERE, "data", "eval", "zh_mixed_ids.jsonl")
THRESHOLD = 0.90


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

    show = 0
    if "--show" in sys.argv:
        try:
            show = int(sys.argv[sys.argv.index("--show") + 1])
        except Exception:
            show = 10
    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = gz.load_symbols(os.path.join(pkg, "phones.txt"))
    table = gz.load_table()
    lex = gz.load_lexicon()

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    ok = 0
    bad = []
    for r in rows:
        ids = gz.text_to_ids(r["text"], table, sym, lex)
        if ids == r["ids"]:
            ok += 1
        else:
            bad.append(r)
    print("中文混合读法逐 id 一致: %d/%d = %.2f%%" % (ok, len(rows), 100.0 * ok / max(1, len(rows))))
    if show:
        for r in bad[:show]:
            rev = {v: k for k, v in sym.items()}
            pn = [rev.get(x, "?%d" % x) for x in gz.text_to_ids(r["text"], table, sym, lex)]
            gn = [rev.get(x, "?%d" % x) for x in r["ids"]]
            print("\nTEXT %r" % r["text"])
            print("  gold", " ".join(gn))
            print("  pred", " ".join(pn))

    root = node_root()
    if root is None:
        print("未找到 node 目录（可设 OS_NODE_ROOT），跳过双端一致性")
        return 0 if ok / max(1, len(rows)) >= THRESHOLD else 1
    tmp_in = os.path.join(os.environ.get("TEMP", HERE), "zh_mixed_parity.jsonl")
    with open(tmp_in, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    out_json = os.path.join(os.environ.get("TEMP", HERE), "zh_mixed_node_ids.json")
    p = subprocess.run(["node", os.path.join(HERE, "tools", "node_zh_ids_batch.js"),
                        root, pkg, tmp_in, out_json],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print("node 运行失败:", (p.stdout or "") + (p.stderr or ""))
        return 1
    node_ids = [r["ids"] for r in json.load(open(out_json, encoding="utf-8"))]
    same = 0
    for r, nd in zip(rows, node_ids):
        py = gz.text_to_ids(r["text"], table, sym, lex)
        if py == nd:
            same += 1
    print("混合读法 Python≡Node: %d/%d" % (same, len(rows)))
    ok &= same == len(rows) and ok / max(1, len(rows)) >= THRESHOLD
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
