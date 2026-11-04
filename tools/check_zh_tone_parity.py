#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_zh_tone_parity.py —— 声调修正生效句的 Python≡Node 逐 id 一致性检查

先找出 Python 侧 zh_tone 覆盖生效的句子（开/关对比），再让 Node SDK 输出同句 ids 对比。
"""
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
    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    table = g2p_zh.load_table()
    sym = g2p_zh.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_zh.load_lexicon()
    from openshenzhou import zh_tone
    cache_path = os.path.join(HERE, "data", "zh_tone_parity_sents.json")
    if os.path.exists(cache_path):
        sents = json.load(open(cache_path, encoding="utf-8"))
        print("声调覆盖句（缓存）:", len(sents))
    else:
        sents = []
        seen = set()
        for name in ("zh_eval_599.jsonl", "zh_large_eval.jsonl", "zh_ui_ids_8756.json.gz"):
            path = os.path.join(HERE, "data", "eval", name)
            if not os.path.exists(path):
                path = os.path.join(HERE, "data", name)
            if not os.path.exists(path):
                continue
            if path.endswith(".gz"):
                import gzip
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    texts = list(json.load(f))
            else:
                texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]
            for t in texts:
                if t in seen:
                    continue
                seen.add(t)
                on = g2p_zh.text_to_ids(t, table, sym, lex)
                zh_tone.set_enabled(False)
                off = g2p_zh.text_to_ids(t, table, sym, lex)
                zh_tone.set_enabled(True)
                if on != off:
                    sents.append(t)
        print("声调覆盖生效句数:", len(sents))
        if sents:
            json.dump(sents, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            print("已保存缓存:", cache_path)
    if not sents:
        print("繁体/声调 Python≡Node: 0/0（无覆盖句，跳过）")
        return 0
    sent_path = os.path.join(os.environ.get("TEMP", "."), "zh_tone_parity_sents.json")
    json.dump(sents, open(sent_path, "w", encoding="utf-8"), ensure_ascii=False)
    js = os.path.join(HERE, "tools", "node_zh_trad_ids.js")
    env = dict(os.environ)
    env.setdefault("OS_NODE_ROOT", os.environ.get("OS_NODE_ROOT") or
                   os.path.join(HERE, "..", "work-node"))
    r = subprocess.run(["node", js, pkg, sent_path], capture_output=True, env=env)
    if r.returncode != 0:
        print("Node 运行失败:", r.stderr.decode("utf-8", "replace")[-500:])
        return 1
    nd = json.loads(r.stdout.decode("utf-8"))
    py = [g2p_zh.text_to_ids(s, table, sym, lex) for s in sents]
    ok = sum(1 for a, b in zip(py, nd) if a == b)
    print("声调覆盖句 Python≡Node: %d/%d" % (ok, len(sents)))
    return 0 if ok == len(sents) else 1


if __name__ == "__main__":
    sys.exit(main())
