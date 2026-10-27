#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_zh_words_batch.py —— 批量探测引擎对任意词语的切分（词表成员判定）

把多个候选词用句号拼成多句文本一次性发给引擎（正常模式，非 --ort-fast）：
引擎逐句编码，ORT 探针为每句写一个 in_n2_i0_*.f32；读取新增文件按计数排序，
逐个解码 token 数即得到「该词被引擎切成几个词」。

输出 jsonl：{"w": 词, "n": token 数, "parts": [各 token 的汉字串]}

用法::
    python tools/probe_zh_words_batch.py --port 17524 \\
        --probe %TEMP%/probe_zb0 --words data/eval/zh_probe_candidates.txt \\
        --out data/eval/zh_probe_words_0.jsonl --batch 20
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CJK = lambda c: "\u4e00" <= c <= "\u9fff"


def load_symbols(pkg):
    names = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        names[int(i)] = n
    return names


def probe(port, text, voice):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/speak" % port,
        data=json.dumps({"text": text, "voice": voice}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
        return True
    except Exception:
        return False


def cidx(fn):
    m = re.search(r"_c(\d+)\.f32$", fn)
    return int(m.group(1)) if m else -1


def decode_parts(ids, names):
    """→ list of token 的汉字数；只处理纯汉字 token"""
    rev = names
    parts = []
    syl = 0
    started = False
    for x in ids:
        nm = rev.get(int(x), "?")
        if nm in ("-bos-", "br0", "br2", "br3", "br4", "~") or nm.startswith("punc"):
            continue
        if nm == "br1":
            if started:
                parts.append(syl + 1)
            syl = 0
            started = False
            continue
        if nm == "-":
            syl += 1
            started = True
        else:
            started = True
    if started:
        parts.append(syl + 1)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--voice", default="Microsoft Xiaoxiao")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()

    from _paths import voice_pkg

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    names = load_symbols(pkg)
    words = [w.strip() for w in open(args.words, encoding="utf-8") if w.strip()]
    words = [w for w in words if 2 <= len(w) <= 6 and all(CJK(c) for c in w)]
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["w"]] = r
    todo = [w for w in words if w not in done]
    print("total", len(words), "done", len(done), "todo", len(todo))
    os.makedirs(args.probe, exist_ok=True)
    out = open(args.out, "a", encoding="utf-8")

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    if args.max_batches:
        batches = batches[:args.max_batches]
    ok_total = 0
    for bi, batch in enumerate(batches):
        text = "。".join(batch) + "。"
        before = set(os.listdir(args.probe))
        before_ids = set(cidx(f) for f in before if f.startswith("in_n2_i0_"))
        probe(args.port, text, args.voice)
        # 等新文件出现
        new_files = []
        t0 = time.time()
        while time.time() - t0 < 120:
            fs = [f for f in os.listdir(args.probe)
                  if f.startswith("in_n2_i0_") and cidx(f) not in before_ids]
            if len(fs) >= len(batch):
                new_files = sorted(fs, key=lambda f: cidx(f))[:len(batch)]
                break
            time.sleep(0.05)
        if len(new_files) < len(batch):
            print("batch fail", bi, "got", len(new_files), "want", len(batch))
            # 逐词兜底
            for w in batch:
                b2 = set(cidx(f) for f in os.listdir(args.probe) if f.startswith("in_n2_i0_"))
                txt = w + "。"
                probe(args.port, txt, args.voice)
                t0 = time.time()
                got = None
                while time.time() - t0 < 60:
                    fs = [f for f in os.listdir(args.probe)
                          if f.startswith("in_n2_i0_") and cidx(f) not in b2]
                    if fs:
                        got = fs[0]
                        break
                    time.sleep(0.05)
                if got is None:
                    continue
                try:
                    ids = np.fromfile(os.path.join(args.probe, got), dtype=np.int32).tolist()
                except Exception:
                    continue
                parts = decode_parts(ids, names)
                r = {"w": w, "n": len(parts), "parts": parts}
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                done[w] = r
                ok_total += 1
            continue
        for w, f in zip(batch, new_files):
            try:
                ids = np.fromfile(os.path.join(args.probe, f), dtype=np.int32).tolist()
            except Exception:
                continue
            parts = decode_parts(ids, names)
            r = {"w": w, "n": len(parts), "parts": parts}
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[w] = r
            ok_total += 1
        out.flush()
        if ok_total % 200 == 0:
            print("probed", ok_total, "/", len(todo))
    out.close()
    print("done", ok_total, "->", args.out)


if __name__ == "__main__":
    main()
