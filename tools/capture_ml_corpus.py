#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""capture_ml_corpus.py —— 用 ORT 探针批量抓取引擎 encoder 输入 ids（任意语种）

前提：TtsEngineServer 已用 --ort-probe=<dir> 启动（建议再加 --ort-fast 提速），
且该语种的自然语音已装好（例如 en: Microsoft Aria）。

用法::

    python tools/capture_ml_corpus.py --port 17512 --probe %TEMP%/probe_en \\
        --voice "Microsoft Aria" --in data/eval/en_sentences.txt \\
        --out data/eval/en_engine_500.jsonl --limit 500

输出 jsonl：{"text": ..., "ids": [...]}
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


def speak(port, text, voice):
    req = urllib.request.Request("http://127.0.0.1:%d/speak" % port,
                                 data=json.dumps({"text": text, "voice": voice},
                                                 ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=180).read()
    except Exception as e:
        print("speak err", e)
        return False
    return True


def cidx(fn):
    m = re.search(r"_c(\d+)\.f32$", fn)
    return int(m.group(1)) if m else -1


def newest_ids(probe, before, timeout=20.0):
    """等新的 in_n2_i0 文件出现并取 ids（按探针计数取最新）"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        files = [f for f in glob.glob(os.path.join(probe, "in_n2_i0_*.f32"))
                 if os.path.basename(f) not in before]
        if files:
            files.sort(key=lambda f: cidx(os.path.basename(f)))
            time.sleep(0.05)
            return np.fromfile(files[-1], dtype=np.int32).tolist()
        time.sleep(0.03)
    return None


def load_sentences(path):
    if path.endswith(".jsonl"):
        out = []
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out.append(r.get("text") or r.get("sentence") or "")
        return [x for x in out if x]
    text = open(path, encoding="utf-8", errors="replace").read()
    return [x for x in (s.strip() for s in text.splitlines()) if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=17512)
    ap.add_argument("--probe", default=os.path.join(os.environ.get("TEMP", ""), "probe_en"))
    ap.add_argument("--voice", default="Microsoft Aria")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--ignore-http-error", action="store_true",
                    help="--ort-fast 模式下引擎会在 encoder 后中止，/speak 返回 500 但探针已写入")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    done = {}
    if args.resume and os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["text"]] = r["ids"]
    sents = load_sentences(args.inp)
    if args.limit:
        sents = sents[:args.limit]
    out = open(args.out, "a", encoding="utf-8")
    ok = 0
    for i, text in enumerate(sents):
        if text in done:
            ok += 1
            continue
        before = set(os.listdir(args.probe))
        ok_speak = speak(args.port, text, args.voice)
        if not ok_speak and not args.ignore_http_error:
            continue
        ids = newest_ids(args.probe, before)
        if not ids:
            print("[MISS]", i, text[:60])
            continue
        out.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")
        out.flush()
        done[text] = ids
        ok += 1
        if ok % 25 == 0:
            print("captured", ok, "/", len(sents), text[:50])
    out.close()
    print("done captured", ok, "->", args.out)


if __name__ == "__main__":
    main()
