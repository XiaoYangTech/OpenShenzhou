#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""harvest_en_lexicon_engine.py —— 用引擎探针标定英文「词 → 音素名序列」

对每个词单独合成（fast 探针模式），从 encoder 输入 ids 中抽取词内音素名。
输出 jsonl：{"w": "rabbit", "phones": ["r","ae1","-","b","ih","t"]}

用法::

    python tools/harvest_en_lexicon_engine.py --port 17515 \\
        --probe %TEMP%/probe_enf --words data/eval/en_words.txt \\
        --out data/lexicon_en_engine.jsonl
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
sys.path.insert(0, os.path.join(HERE, "tools"))
from _paths import voice_pkg  # noqa: E402


def speak(port, text, voice):
    req = urllib.request.Request("http://127.0.0.1:%d/speak" % port,
                                 data=json.dumps({"text": text, "voice": voice},
                                                 ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=180).read()
        return True
    except Exception:
        return False


def cidx(fn):
    m = re.search(r"_c(\d+)\.f32$", fn)
    return int(m.group(1)) if m else -1


def newest_ids(probe, before, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        files = [f for f in glob.glob(os.path.join(probe, "in_n2_i0_*.f32"))
                 if os.path.basename(f) not in before]
        if files:
            files.sort(key=lambda f: cidx(os.path.basename(f)))
            time.sleep(0.03)
            return np.fromfile(files[-1], dtype=np.int32).tolist()
        time.sleep(0.02)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=17515)
    ap.add_argument("--probe", default=os.path.join(os.environ.get("TEMP", ""), "probe_enf"))
    ap.add_argument("--voice", default="Microsoft Aria")
    ap.add_argument("--pkg", default="Aria_Female_en-US")
    ap.add_argument("--words", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pkg = voice_pkg(args.pkg)
    sym = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        name, idx = line.rsplit(":", 1)
        sym[name] = int(idx)
    id2name = {v: k for k, v in sym.items()}
    br0, br4, bos, end = sym.get("br0"), sym.get("br4"), sym.get("-bos-"), sym.get("~")

    words = [w.strip() for w in open(args.words, encoding="utf-8") if w.strip()]
    if args.limit:
        words = words[:args.limit]
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["w"]] = r["phones"]
    out = open(args.out, "a", encoding="utf-8")
    ok = 0
    for i, w in enumerate(words):
        if w in done:
            ok += 1
            continue
        before = set(os.listdir(args.probe))
        speak(args.port, w, args.voice)
        ids = newest_ids(args.probe, before)
        if not ids:
            continue
        # 解析：-bos- br0 <phones> br4 [punc.] ~
        try:
            a = ids.index(br0) + 1
        except ValueError:
            continue
        b = ids.index(br4) if br4 in ids else len(ids)
        seq = ids[a:b]
        names = [id2name.get(x, "?%d" % x) for x in seq]
        if any(n.startswith("?") or n in ("br1", "br3") for n in names):
            continue
        out.write(json.dumps({"w": w, "phones": names}, ensure_ascii=False) + "\n")
        out.flush()
        done[w] = names
        ok += 1
        if ok % 100 == 0:
            print("harvested", ok, "/", len(words), w)
    out.close()
    print("done", ok, "->", args.out)


if __name__ == "__main__":
    main()
