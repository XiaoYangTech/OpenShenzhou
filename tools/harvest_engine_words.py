#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""harvest_engine_words.py —— 用引擎探针采集任意文本的「词 → 读音」

用于补标定词表：给出若干文本，输出引擎实际分词与逐词音素（符号名）。
用法::  python tools/harvest_engine_words.py "本卷" "一起"
"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(os.environ.get("TEMP", ""), "probe_harvest")
PORT = int(os.environ.get("OS_ENGINE_PORT", "17497"))
VOICE = "Microsoft Xiaoxiao"


def load_id2name():
    pkg = os.path.join(os.environ.get("TEMP", ""), "AllVoice", "Xiaoxiao_Female_zh-CN")
    m = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or ":" not in line or line.startswith("#"):
            continue
        n, i = line.rsplit(":", 1)
        m[int(i)] = n
    return m


def speak(text):
    req = urllib.request.Request("http://127.0.0.1:%d/speak" % PORT,
                                 data=json.dumps({"text": text, "voice": VOICE}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:
        print("speak err", e)
        return False


def wait_new(before, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        new = [f for f in glob.glob(os.path.join(PROBE, "in_n2_i0_*.f32")) if f not in before]
        if new:
            new.sort(key=lambda f: int(f.rsplit("_c", 1)[1].split(".")[0]))
            return new[-1]
        time.sleep(0.05)
    return None


def capture(text):
    before = set(glob.glob(os.path.join(PROBE, "in_n2_i0_*.f32")))
    if not speak(text):
        return None
    fn = wait_new(before)
    if not fn:
        return None
    return np.fromfile(fn, dtype=np.int32).tolist()


def words_of(ids, id2name):
    words = []
    cur = []
    nsym = 0
    for x in ids:
        nm = id2name.get(x, "?")
        if nm in ("br0", "br1", "br2", "br3", "br4", "~", "-bos-", "padding"):
            if cur:
                words.append(cur)
                cur = []
            nsym = 0
            continue
        if nm == "-":
            nsym = 0
            continue
        if nm.startswith("punc"):
            if cur:
                words.append(cur)
                cur = []
            nsym = 0
            continue
        nsym += 1
        if nsym == 1:
            cur.append(nm)
        elif nsym == 2:
            cur.append("." + nm)
        elif nsym >= 3:
            cur[-1] += "." + nm
            nsym = 0
            if len(cur) > 1:
                cur[-2] = cur[-2]
    # 每 3 个符号一个音节，合并为词
    out = []
    for w in words:
        out.append(w)
    return out


def main():
    texts = sys.argv[1:] or ["本卷", "一起", "奇努克", "负载", "空负载"]
    id2name = load_id2name()
    os.makedirs(PROBE, exist_ok=True)
    for text in texts:
        ids = capture(text)
        if ids is None:
            print("MISS", text)
            continue
        # 逐 3 符号组音节，词边界由 br 标记
        words = []
        cur = []
        trip = []
        for x in ids:
            nm = id2name.get(x, "?")
            if nm in ("br0", "br1", "br2", "br3", "br4", "~", "-bos-", "padding"):
                if trip:
                    cur.append("".join(trip))
                    trip = []
                if cur:
                    words.append(" ".join(cur))
                    cur = []
                continue
            if nm == "-":
                if trip:
                    cur.append("".join(trip))
                    trip = []
                continue
            if nm.startswith("punc"):
                if trip:
                    cur.append("".join(trip))
                    trip = []
                if cur:
                    words.append(" ".join(cur))
                    cur = []
                continue
            trip.append(nm)
            if len(trip) == 3:
                cur.append("".join(trip))
                trip = []
        if cur:
            words.append(" ".join(cur))
        print(text, "->", words)


if __name__ == "__main__":
    main()
