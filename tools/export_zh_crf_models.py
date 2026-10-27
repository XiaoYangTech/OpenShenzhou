#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_zh_crf_models.py —— 导出中文语音包 .dat 里全部 CRF 模型元数据

输出 data/zh_crf_models.json：
  [{idx, tag, L, T, nfeat, lenA, lenB, labels:[...], templates:[...],
    A_off, B_off, C_off, dat_sha1}]
（A/B 二进制仍从语音包 .dat 按偏移读取，不入库）
"""
import hashlib
import json
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
import parse_crf_models as P  # noqa: E402
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402


def main():
    dat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(_voice_pkg("Xiaoxiao_Female_zh-CN")),
        "MSTTSLocZhCN.dat")
    data = open(dat, "rb").read()
    secs = P.find_sections(data)
    out = []
    for idx, sec in enumerate(secs):
        pool = P.load_string_pool(data, sec)
        lab_off = struct.unpack("<%dI" % sec["L"],
                                data[sec["S"] + 68:sec["S"] + 68 + 4 * sec["L"]])
        tm_off = struct.unpack("<%dI" % sec["T"],
                               data[sec["S"] + 68 + 4 * sec["L"]:
                                    sec["S"] + 68 + 4 * sec["L"] + 4 * sec["T"]])
        out.append({
            "idx": idx,
            "tag": sec["tag"],
            "L": sec["L"],
            "T": sec["T"],
            "nfeat": sec["nfeat"],
            "lenA": sec["lenA"],
            "lenB": sec["lenB"],
            "A_off": sec["A_off"],
            "B_off": sec["A_off"] + sec["lenA"],
            "C_off": sec["A_off"] + sec["lenA"] + sec["lenB"],
            "labels": [P.get_str(pool, o) for o in lab_off],
            "templates": [P.get_str(pool, o) for o in tm_off],
        })
    path = os.path.join(HERE, "data", "zh_crf_models.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    print("模型数", len(out), "->", path)
    print("dat sha1", hashlib.sha1(data).hexdigest())


if __name__ == "__main__":
    main()
