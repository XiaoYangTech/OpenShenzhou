#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_crf_map_final.py —— 合并「动态探测结果 + pypinyin 候选最佳猜测」生成最终模型→汉字映射"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 动态探测已确认不可靠/无单字触发的模型，人工从标签读音推断（仍会被语料校验过滤）
MANUAL = {
    0: "挨", 11: "担", 17: "都", 22: "干", 26: "冠", 27: "行", 36: "教",
    39: "乐", 40: "勒", 42: "溜", 43: "露", 45: "蒙", 46: "模", 48: "难",
    50: "宁", 55: "圈", 62: "舍", 63: "什", 64: "乘", 73: "降", 78: "削",
    86: "卒", 10: "待",
}


def main():
    models = json.load(open(os.path.join(HERE, "data", "zh_crf_models.json"),
                            encoding="utf-8"))
    probed = {}
    pp = os.path.join(HERE, "data", "zh_crf_model_chars_probed.json")
    if os.path.exists(pp):
        probed = json.load(open(pp, encoding="utf-8"))
    pya = json.load(open(os.path.join(HERE, "data", "zh_char_pinyin_all.json"),
                         encoding="utf-8"))
    obs = json.load(open(os.path.join(HERE, "data", "zh_char_reading_sets.json"),
                         encoding="utf-8"))
    out = {}
    for m in models:
        idx = m["idx"]
        labels = [tuple(x.lower() for x in l.split("%")) for l in m["labels"]]
        entry = {"labels": labels, "char": None, "reason": ""}
        if any("-" in l for l in m["labels"]):
            entry["reason"] = "label-pair"
            out[str(idx)] = entry
            continue
        p = (probed.get(str(idx)) or {}).get("char")
        if p:
            entry["char"] = p
            entry["reason"] = "probed"
            out[str(idx)] = entry
            continue
        if idx in MANUAL:
            entry["char"] = MANUAL[idx]
            entry["reason"] = "manual"
            out[str(idx)] = entry
            continue
        L = set(labels)
        scored = []
        for ch, pyrs in pya.items():
            rs = set(tuple(x.split(".")) for x in pyrs)
            inter = len(rs & L)
            if inter == 0:
                continue
            miss = len(rs - L)
            extra = len(L - rs)
            freq = sum(obs.get(ch, {}).values())
            scored.append((-inter, miss, extra, -freq, ch))
        scored.sort()
        if scored:
            entry["char"] = scored[0][4]
            entry["reason"] = "guess"
            entry["cands"] = [c[4] for c in scored[:5]]
        else:
            entry["reason"] = "no-match"
        out[str(idx)] = entry
    json.dump(out, open(os.path.join(HERE, "data", "zh_crf_model_chars.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = sum(1 for e in out.values() if e["char"])
    print("最终映射", n, "/", len(out))
    for k in sorted(out, key=lambda x: int(x)):
        e = out[k]
        if e["reason"] in ("manual", "guess", "probed"):
            print("%3s %s %-3s %s" % (k, e["reason"], e["char"], "" ))


if __name__ == "__main__":
    main()
