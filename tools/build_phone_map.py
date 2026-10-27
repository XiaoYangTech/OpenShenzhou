#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_phone_map.py —— 音素映射表的生成 / 校验工具

三件事：
  1) --sapi      从 /phoneset 拉 SAPI 中文音素表（408 音节 + 5 声调），写成 json。
                 这张表是**完整且实测可得**的（SpPhoneConverter 提供），SAPI 系音色直接用。
  2) --template  生成 am_v5 音素表的**骨架**：408 个音节按 声母/韵母 分组，值留空。
                 填法：把 am_v5 的 193 个符号对到 21 个声母 + 35 个韵母上，不是 408 条。
  3) --check     校验已填的表：覆盖率、引用了但不存在的符号、缺失音节。

用法：
  python tools/build_phone_map.py --sapi      --server http://127.0.0.1:17321
  python tools/build_phone_map.py --template
  python tools/build_phone_map.py --check frontend/phone_map.json --package "D:\\Voice\\Xiaoxiao_Female_zh-CN"
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "..", "frontend")

# 普通话声母（含零声母）；韵母部分从音节里反推
INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]


def fetch_phoneset(server, maxid=500):
    with urllib.request.urlopen("%s/phoneset?max=%d" % (server.rstrip("/"), maxid), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def split_syllable(syl):
    """zhang -> (zh, ang)"""
    for ini in INITIALS:
        if syl.startswith(ini) and len(syl) > len(ini):
            return ini, syl[len(ini):]
    return "", syl


def cmd_sapi(server, out):
    d = fetch_phoneset(server)
    sym = {p["symbol"]: p["id"] for p in d["phones"]}
    syllables = sorted(s for s in sym if s.isalpha() and s.islower())
    tones = sorted(s for s in sym if s.isdigit())
    seps = sorted(s for s in sym if not s.isalnum())
    table = {
        "_note": "SAPI 中文音素表（SpPhoneConverter, LanguageId=0x804 实测导出）。音节是整音节，声调是独立 token。",
        "language": "zh-CN",
        "count": len(sym),
        "syllables": syllables,
        "tones": tones,
        "separators": seps,
        "symbol2id": sym,
        "id2symbol": {str(v): k for k, v in sym.items()},
    }
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print("写出 %s：%d 音节, %d 声调, %d 分隔符" % (out, len(syllables), len(tones), len(seps)))
    return table


def cmd_template(out):
    syl_path = os.path.join(FRONTEND, "sapi_zh_cn_phonetable.json")
    if not os.path.exists(syl_path):
        print("先跑 --sapi 生成音节表")
        return 1
    with io.open(syl_path, encoding="utf-8") as f:
        table = json.load(f)

    entries, groups = {}, {}
    for s in table["syllables"]:
        ini, fin = split_syllable(s)
        tone = "T"          # 声调待定：1/2/3/4/5 怎么编码（数字后缀 还是 _h/_m/_l 寄存器）
        entries[s] = []
        groups.setdefault(ini or "∅", set()).add(fin)

    doc = {
        "_note": "am_v5 音素表骨架：408 个音节，值请填成 phones.txt 里的符号序列，例如 \"zhang1\": [\"zh\",\"aa\",\"ng1\"]",
        "_status": "未填充 —— 数值需要‘对原版词典挂钩’或逐条听测才能确定，见 docs/phone_map_reverse.md",
        "_tone_placeholder": tone,
        "_am_v5_symbols_hint": "phones.txt 里是 ARPAbet/UPS 风格：aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw + b ch d f g hh jh k l m n ng p r s sh t th tsh v z zh zz …（音节里带 1~5 或 _h/_m/_l）",
        "initials": {k: sorted(v) for k, v in sorted(groups.items())},
        "syllables": entries,
    }
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("写出 %s：%d 个音节；声母 %d 组" % (out, len(entries), len(groups)))
    print("声母/韵母对照（填表只需对 %d 个声母 + %d 个韵母）:" % (
        len([g for g in groups if g != "∅"]), len(set().union(*groups.values()))))
    for g in sorted(groups):
        print("  %-4s -> %s" % (g or "∅", " ".join(sorted(groups[g]))))
    return 0


def cmd_check(map_path, package):
    with io.open(map_path, encoding="utf-8") as f:
        m = json.load(f)
    entries = m.get("syllables", m)
    filled = {k: v for k, v in entries.items() if isinstance(v, list) and v}

    pkg_syms = set()
    if package:
        p = os.path.join(package, "phones.txt")
        if os.path.exists(p):
            for line in io.open(p, encoding="utf-8", errors="replace"):
                if ":" in line:
                    pkg_syms.add(line.rsplit(":", 1)[0].strip())

    used = set()
    for k, v in filled.items():
        used.update(v)

    bad = sorted(u for u in used if pkg_syms and u not in pkg_syms)
    print("总音节: %d, 已填: %d (%.1f%%)" % (len(entries), len(filled),
                                            100.0 * len(filled) / max(1, len(entries))))
    if pkg_syms:
        print("引用到的符号: %d, 其中 phones.txt 里不存在: %d" % (len(used), len(bad)))
        if bad:
            print("  未知符号: " + ", ".join(bad[:30]))
    missing = [k for k in entries if k not in filled]
    if missing:
        print("未填示例: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapi", action="store_true")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--check")
    ap.add_argument("--package")
    ap.add_argument("--server", default="http://127.0.0.1:17321")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.sapi:
        out = a.out or os.path.join(FRONTEND, "sapi_zh_cn_phonetable.json")
        cmd_sapi(a.server, out)
    if a.template:
        out = a.out or os.path.join(FRONTEND, "phone_map.json")
        return cmd_template(out)
    if a.check:
        return cmd_check(a.check, a.package)
    if not (a.sapi or a.template or a.check):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
