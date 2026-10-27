#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_ml_offline.py —— 8 语种前端「逐 id 一致率」离线评测（无需引擎）

golden 来自引擎 encoder 输入 ids（data/eval/ml_golden_138.json）。
用法::  python tools/eval_ml_offline.py [lang]
"""
import json
import os
import sys
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE
sys.path.insert(0, ROOT)
PKG_ROOT = _voice_root()
VOICE2PKG = {
    "Microsoft Xiaoxiao": "Xiaoxiao_Female_zh-CN",
    "Microsoft Aria": "Aria_Female_en-US",
    "Microsoft Denise": "Denise_Female_fr-FR",
    "Microsoft Katja": "Katja_Female_de-DE",
    "Microsoft Alvaro": "Alvaro_Male_es-ES",
    "Microsoft Francisca": "Francisca_Female_pt-BR",
    "Microsoft Keita": "Keita_Male_ja-JP",
    "Microsoft InJoon": "InJoon_Male_ko-KR",
}


def load_sym(pkg):
    sym = {}
    for line in open(os.path.join(PKG_ROOT, pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        sym[n] = int(i)
    return sym


def synth(module, text, pkg):
    sym = load_sym(pkg)
    if module == "zh":
        import openshenzhou.g2p_zh as m
        return m.text_to_ids(text, m.load_table(), sym, m.load_lexicon())
    if module == "ja":
        import openshenzhou.g2p_ja as m
        return m.text_to_ids(text, sym, m.load_kana(), m.load_lexicon())
    if module == "ko":
        import openshenzhou.g2p_ko as m
        return m.text_to_ids(text, sym, m.load_lexicon(), m.load_syllables())
    if module == "en":
        import openshenzhou.g2p_en as m
        return m.text_to_ids(text, sym, m.load_lexicon())
    import openshenzhou.g2p_latin as m
    return m.text_to_ids(text, sym, m.load_lexicon(module), module)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = json.load(open(os.path.join(HERE, "data", "eval", "ml_golden_138.json"),
                          encoding="utf-8"))
    stats = {}
    for r in rows:
        lang = r["lang"]
        if only and lang != only:
            continue
        st = stats.setdefault(lang, [0, 0])
        st[1] += 1
        try:
            ours = synth(lang, r["text"], VOICE2PKG.get(r["voice"], r["voice"]))
        except Exception as e:
            print("[%s] ERR %s: %s" % (lang, r["text"][:20], e))
            continue
        if ours == r["ids"]:
            st[0] += 1
        else:
            print("[%s] ✗ %s" % (lang, r["text"][:32]))
    tot = [0, 0]
    for lang in ("zh", "en", "fr", "de", "es", "pt", "ja", "ko"):
        if lang not in stats:
            continue
        ok, n = stats[lang]
        tot[0] += ok
        tot[1] += n
        print("%-3s %d/%d = %.1f%%" % (lang, ok, n, 100.0 * ok / max(1, n)))
    print("合计 %d/%d = %.1f%%" % (tot[0], tot[1], 100.0 * tot[0] / max(1, tot[1])))
    return 0 if tot[0] == tot[1] else 1


if __name__ == "__main__":
    sys.exit(main())
