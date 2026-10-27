#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把按 id 存储的标定数据转成「符号名」存储，使其与具体音色包无关。

同一语言不同音色（如 Xiaoxiao / Yunxi）的 phones.txt 的 id 映射不同，
所以 lexicon / kana / syllables 必须按符号名保存，运行时再用目标音色包的
phones.txt 映射回 id。

  python tools/symbolize_data.py --voice-root %TEMP%/AllVoice --data-dir ../OpenShenzhou/data
"""
import argparse
import json
import os
import shutil

LANG_PKGS = {
    "zh": "Xiaoxiao_Female_zh-CN",
    "en": "Aria_Female_en-US",
    "es": "Alvaro_Male_es-ES",
    "fr": "Denise_Female_fr-FR",
    "de": "Conrad_Male_de-DE",
    "pt": "Francisca_Female_pt-BR",
    "ja": "Keita_Male_ja-JP",
    "ko": "InJoon_Male_ko-KR",
}


def load_phones(pkg):
    m = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip().replace("\ufeff", "")
        if not line:
            continue
        k, v = line.rsplit(":", 1)
        m[int(v)] = k
    return m


def symbolize(entry, id2name):
    return [id2name.get(x, "?%d" % x) for x in entry]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-root", required=True)
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    d = a.data_dir
    for lang, pkgname in LANG_PKGS.items():
        pkg = os.path.join(a.voice_root, pkgname)
        if not os.path.exists(pkg):
            print("跳过", lang, "(缺", pkg, ")")
            continue
        id2name = load_phones(pkg)
        lex_path = os.path.join(d, "lexicon_%s.json" % lang)
        if os.path.exists(lex_path):
            lex = json.load(open(lex_path, encoding="utf-8"))
            # 备份原 id 版
            base = os.path.join(d, "lexicon_%s.ids.json" % lang)
            if not os.path.exists(base):
                shutil.copy(lex_path, base)
            named = {w: symbolize(v, id2name) for w, v in lex.items()}
            json.dump(named, open(os.path.join(d, "lexicon_%s.names.json" % lang), "w", encoding="utf-8"),
                      ensure_ascii=False, separators=(",", ":"))
            print("%-3s lexicon %d 条 -> names" % (lang, len(named)))
        # 其它按 id 存的结构
        for fname, out in (("kana_ja.json", "kana_ja.names.json"),
                           ("syllables_ko.json", "syllables_ko.names.json")):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            if lang == "ja" and fname != "kana_ja.json":
                continue
            if lang == "ko" and fname != "syllables_ko.json":
                continue
            data = json.load(open(p, encoding="utf-8"))
            base = os.path.join(d, fname.replace(".json", ".ids.json"))
            if not os.path.exists(base):
                shutil.copy(p, base)
            named = {k: symbolize(v, id2name) for k, v in data.items()}
            json.dump(named, open(os.path.join(d, out), "w", encoding="utf-8"),
                      ensure_ascii=False, separators=(",", ":"))
            print("%-3s %s %d 条 -> names" % (lang, fname, len(named)))


if __name__ == "__main__":
    main()
