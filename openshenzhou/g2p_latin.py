#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2p_latin.py —— 拉丁语系通用文本前端（西/法/德/葡）

词表来自引擎自动标定（tools/harvest_lexicon_latin.py），OOV 用基础字母规则兜底。

  python tools/g2p_latin.py --lang es --text "hola mundo." --package <语音包目录>
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUNCT_KEYS = {",": "punc,", ".": "punc.", "?": "punc?", "!": "punc!", ";": "punc;"}


def load_symbols(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        k, v = line.rsplit(":", 1)
        m[k] = int(v)
    return m


def S(sym, name, default=None):
    """大小写不敏感地查符号（不同语音包 -bos- 大小写不一致）"""
    if name in sym:
        return sym[name]
    low = {k.lower(): v for k, v in sym.items()}
    v = low.get(name.lower())
    if v is not None:
        return v
    if default is not None:
        return default
    raise KeyError(name)


def load_lexicon(lang):
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "lexicon_%s" % lang))
    if d:
        return d
    p = os.path.join(ROOT, "data", "lexicon_%s.json" % lang)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


# 各语言基础字母→音素（兜底，仅覆盖常见拼读）
LETTER_RULES = {
    "es": {"a": ["a"], "e": ["e"], "i": ["i"], "o": ["o"], "u": ["u"],
           "b": ["b"], "c": ["k"], "d": ["d"], "f": ["f"], "g": ["g"], "h": [],
           "j": ["x"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"], "ñ": ["nj"],
           "p": ["p"], "q": ["k"], "r": ["rr"], "s": ["s"], "t": ["t"], "v": ["b"],
           "w": ["w"], "x": ["x"], "y": ["y"], "z": ["s"]},
    "fr": {"a": ["a"], "e": ["eh"], "i": ["iy"], "o": ["oh"], "u": ["uy"],
           "b": ["b"], "c": ["k"], "d": ["d"], "f": ["f"], "g": ["g"], "h": [],
           "j": ["zh"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"],
           "p": ["p"], "q": ["k"], "r": ["r"], "s": ["s"], "t": ["t"], "v": ["v"],
           "w": ["w"], "x": ["ks"], "y": ["iy"], "z": ["z"]},
    "de": {"a": ["a"], "e": ["eh"], "i": ["iy"], "o": ["oh"], "u": ["uh"],
           "b": ["b"], "c": ["k"], "d": ["d"], "f": ["f"], "g": ["g"], "h": ["h"],
           "j": ["y"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"],
           "p": ["p"], "q": ["k"], "r": ["r"], "s": ["z"], "t": ["t"], "v": ["f"],
           "w": ["v"], "x": ["ks"], "y": ["ue"], "z": ["ts"]},
    "pt": {"a": ["a"], "e": ["e"], "i": ["i"], "o": ["o"], "u": ["u"],
           "b": ["b"], "c": ["k"], "d": ["d"], "f": ["f"], "g": ["g"], "h": [],
           "j": ["zh"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"],
           "p": ["p"], "q": ["k"], "r": ["rr"], "s": ["s"], "t": ["t"], "v": ["v"],
           "w": ["w"], "x": ["sh"], "y": ["y"], "z": ["z"]},
}


def word_phones(word, lang, sym):
    out = []
    for ch in word.lower():
        for p in LETTER_RULES.get(lang, {}).get(ch, []):
            if p in sym:
                out.append(sym[p])
    return out


def _has_bos(sym):
    return any(k.lower() == "-bos-" for k in sym)


FR_LIAISON = {
    "est": "t", "les": "z", "des": "z", "mes": "z", "tes": "z", "ses": "z",
    "ces": "z", "deux": "z", "trois": "z", "nous": "z", "vous": "z",
    "ils": "z", "elles": "z", "on": "n", "en": "n", "un": "n", "mon": "n",
    "ton": "n", "son": "n", "bien": "n", "trop": "p", "quand": "t", "grand": "t",
    "dix": "z", "six": "z", "neuf": "v", "pas": "z", "ont": "t", "vont": "t",
    "font": "t", "sont": "t", "faut": "t", "chez": "z", "très": "z", "plus": "z",
}
FR_VOWEL_INIT = set("aeiouyAEIOUY")


def _fr_is_vowel(ph_name):
    if not ph_name:
        return False
    return ph_name[0] in FR_VOWEL_INIT and not ph_name.startswith(("uw", "oh", "oy"))


# 需要「替换词尾辅音」的连诵（dix→diz、six→siz、neuf→nœv）
FR_LIAISON_REPLACE = {"dix": ("s", "z"), "six": ("s", "z"), "neuf": ("f", "v")}


def apply_fr_liaison(ids, word_infos, sym, rev):
    """法语连诵：prev 词以特定辅音连诵到元音开头的 next 词（直接改 ids）。

    - 普通连诵：在 next 词开头插入辅音（est→t、les→z、pas→z…）；
    - 替换型（dix/six/neuf）：删掉 prev 词尾的 s/f，再在 next 词开头插入 z/v。
    """
    dels = []
    ins = []
    for k in range(1, len(word_infos)):
        prev = word_infos[k - 1]
        nxt = word_infos[k]
        core = prev.get("core") or []
        nxt_core = nxt.get("core") or []
        if not core or not nxt_core:
            continue
        if not _fr_is_vowel(rev.get(nxt_core[0], "")):
            continue
        w_prev = (prev.get("word") or "").lower()
        rep = FR_LIAISON_REPLACE.get(w_prev)
        if rep:
            old, newc = rep
            nid = sym.get(newc)
            if nid is None:
                continue
            for ti in range(len(core) - 1, -1, -1):
                if rev.get(core[ti], "") == old:
                    pos = prev["end"] - (len(core) - ti)
                    if 0 <= pos < len(ids):
                        dels.append(pos)
                    ins.append((nxt["start"], nid))
                    break
            continue
        cons_name = FR_LIAISON.get(w_prev)
        if not cons_name:
            continue
        cons = sym.get(cons_name)
        if cons is None:
            continue
        ins.append((nxt["start"], cons))
    if not dels and not ins:
        return ids
    for pos, cid in sorted(ins, key=lambda x: -x[0]):
        ids.insert(pos, cid)
    for pos in sorted(dels, reverse=True):
        del ids[pos]
    return ids


def text_to_ids(text, sym, lexicon, lang):
    if not _has_bos(sym):
        return text_to_ids_pt(text, sym, lexicon)
    ids = [S(sym, "-bos-"), S(sym, "br0")]
    first = True
    ended = False
    word_infos = []
    from .symbols import resolve_lexicon
    lex = {k.lower(): v for k, v in resolve_lexicon(lexicon or {}, sym).items()}
    for chunk in re.split(r"([,.?!;])", text):
        if not chunk.strip():
            continue
        if chunk in PUNCT_KEYS:
            key = PUNCT_KEYS[chunk]
            if chunk in (".", "?", "!"):
                ids.append(S(sym, "br4"))
                ids.append(sym.get(key, sym.get("punc.", 6)))
                ended = True
            else:
                ids.append(S(sym, "br3", S(sym, "br2", 0)))
                ids.append(sym.get(key, 0))
                ids.append(S(sym, "br0"))
                first = True
            continue
        words = re.findall(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", chunk, re.UNICODE)
        for w in words:
            core = lex.get(w.lower())
            if core is None and ("'" in w or "’" in w):
                parts = [p for p in re.split(r"[’']", w) if p]
                cores = []
                for pi2, p2 in enumerate(parts):
                    if pi2 < len(parts) - 1 and len(p2) == 1:
                        cores.append(word_phones(p2, lang, sym))  # 省音字母读辅音
                    else:
                        cores.append(lex.get(p2.lower()) or word_phones(p2, lang, sym))
                core = []
                for c in cores:
                    core.extend(c)
            elif core is None and "-" in w:
                # 连字符复合词：引擎按一个词处理，连字符处是音节分隔
                parts = [p for p in w.split("-") if p]
                cores = [lex.get(p.lower()) or word_phones(p, lang, sym) for p in parts]
                core = []
                sep = sym.get("-")
                for gi, c in enumerate(cores):
                    if gi > 0 and sep is not None:
                        core.append(sep)
                    core.extend(c)
            elif core is None:
                core = word_phones(w, lang, sym)
            if not core:
                continue
            if not first:
                ids.append(S(sym, "br1"))
            start_i = len(ids)
            ids.extend(core)
            word_infos.append({"word": w, "core": core,
                               "start": start_i, "end": len(ids)})
            first = False
    if lang == "fr" and word_infos:
        rev = {v: k for k, v in sym.items()}
        try:
            apply_fr_liaison(ids, word_infos, sym, rev)
        except Exception:
            pass
    if not ended:
        ids.append(S(sym, "br4"))
    # 德语语音包格式：句末 punc 之后还有一个 "/" 再过 "~"（引擎实测）
    if lang == "de" and not _pt_flat(sym):
        slash = sym.get("/")
        if slash is not None:
            ids.append(slash)
    ids.append(S(sym, "~"))
    return ids


# ---------------- 葡萄牙语：没有 -BOS-/br，用 & 分音节、/ 分词 ----------------
def _pt_flat(sym):
    """实测 Francisca 也使用「词内 -、词间 /、punc. /」格式，故不再走纯平铺分支。"""
    return False


def text_to_ids_pt(text, sym, lexicon):
    def sid(name):
        for k, v in sym.items():
            if k.lower() == name:
                return v
        return None
    slash = sid("/")
    tilde = sid("~")
    from .symbols import resolve_lexicon
    lex = {k.lower(): v for k, v in resolve_lexicon(lexicon or {}, sym).items()}
    flat = _pt_flat(sym)
    word_groups = []
    end_punct = None
    for chunk in re.split(r"([,.?!;])", text):
        if not chunk.strip():
            continue
        if chunk in PUNCT_KEYS:
            key = PUNCT_KEYS[chunk]
            end_punct = sym.get(key, sym.get("punc.", 5))
            continue
        for w in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", chunk, re.UNICODE):
            core = lex.get(w.lower())
            if core is None and "-" in w:
                parts = [p for p in w.split("-") if p]
                cores = [lex.get(p.lower()) or word_phones(p, "pt", sym) for p in parts]
                core = []
                sep = sym.get("-")
                for gi, c in enumerate(cores):
                    if gi > 0 and sep is not None:
                        core.append(sep)
                    core.extend(c)
            elif core is None:
                core = word_phones(w, "pt", sym)
            else:
                while core and slash is not None and core[-1] == slash:
                    core = core[:-1]
            if core:
                if flat:
                    # 标定数据里词尾标记会重复一次，去掉多余尾
                    while len(core) >= 2 and core[-1] == core[-2]:
                        core = core[:-1]
                word_groups.append(core)
    if flat:
        # Francisca 类葡语音色：纯音素串 + 结尾 "-"
        ids = []
        for grp in word_groups:
            ids.extend([x for x in grp if x is not None])
        dash = sid("-")
        if dash is not None:
            ids.append(dash)
        return ids
    ids = []
    for i, grp in enumerate(word_groups):
        if i > 0 and slash is not None:
            ids.append(slash)
        ids.extend(grp)
    if end_punct is not None:
        if slash is not None:
            ids.append(slash)
        ids.append(end_punct)
    if slash is not None:
        ids.append(slash)
    if tilde is not None:
        ids.append(tilde)
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--package", default="")
    a = ap.parse_args()
    pkg = a.package or os.path.join(os.environ.get("TEMP", ""), "decoded", {
        "es": "Alvaro_Male_es-ES", "fr": "Denise_Female_fr-FR",
        "de": "Conrad_Male_de-DE", "pt": "Antonio_Male_pt-BR"}.get(a.lang, ""))
    sym = load_symbols(os.path.join(pkg, "phones.txt"))
    ids = text_to_ids(a.text, sym, load_lexicon(a.lang), a.lang)
    rev = {v: k for k, v in sym.items()}
    print(json.dumps(ids))
    print(" ".join(rev.get(i, "?%d" % i) for i in ids))
