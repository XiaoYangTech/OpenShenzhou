#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2p_ko.py —— 韩文文本 → 音素 ID（引擎标定词表 + 谚文分解兜底）

引擎实测格式（InJoon/SunHi）：
    안녕하세요.
    a n & n yeo ng & h a & s ei & yo / punc. / ~
- "&"(12) 分音节，"/"(46) 分词，结尾 "/ punc. / ~"
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUNCT = {".": "punc.", "?": "punc?", "!": "punc!", ",": "punc,", ";": "punc;"}

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def load_symbols(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip().replace("\ufeff", "")
        if not line:
            continue
        k, v = line.rsplit(":", 1)
        m[k] = int(v)
    return m


def load_lexicon():
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "lexicon_ko"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "lexicon_ko.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_syllables():
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "syllables_ko"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "syllables_ko.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def decompose(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def word_groups(word, lexicon, syllables):
    """词 → 每音节音素列表；词表命中则直接切 &"""
    if word in lexicon:
        ids = lexicon[word]
        groups = []
        cur = []
        amp = None
        for i, x in enumerate(ids):
            if i and x == ids[i]:
                pass
            cur.append(x)
        # 按 & 切分（& 的 id 在 build 时注入）
        return None
    return None


def _resolve_names_for(names, sym):
    out = []
    for x in names:
        if isinstance(x, int):
            out.append(x); continue
        if x in sym:
            out.append(sym[x]); continue
        cand = None
        for alt in (x.rstrip("0123456789"), x.replace("hh", "h"), x.replace("ss", "s")):
            if alt in sym:
                cand = sym[alt]; break
        # 引擎对未映射音素使用 '-' 占位（实测 SunHi: id 11='-'），而不是 'padding'
        if cand is None:
            cand = sym.get("-", sym.get("&", sym.get("padding", 0)))
        out.append(cand if cand is not None else 0)
    return out


def text_to_ids(text, sym, lexicon, syllables):
    from .symbols import resolve_table, resolve_seq, is_name_table
    syllables = resolve_table(syllables, sym)
    # 音节分隔符名：优先 '&'；部分音色包（SunHi）没有 '&'，用 '-' 充当分隔符。
    # 直接逐条解析（不走 resolve_lexicon 缓存，避免不同 dict 的 id 复用问题）
    sep_name = "&" if "&" in sym else ("-" if "-" in sym else None)
    if sep_name and sep_name != "&" and is_name_table(lexicon):
        lexicon = {
            k: resolve_seq([sep_name if x == "&" else x for x in v], sym)
            for k, v in lexicon.items()
        }
    else:
        from .symbols import resolve_lexicon
        lexicon = resolve_lexicon(lexicon, sym)
    amp = sym.get("&")
    slash = sym.get("/")
    tilde = sym.get("~")
    frame = ("-bos-" in sym or "-BOS-" in sym) and ("br0" in sym)
    bos = sym.get("-bos-", sym.get("-BOS-"))
    ids = []
    if frame and bos is not None:
        ids.extend([bos, sym.get("br0", 0)])
    words_out = []
    ended = False
    import re
    for chunk in re.split(r"([.?!,;])", text):
        if not chunk.strip():
            continue
        if chunk in PUNCT:
            ended = True
            punc_key = PUNCT[chunk]
            words_out.append(("P", punc_key))
            continue
        for w in re.findall(r"[\uac00-\ud7a3]+", chunk):
            words_out.append(("W", w))
    emitted = 0
    for kind, val in words_out:
        if kind == "P":
            if frame:
                ids.append(sym.get("br4", 0))
                ids.append(sym.get(val, 0))
                continue
            if slash is not None: ids.append(slash)
            ids.append(sym.get(val, 0))
            if slash is not None: ids.append(slash)
            continue
        w = val
        if frame:
            groups = []
            raw = lexicon.get(w)
            if raw:
                # 音节分隔符：优先 '&'，部分音色包（SunHi）用 '-' 充当分隔符
                sep = sym.get("&")
                if sep is None:
                    sep = sym.get("-")
                toks = []
                for x in list(raw):
                    if x == "&" or (sep is not None and x == sep):
                        toks.append(sep)
                    else:
                        toks.extend(_resolve_names_for([x], sym))
                cur = []
                for x in toks:
                    if sep is not None and x == sep:
                        if cur: groups.append(cur)
                        cur = []
                    else:
                        cur.append(x)
                if cur: groups.append(cur)
            else:
                for ch in w:
                    g = syllables.get(ch)
                    if g:
                        groups.append(list(g))
            if not groups:
                continue
            if emitted > 0:
                ids.append(sym.get("br1", sym.get("br0", 0)))
            sep = sym.get("-")
            for gi, g in enumerate(groups):
                if gi > 0 and sep is not None:
                    ids.append(sep)
                ids.extend(g)
            emitted += 1
            continue
        core = None
        if w in lexicon:
            raw = lexicon[w]
            # 词表存的是带 & 的 id 序列
            core = list(raw)
        else:
            groups = []
            for ch in w:
                g = syllables.get(ch)
                if g is None:
                    d = decompose(ch)
                    if d is None:
                        continue
                    g = []
                    for j in d:
                        if j and j in syllables:
                            g = g + syllables[j]
                    if not g:
                        continue
                groups.append(g)
            core = []
            for gi, g in enumerate(groups):
                if gi > 0 and amp is not None:
                    core.append(amp)
                core.extend(g)
        if not core:
            continue
        if emitted > 0 and slash is not None:
            ids.append(slash)
        ids.extend(core)
        emitted += 1
    if frame:
        if not ended:
            ids.append(sym.get("br4", 0))
        ids.append(sym.get("~", 0))
        return ids
    if not ended and slash is not None:
        ids.append(slash)
    ids.append(tilde)
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--package", default=os.path.join(
        os.environ.get("TEMP", ""), "decoded", "InJoon_Male_ko-KR"))
    a = ap.parse_args()
    sym = load_symbols(os.path.join(a.package, "phones.txt"))
    ids = text_to_ids(a.text, sym, load_lexicon(), load_syllables())
    rev = {v: k for k, v in sym.items()}
    print(json.dumps(ids))
    print(" ".join(rev.get(i, "?%d" % i) for i in ids))
