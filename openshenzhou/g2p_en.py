#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2p_en.py —— 英文文本 → 微软语音包音素 ID

音素集：ARPAbet（CMUdict 风格），元音带重音号 1/2，0 重音写作无号；
schwa（AH0）用 ax 表示；HH 映射为 h。

  python tools/g2p_en.py --text "hello world" --package <语音包目录>
"""
import argparse
import json
import os
import re
import sys

try:
    import pronouncing
except ImportError:
    pronouncing = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOS, END = "-bos-", "~"


def load_lexicon():
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "lexicon_en"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "lexicon_en.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_symbols(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        name, idx = line.rsplit(":", 1)
        m[name] = int(idx)
    return m


# CMU → 引擎音素名
def cmu_to_engine(ph):
    m = re.match(r"^([A-Z]+)([012])?$", ph)
    if not m:
        return ph.lower()
    base, stress = m.group(1), m.group(2)
    low = base.lower()
    if base == "HH":
        return "h"
    if base == "AH" and (stress is None or stress == "0"):
        return "ax"
    if stress in ("1", "2"):
        return low + stress
    return low


# 字母 → 音素（OOV 兜底，粗略规则）
LETTER_RULES = {
    "a": ["AE1"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH1"], "f": ["F"],
    "g": ["G"], "h": ["HH"], "i": ["IH1"], "j": ["JH"], "k": ["K"], "l": ["L"],
    "m": ["M"], "n": ["N"], "o": ["OW1"], "p": ["P"], "q": ["K"], "r": ["R"],
    "s": ["S"], "t": ["T"], "u": ["AH1"], "v": ["V"], "w": ["W"], "x": ["K", "S"],
    "y": ["Y"], "z": ["Z"],
}


def word_phones(word):
    """英文单词 → CMU 音素列表"""
    w = word.lower()
    if pronouncing is not None:
        ph = pronouncing.phones_for_word(w)
        if ph:
            return ph[0].split()
    # 数字
    if w.isdigit():
        return number_phones(w)
    # 兜底：逐字母
    out = []
    for ch in w:
        out.extend(LETTER_RULES.get(ch, []))
    return out


ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    if n < 1000000:
        return number_words(n // 1000) + " thousand" + (" " + number_words(n % 1000) if n % 1000 else "")
    return number_words(n // 1000000) + " million" + (" " + number_words(n % 1000000) if n % 1000000 else "")


def number_phones(ds):
    out = []
    for w in number_words(int(ds)).split():
        out.extend(word_phones(w))
    return out


PUNCT = {",": "punc,", ".": "punc.", "?": "punc?", "!": "punc!", ";": "punc;",
         "…": "punc…", "...": "punc..."}


VOWELS = set("aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw".split())

# 引擎把冠词 a/an 读成弱读（a→ax），词表里存的是字母读音 → 特判
EN_WEAK = {
    "a": ["ax"],
    "an": ["ax", "n"],
    "the": ["dh", "ax1"],
    "for": ["f", "ax1", "r"],
    "of": ["ax", "v"],
    "to": ["t", "ax"],
    "and": ["ax", "n", "d"],
    "are": ["aa1", "r"],
}

VOWEL_LETTERS = set("aeiou")


ONSETS = set("""
p b t d k g f v th dh s z sh zh ch jh m n ng l r h w y
pl pr bl br tr dr kl kr gl gr fl fr th r sl sw sp st sk sm sn
spl spr str skr skw shr thr
""".split())


def split_syllables(phones):
    """最大 onset 原则的音节切分（用于词内 '-' 分隔）。"""
    n = len(phones)
    nuclei = [i for i, p in enumerate(phones) if p.rstrip("012") in VOWELS]
    if not nuclei:
        return [list(phones)]
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phones[a + 1:b]
        if not cluster:
            cuts.append(b)
            continue
        # 从最长的合法 onset 后缀开始找
        cut = a + 1
        for k in range(len(cluster), 0, -1):
            if "".join(x.rstrip("012") for x in cluster[len(cluster) - k:]) in ONSETS:
                cut = a + 1 + (len(cluster) - k)
                break
        cuts.append(cut)
    groups = []
    prev = 0
    for c in cuts:
        if c <= prev:
            continue
        groups.append(list(phones[prev:c]))
        prev = c
    groups.append(list(phones[prev:]))
    return [g for g in groups if g]


def text_to_ids(text, sym, lexicon=None):
    """引擎实测格式：
       -bos- br0 <word1 phones> br1 <word2 phones> ... br4 [punc.] ~
       词内音节用 "-" 分隔；逗号 → br3 punc, br0；句号 → br4 punc."""
    ids = [sym[BOS], sym["br0"]]
    first = True
    ended = False
    from .symbols import resolve_lexicon
    lex = resolve_lexicon(lexicon or {}, sym)
    for chunk in re.split(r"([,.?!;…])", text):
        if not chunk.strip():
            continue
        if chunk in PUNCT:
            key = PUNCT[chunk]
            if chunk in (".", "?", "!"):
                ids.append(sym["br4"])
                ids.append(sym.get(key, 0))
                ended = True
            else:
                ids.append(sym.get("br3", 0))
                ids.append(sym.get(key, 0))
                ids.append(sym["br0"])
                first = True
            continue
        words = re.findall(r"[A-Za-z0-9']+", chunk)
        for wi, w in enumerate(words):
            weak = EN_WEAK.get(w.lower())
            nxt = words[wi + 1] if wi + 1 < len(words) else ""
            if w.lower() == "the" and nxt and nxt[0].lower() in VOWEL_LETTERS:
                weak = ["dh", "ih1"]
            if w.lower() == "a" and nxt and nxt[0].lower() in VOWEL_LETTERS:
                weak = ["ax", "n"]
            core = ([sym.get(x, sym.get("<pad>", 0)) for x in weak] if weak
                    else lex.get(w.lower()))
            if core is None:
                phones = [cmu_to_engine(p) for p in word_phones(w)]
                sep = sym.get("-")
                groups = split_syllables(phones)
                core = []
                for gi, g in enumerate(groups):
                    if gi > 0 and sep is not None:
                        core.append(sep)
                    core.extend(sym.get(p, sym.get("<pad>", 0)) for p in g)
            if not first:
                ids.append(sym["br1"])
            ids.extend(core)
            first = False
    if not ended:
        ids.append(sym["br4"])
    ids.append(sym[END])
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--package", default=os.path.join(
        os.environ.get("TEMP", ""), "decoded", "Aria_Female_en-US"))
    a = ap.parse_args()
    sym = load_symbols(os.path.join(a.package, "phones.txt"))
    ids = text_to_ids(a.text, sym)
    rev = {v: k for k, v in sym.items()}
    print(json.dumps(ids))
    print(" ".join(rev.get(i, "?%d" % i) for i in ids))
