#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""g2p_en.py —— 英文文本 → 微软语音包音素 ID（引擎行为复刻版）

数据来源全部为引擎实测（ORT 探针标定）：
  data/lexicon_en_context.json   语料句内「词 → 音素名序列」（含弱读形态）
  data/lexicon_en_engine.json    单词隔离标定的「词 → 音素名序列」
  data/lexicon_en.names.json     早期标定词表
  data/cmudict_en.json           CMUdict 兜底

TN（数字/日期/时间/货币/单位/电话/分数/罗马数字/缩写）规则来自引擎探针实测，
见 data/eval/en_engine_tn.jsonl 的逐 id 对比。
"""
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

_CONTEXT = None
_ISO = None
_VOCAB = None
_HETERO = None
_ACRONYMS = None
_CASE = None
HETERO_NOUN_PREV = {"ip", "email", "web", "home", "street", "url", "ftp", "http"}
VERB_PREV = {"to", "will", "would", "can", "could", "should", "must", "may", "might",
             "do", "does", "did", "don't", "doesn't", "didn't", "i", "you", "we", "they",
             "he", "she", "let", "please", "cannot", "won't"}
NOUN_PREV = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
             "her", "its", "our", "their", "no", "every", "each", "any", "some"}

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]
MONTH_ABBR = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
}
ORD_ONES = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
            "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
            "eighteenth", "nineteenth"]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]
ORD_TENS = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
            "sixtieth", "seventieth", "eightieth", "ninetieth"]
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
CNT_WORDS = {"chapter", "part", "section", "book", "volume", "act", "scene",
             "article", "item", "figure", "table", "appendix", "page", "no"}

UNITS = {
    "km": ["kilometers"], "kmh": ["kilometers", "per", "hour"],
    "kg": ["kilograms"], "mg": ["milligrams"], "g": ["grams"],
    "cm": ["centimeters"], "mm": ["millimeters"],
    "ft": ["feet"], "yd": ["yards"], "mi": ["miles"],
    "mph": ["miles", "per", "hour"], "kbps": ["kilobits", "per", "second"],
    "mb": ["megabytes"], "gb": ["gigabytes"], "kb": ["kilobytes"],
    "tb": ["terabytes"], "ms": ["milliseconds"],
    "h": ["hours"], "hr": ["hours"], "hrs": ["hours"],
    "min": ["minutes"], "mins": ["minutes"],
    "sec": ["seconds"], "secs": ["seconds"],
    "ml": ["milliliters"], "oz": ["ounces"], "lb": ["pounds"],
    "lbs": ["pounds"],
}
UNIT_SINGULAR = {"hours": "hour", "minutes": "minute", "seconds": "second",
                 "feet": "foot", "inches": "inch", "miles": "mile",
                 "meters": "meter", "centimeters": "centimeter",
                 "kilometers": "kilometer", "kilograms": "kilogram"}
CURRENCY = {"$": ("dollar", "cent"), "€": ("euro", "cent"),
            "£": ("pound", "penny"), "¥": ("yen", "sen"),
            "￥": ("yen", "sen"), "usd": ("dollar", "cent"),
            "eur": ("euro", "cent"), "gbp": ("pound", "penny")}
SPECIAL_PHONES = {
    "underscore": ["ah2", "n", "-", "d", "ax", "r", "-", "s", "k", "ao1", "r"],
    "asterisk": ["ae1", "-", "s", "t", "ax", "-", "r", "ih", "s", "k"],
    "o'clock": ["ax", "-", "k", "l", "aa1", "k"],
    "covid": ["k", "ah1", "-", "v", "ih", "d"],
}
SYMBOL_WORDS = {"*": "asterisk", "_": "underscore", "#": "number", "@": "at",
                "+": "plus", "<": "less", ">": "greater",
                "~": None, "|": "bar", "\\": None}
STRONG_AND = ["ae1", "n", "d"]
STRONG_FOR = ["f", "ao1", "r"]
PAUSE_PUNCT = {",": "punc,", ";": "punc;", ":": None, "(": "punc,",
               ")": "punc,", "[": "punc,", "]": "punc,"}
BQUOTE = "\"’”'"
END_PUNCT = {".": "punc.", "?": "punc?", "!": "punc!"}


# ---------------- 基础工具 ----------------
def _load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def load_context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = _load_json(os.path.join(ROOT, "data", "lexicon_en_context.json"), {})
    return _CONTEXT


def load_engine_lexicon():
    global _ISO
    if _ISO is None:
        iso = {}
        d = _load_json(os.path.join(ROOT, "data", "lexicon_en_engine.json"), {})
        if d:
            iso = dict(d)
        p2 = os.path.join(ROOT, "data", "lexicon_en_engine.jsonl")
        if os.path.exists(p2):
            for line in open(p2, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                iso.setdefault(r["w"], r["phones"])
        _ISO = iso
    return _ISO


def load_acronyms():
    global _ACRONYMS
    if _ACRONYMS is None:
        _ACRONYMS = _load_json(os.path.join(ROOT, "data", "lexicon_en_acronyms.json"), {})
    return _ACRONYMS


def load_case_words():
    global _CASE
    if _CASE is None:
        _CASE = _load_json(os.path.join(ROOT, "data", "lexicon_en_case.json"), {})
    return _CASE


def acronym_phones(letters):
    acr = load_acronyms()
    ph = acr.get(letters) or acr.get(letters.upper())
    if ph:
        return list(ph)
    return _letter_token(letters.lower())


def load_heteronyms():
    global _HETERO
    if _HETERO is None:
        _HETERO = _load_json(os.path.join(ROOT, "data", "lexicon_en_heteronyms.json"), {})
    return _HETERO


_VARIANT_LEX = {}


_ACTIVE_LEX = None
_ACTIVE_VARIANT = ""


def _active_lexicon():
    return _ACTIVE_LEX if _ACTIVE_LEX is not None else load_engine_lexicon()


def load_lexicon(variant=None):
    """英文词表；variant='gb'/'in' 时用引擎实测的英式/印式发音词表覆盖。"""
    base = load_engine_lexicon()
    if variant not in ("gb", "in"):
        return base
    if variant in _VARIANT_LEX:
        return _VARIANT_LEX[variant]
    import json as _json
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data",
                         "lexicon_en_%s.json" % variant)
    lex = dict(base)
    if _os.path.exists(path):
        try:
            lex.update(_json.load(open(path, encoding="utf-8")))
        except Exception:
            pass
    _VARIANT_LEX[variant] = lex
    return lex


def text_to_ids(text, sym, lexicon=None, variant=""):
    """包装：在本次转换期间把词表切换为传入的（变体）词表"""
    global _ACTIVE_LEX, _ACTIVE_VARIANT
    prev = _ACTIVE_LEX
    prev_var = _ACTIVE_VARIANT
    # 仅 en-GB/en-IN 变体启用"变体词表优先"；en-US 保持原有上下文/规则路径
    if lexicon is not None and variant in ("gb", "in"):
        _ACTIVE_LEX = lexicon
        _ACTIVE_VARIANT = variant
    try:
        return _text_to_ids_impl(text, sym, lexicon, variant)
    finally:
        _ACTIVE_LEX = prev
        _ACTIVE_VARIANT = prev_var


# 印式英语：个别音素在 *_hin 符号表中缺失时的回退映射（引擎实测的印式口音音素）
_ALIAS_IN = {"w": "vw", "v": "vw", "d": "dr", "t": "tr", "er": "ah", "dh": "d", "ax": "ah"}


def _sym_id(sym, ph, variant=""):
    v = sym.get(ph)
    if v is not None:
        return v
    base = (ph or "").rstrip("012")
    if variant == "in":
        alt = _ALIAS_IN.get(base)
        if alt is not None and sym.get(alt) is not None:
            return sym[alt]
        if sym.get(base) is not None:
            return sym[base]
    elif variant in ("gb", "in"):
        if sym.get(base) is not None:
            return sym[base]
    return sym.get("<pad>", 0)


def load_symbols(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        name, idx = line.rsplit(":", 1)
        m[name] = int(idx)
    return m


FUNCTION_WORDS = set("""a an the and of to for in on at by with from as or but not no so
do does did has have had is are was were be been am being
i you he she we they it me him her us them my your his its our their
this that these those there here who what which when where why how
will would can could shall should may might must""".split())

VOWELS = set("aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw".split())
ONSETS = set("""
p b t d k g f v th dh s z sh zh ch jh m n ng l r h w y
pl pr bl br tr dr kl kr gl gr fl fr sl sw sp st sk sm sn
spl spr str skr skw shr thr
""".split())
LETTER_RULES = {
    "a": ["ey1"], "b": ["b", "iy1"], "c": ["s", "iy1"], "d": ["d", "iy1"],
    "e": ["iy1"], "f": ["eh1", "f"], "g": ["jh", "iy1"], "h": ["ey1", "ch"],
    "i": ["ay1"], "j": ["jh", "ey1"], "k": ["k", "ey1"], "l": ["eh1", "l"],
    "m": ["eh1", "m"], "n": ["eh1", "n"], "o": ["ow1"], "p": ["p", "iy1"],
    "q": ["k", "y", "uw1"], "r": ["aa1", "r"], "s": ["eh1", "s"], "t": ["t", "iy1"],
    "u": ["y", "uw1"], "v": ["v", "iy1"], "w": ["d", "ah1", "b", "ax", "l", "y", "uw1"],
    "x": ["eh1", "k", "s"], "y": ["w", "ay1"], "z": ["z", "iy1"],
}


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


def split_syllables(phones):
    nuclei = [i for i, p in enumerate(phones) if p.rstrip("012") in VOWELS]
    if not nuclei:
        return [list(phones)]
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phones[a + 1:b]
        if not cluster:
            cuts.append(b)
            continue
        cut = a + 1
        for k in range(len(cluster), 0, -1):
            if "".join(x.rstrip("012") for x in cluster[len(cluster) - k:]) in ONSETS:
                cut = a + 1 + (len(cluster) - k)
                break
        cuts.append(cut)
    groups, prev = [], 0
    for c in cuts:
        if c <= prev:
            continue
        groups.append(list(phones[prev:c]))
        prev = c
    groups.append(list(phones[prev:]))
    return [g for g in groups if g]


def with_syllables(phones):
    out = []
    for gi, g in enumerate(split_syllables(phones)):
        if gi:
            out.append("-")
        out.extend(g)
    return out


def join_phone_seqs(parts):
    """多段音素（如复合词/字母串）→ 重新按音节插 '-'"""
    flat = [x for p in parts for x in p if x != "-"]
    return with_syllables(flat)


# ---------------- 数字词 ----------------
def cardinal_words(n):
    if n < 0:
        return ["minus"] + cardinal_words(-n)
    if n < 20:
        return [ONES[n]]
    if n < 100:
        r = n % 10
        return [TENS[n // 10]] + ([ONES[r]] if r else [])
    if n < 1000:
        q, r = divmod(n, 100)
        w = [ONES[q], "hundred"]
        if r:
            w += ["and"] + cardinal_words(r)
        return w
    if n < 1000000:
        q, r = divmod(n, 1000)
        w = cardinal_words(q) + ["thousand"]
        if r:
            w += (["and"] + cardinal_words(r)) if r < 100 else cardinal_words(r)
        return w
    if n < 1000000000:
        q, r = divmod(n, 1000000)
        w = cardinal_words(q) + ["million"]
        if r:
            w += (["and"] + cardinal_words(r)) if r < 100 else cardinal_words(r)
        return w
    q, r = divmod(n, 1000000000)
    w = cardinal_words(q) + ["billion"]
    if r:
        w += cardinal_words(r)
    return w


def ordinal_words(n):
    if n < 20:
        return [ORD_ONES[n]]
    if n < 100:
        if n % 10 == 0:
            return [ORD_TENS[n // 10]]
        return [TENS[n // 10], ORD_ONES[n % 10]]
    if n < 1000:
        q, r = divmod(n, 100)
        w = [ONES[q], "hundred"]
        if r:
            w += (["and"] + ordinal_words(r)) if r < 100 else ordinal_words(r)
        else:
            w[-1] = "hundredth"
        return w
    base = cardinal_words(n)
    if n % 100 == 0:
        return base[:-1] + [base[-1] + "th"]
    return base[:-1] + [ordinal_words(n % 100)[-1]]


def digits_words(ds):
    return [ONES[int(c)] for c in ds if c.isdigit()]


def year_words(n):
    if 1000 <= n < 2000:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return cardinal_words(hi * 100)
        return cardinal_words(hi) + (cardinal_words(lo) if lo >= 10 else ["oh"] + cardinal_words(lo))
    if 2000 <= n < 2100:
        lo = n - 2000
        if lo == 0:
            return ["two", "thousand"]
        if lo < 10:
            return ["two", "thousand", "and"] + cardinal_words(lo)
        return cardinal_words(20) + cardinal_words(lo)
    return cardinal_words(n)


def plural(n, word):
    if word == "yen" or word == "sen":
        return word
    if n == 1:
        return word
    return {"penny": "pence", "cent": "cents", "dollar": "dollars", "euro": "euros",
            "pound": "pounds"}.get(word, word + "s")


# ---------------- 词 → 音素 ----------------
def _engine_phones(word):
    w = word.lower().replace("’", "'")
    # 变体词表（en-GB/en-IN 引擎实测）优先于上下文词表
    if _ACTIVE_LEX is not None:
        v = _ACTIVE_LEX.get(w)
        if v:
            return list(v)
    for src in (load_context(), _active_lexicon(), _VOCAB):
        v = src.get(w) if src else None
        if v:
            return list(v)
    return None


def _load_vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _load_json(os.path.join(ROOT, "data", "lexicon_en.names.json"), {})
    return _VOCAB


_CMU = None


def _cmu_dict():
    global _CMU
    if _CMU is None:
        _CMU = _load_json(os.path.join(ROOT, "data", "cmudict_en.json"), {})
    return _CMU


def _fallback_phones(word):
    w = word.lower()
    cmu = _cmu_dict()
    for key in (w, w.upper()):
        v = cmu.get(key)
        if isinstance(v, str) and v:
            return [cmu_to_engine(p) for p in v.split()]
        if isinstance(v, list) and v:
            return [cmu_to_engine(p) for p in v]
    out = []
    for ch in w:
        if ch.isalpha():
            out.extend(LETTER_RULES.get(ch, []))
    return out


CONTRACTIONS = {
    "don't": ["d", "ow1", "n", "t"], "can't": ["k", "ae1", "n", "t"],
    "won't": ["w", "ow1", "n", "t"], "shan't": ["sh", "ae1", "n", "t"],
    "ain't": ["ey1", "n", "t"],
    "isn't": ["ih1", "z", "ax", "n", "t"], "aren't": ["aa1", "r", "n", "t"],
    "wasn't": ["w", "ah1", "z", "ax", "n", "t"], "weren't": ["w", "er1", "n", "t"],
    "doesn't": ["d", "ah1", "z", "ax", "n", "t"], "didn't": ["d", "ih1", "d", "ax", "n", "t"],
    "couldn't": ["k", "uh1", "d", "ax", "n", "t"], "shouldn't": ["sh", "uh1", "d", "ax", "n", "t"],
    "wouldn't": ["w", "uh1", "d", "ax", "n", "t"], "mustn't": ["m", "ah1", "s", "ax", "n", "t"],
    "needn't": ["n", "iy1", "d", "ax", "n", "t"],
    "it's": ["ih1", "t", "s"], "that's": ["dh", "ae1", "t", "s"],
    "what's": ["w", "ah1", "t", "s"], "let's": ["l", "eh1", "t", "s"],
    "he's": ["h", "iy1", "z"], "she's": ["sh", "iy1", "z"],
    "there's": ["dh", "eh1", "r", "z"], "here's": ["hh", "ih1", "r", "z"],
    "where's": ["w", "eh1", "r", "z"], "who's": ["hh", "uw1", "z"],
    "how's": ["hh", "aw1", "z"],
    "i'm": ["ay1", "m"], "i've": ["ay1", "v"], "i'll": ["ay1", "l"], "i'd": ["ay1", "d"],
    "you're": ["y", "ao1", "r"], "you've": ["y", "uw1", "v"], "you'll": ["y", "uw1", "l"], "you'd": ["y", "uw1", "d"],
    "we're": ["w", "iy1", "r"], "we've": ["w", "iy1", "v"], "we'll": ["w", "iy1", "l"], "we'd": ["w", "iy1", "d"],
    "they're": ["dh", "ey1", "r"], "they've": ["dh", "ey1", "v"], "they'll": ["dh", "ey1", "l"], "they'd": ["dh", "ey1", "d"],
    "he'll": ["hh", "iy1", "l"], "he'd": ["hh", "iy1", "d"],
    "she'll": ["sh", "iy1", "l"], "she'd": ["sh", "iy1", "d"],
    "it'll": ["ih1", "t", "ax", "l"],
    "who'll": ["hh", "uw1", "l"], "who'd": ["hh", "uw1", "d"], "who've": ["hh", "uw1", "v"],
}

VOICELESS = {"p", "t", "k", "f", "th", "s", "sh", "ch", "hh"}


def _contraction_phones(word):
    w = word.replace("’", "'").lower()
    if w in CONTRACTIONS:
        return list(CONTRACTIONS[w])
    if w.endswith("n't") and len(w) > 3:
        base = w[:-3]
        base = {"ca": "can", "wo": "will", "sha": "shall", "ai": "am"}.get(base, base)
        bp = pron_word(base)
        return list(bp) + ["n", "t"]
    if w.endswith("'s") and len(w) > 2:
        base = w[:-2]
        bp = pron_word(base)
        last = (bp[-1].rstrip("012") if bp else "")
        return list(bp) + (["s"] if last in VOICELESS else ["z"])
    for suf, ph in (("'ll", "l"), ("'re", "r"), ("'ve", "v"), ("'d", "d"), ("'m", "m")):
        if w.endswith(suf) and len(w) > len(suf):
            return list(pron_word(w[:-len(suf)])) + [ph]
    return None


def pron_word(word, next_word="", prev_word=""):
    acro = load_acronyms().get(word)
    if acro:
        return list(acro)
    case = load_case_words().get(word)
    if case:
        return list(case)
    if len(word) == 1 and word.isupper() and word != "I":
        nxt = (next_word or "").strip()
        # 句首的 "A"：后接辅音开头的词读冠词弱读 ax；后接元音开头的词读 /eɪ/（引擎实测）
        if word == "A" and nxt and not (len(nxt) == 1 and nxt.isupper()):
            ph = _engine_phones(nxt.split()[0]) if nxt.split() else None
            first = [x for x in (ph or []) if x != "-"]
            is_vowel = bool(first) and first[0].rstrip("012") in VOWELS
            return ["ey1"] if is_vowel else pron_word("a", next_word, prev_word)
        return _letter_token(word.lower())
    w = word.lower().replace("’", "'")
    # 变体词表（en-GB/en-IN 引擎实测）优先于内置规则，但功能词（弱读/上下文读法）
    # 实测是"孤立词 citation 读法"，在句子里不适用，仍交给内置规则。
    if _ACTIVE_LEX is not None and "'" not in w and w not in FUNCTION_WORDS:
        v = _ACTIVE_LEX.get(w)
        if v:
            return list(v)
    if "'" in w:
        cph = _contraction_phones(w)
        if cph:
            return cph
    het = load_heteronyms().get(w)
    if het:
        pv = (prev_word or "").lower()
        if w == "read" and het.get("past"):
            if pv in {"had", "has", "have", "was", "were", "been", "being", "is", "are", "am"}:
                return list(het["past"])
        if w == "use" and prev_word:
            if pv in {"is", "are", "was", "were", "be", "being", "am"} and het.get("noun"):
                return list(het["noun"])
        if w == "content":
            nxt0 = (next_word or "").lower()
            modal = pv in {"will", "would", "can", "could", "should", "must", "may", "might"}
            pron = nxt0 in {"me", "you", "him", "her", "us", "them", "it"}
            if modal and pron and het.get("verb"):
                return list(het["verb"])
            if het.get("noun"):
                return list(het["noun"])
        if pv in VERB_PREV and het.get("verb"):
            return list(het["verb"])
        if pv in NOUN_PREV and het.get("noun"):
            return list(het["noun"])
        if pv in HETERO_NOUN_PREV and het.get("noun"):
            return list(het["noun"])
    if w == "the":
        # en-IN 实测：句子里 the 读 [d, ah]（与孤立词一致）
        if _ACTIVE_VARIANT == "in" and _ACTIVE_LEX is not None:
            v = _ACTIVE_LEX.get("the")
            if v:
                return list(v)
        nxt = (next_word or "").lower()
        nxt_ph = _engine_phones(nxt) if nxt else None
        if nxt_ph:
            first = [x for x in nxt_ph if x != "-"]
            is_vowel = bool(first) and first[0].rstrip("012") in VOWELS
        else:
            is_vowel = nxt[:1] in ("a", "e", "i", "o", "u")
        return ["dh", "ih1"] if is_vowel else ["dh", "ax1"]
    if w in SPECIAL_PHONES:
        return list(SPECIAL_PHONES[w])
    v = _engine_phones(w)
    if v:
        return v
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        out = []
        for p in parts:
            ph = pron_word(p)
            if out:
                out.append("-")
            out.extend(ph)
        return out
    if w.endswith("s") and len(w) > 2:
        base = pron_word(w[:-1])
        if base and len(base) > 1:
            first = base[0]
            voiceless = first.rstrip("012") in ("p", "t", "k", "f", "th")
            return base + (["s"] if voiceless else ["z"])
    return with_syllables(_fallback_phones(w))


# ---------------- TN 展开 ----------------
def _letter_token(letters):
    """U.S.A. / i.e. / ABC → 单个 token，字母之间用 '-'"""
    iso = _active_lexicon()
    names = []
    for k, ch in enumerate(letters):
        if k:
            names.append("-")
        ph = iso.get(ch) or _engine_phones(ch) or LETTER_RULES.get(ch, [])
        names.extend(ph)
    return names


def _expand_date(md=0, dd=0, yy=0):
    out = []
    if not (1 <= md <= 12):
        return None
    out.append(("w", MONTH_NAMES[md - 1]))
    out.append(("w", ordinal_words(dd)[-1] if dd else None))
    if dd:
        out[-1] = ("w", None)  # placeholder replaced below
        out[-1] = ("ords", dd)
    return out


def _date_words(md, dd, yy, year_style="year"):
    words = [MONTH_NAMES[md - 1]] + ordinal_words(dd)
    events = [("w", w) for w in words] + [("pause_plain", None)]
    if year_style == "cardinal":
        events += [("w", w) for w in cardinal_words(yy)]
    else:
        events += [("w", w) for w in year_words(yy)]
    return events


def _time_words(hh, mm, ampm):
    events = [("w", w) for w in cardinal_words(hh)]
    if mm:
        events += [("w", w) for w in cardinal_words(mm)]
    if ampm:
        ap = "a" if ampm.lower().startswith("a") else "p"
        events += [("phones", _letter_token(ap)), ("phones", _letter_token("m"))]
    elif mm == 0:
        events += [("phones", SPECIAL_PHONES["o'clock"])]
    return events


def _currency_words(sym, numstr):
    major, minor = CURRENCY[sym]
    if "." in numstr:
        a, b = numstr.split(".")
        a = a.replace(",", "")
        major_n = int(a) if a else 0
        cents = int((b + "00")[:2])
        words = (cardinal_words(major_n) if major_n or not cents else []) + \
                ([plural(major_n, major)] if major_n else [])
        if cents:
            if major_n:
                words += ["and"]
            words += cardinal_words(cents) + [plural(cents, minor)]
        events = [("w", w) for w in words]
        return events
    n = int(numstr.replace(",", ""))
    if sym == "¥" and n:  # U+00A5 半角日元符号 → yen
        return [("w", w) for w in cardinal_words(n) + ["yen"]]
    if sym == "￥" and n:  # U+FFE5 全角符号 → 原版读 yuan
        return [("w", w) for w in cardinal_words(n)] + [("phones", ["y", "uw", "-", "aa1", "n"])]
    return [("w", w) for w in cardinal_words(n) + [plural(n, major)]]


def _phone_words(pattern, text):
    groups = []
    letters = []
    clean = re.sub(r"[()]", " ", text)
    for part in re.split(r"[-.\s]+", clean.strip()):
        part = part.strip().lstrip("+")
        if not part:
            continue
        digits_only = re.sub(r"\D", "", part)
        if digits_only:
            groups.append(digits_only)
        else:
            letters.append(part)
    events = []
    for gi, g in enumerate(groups):
        if gi:
            events.append(("pause_plain", None))
        if len(g) == 1:
            events += [("w", w) for w in digits_words(g)]
        elif len(g) == 3 and g[0] >= "8" and g[1:] == "00":
            events += [("w", w) for w in ([ONES[int(g[0])], "hundred"])]
        elif len(g) == 4:
            events += [("w", w) for w in digits_words(g)]
        else:
            events += [("w", w) for w in digits_words(g)]
    for part in letters:
        ev = _letter_token(part.lower())
        if ev:
            events.append(("phones", ev))
    return events


DENOM = {2: "half", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
         8: "eighth", 9: "ninth", 10: "tenth", 100: "hundredth"}


def _fraction_words(num, den):
    if den == 2 and num == 1:
        return ["one", "half"]
    d = DENOM.get(den, ordinal_words(den)[-1])
    if num == den:
        return cardinal_words(num) + [d + "s"]
    if num == 1:
        return ["one", d]
    return cardinal_words(num) + [d + "s"]


def _unit_words(unit):
    u = unit.lower().strip(".")
    if u in UNITS:
        return UNITS[u]
    return None


# ---------------- 解析器 ----------------
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:['’][A-Za-z]+)*(?:-[A-Za-z0-9]+)*")
DOT_ABBR_RE = re.compile(r"(?:[A-Za-z]\.){2,}")
NUM_PATTERNS = [
    ("phone", re.compile(r"\+\d{1,3}(?:[-.]\d{2,4}){2,}")),
    ("phone", re.compile(r"\(\d{3}\)\s*\d{3}[-.]\d{4}")),
    ("phone", re.compile(r"\d{3}[-.]\d{3}[-.]\d{4}")),
    ("phone", re.compile(r"\d{3}[-.]\d{4}(?![\d])")),
    ("ip", re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")),
    ("date", re.compile(r"\d{4}[./-]\d{1,2}[./-]\d{1,4}")),
    ("date2", re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")),
    ("date", re.compile(r"\d{4}-\d{1,2}-\d{1,2}")),
    ("date", re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")),
    ("time_range", re.compile(r"\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}")),
    ("time", re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm])?")),
    ("time", re.compile(r"\d{1,2}\s?(?:[AaPp]\.?[Mm])(?![A-Za-z])")),
    ("alnum", re.compile(r"\d{2,4}-[A-Za-z]{2,4}-\d{2,4}")),
    ("ratio", re.compile(r"\d{1,2}:\d{1}")),
    ("version", re.compile(r"\d+\.\d+(?:\.\d+)+")),
    ("ordinal", re.compile(r"\d+(?:st|nd|rd|th)")),
    ("percent", re.compile(r"\d+(?:\.\d+)?%")),
    ("decimal", re.compile(r"\d*\.\d+")),
    ("ordinal", re.compile(r"\d{1,3}(?:,\d{3})+(?:st|nd|rd|th)")),
    ("thousands", re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")),
    ("dashgroup", re.compile(r"\d{2}(?:-[0-9A-Za-z]{2}){2,}")),
    ("range", re.compile(r"\d+(?:-\d+)+")),
    ("mixed", re.compile(r"\d+\s+\d+\s*/\s*\d+")),
    ("date_ym", re.compile(r"\d{4}/\d{1,2}")),
    ("date_my", re.compile(r"\d{1,2}/\d{4}")),
    ("fraction", re.compile(r"\d+\s*/\s*\d+")),
    ("currency", re.compile(r"[$€£¥￥]\s*\d[\d,]*(?:\.\d+)?")),
    ("number", re.compile(r"\d+")),
]


def _match_number(text, i):
    for kind, pat in NUM_PATTERNS:
        m = pat.match(text, i)
        if m:
            return kind, m.group(0), m.end()
    return None, None, i


def _parse(text):
    events = []
    i, n = 0, len(text)
    last_word = ""
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        # http(s):// → 逐字母读协议名 + colon slash slash
        m_url = re.match(r"(?i)(https?)://", text[i:])
        if m_url:
            scheme = m_url.group(1).lower()
            for ch2 in scheme:
                events.append(("phones", _letter_token(ch2)))
            events.append(("w", "colon"))
            events.append(("w", "slash"))
            events.append(("w", "slash"))
            i += m_url.end()
            continue
        # mailto: → 单词 mailto（不读冒号）
        m_mail = re.match(r"(?i)mailto:", text[i:])
        if m_mail:
            events.append(("word", "mailto"))
            i += m_mail.end()
            continue
        # 数字类
        kind, tok, j = _match_number(text, i)
        if kind:
            if kind == "currency":
                sym = tok[0]
                num = tok[1:].strip()
                events.append(("currency", (sym, num)))
            elif kind in ("date", "date2"):
                events.append(("date", tok))
            elif kind == "date_my":
                events.append(("date_my", tok))
            elif kind == "date_ym":
                events.append(("date_ym", tok))
            elif kind == "time":
                events.append(("time", tok))
            elif kind == "time_range":
                events.append(("time_range", tok))
            elif kind == "ratio":
                events.append(("ratio", tok))
            elif kind == "percent":
                num = tok[:-1]
                events.append(("decimal" if "." in num else "number", num))
                events.append(("percent", None))
            elif kind == "ordinal":
                events.append(("ordinal", tok))
            elif kind == "phone":
                events.append(("phone", tok))
            elif kind == "ip":
                events.append(("ip", tok))
            elif kind == "dashgroup":
                events.append(("dashgroup", tok))
            elif kind == "alnum":
                events.append(("alnum", tok))
            elif kind == "groupnum":
                events.append(("groupnum", tok))
            elif kind == "fraction":
                events.append(("fraction", tok))
            elif kind == "mixed":
                events.append(("mixed", tok))
            elif kind in ("decimal", "thousands", "version", "range"):
                events.append((kind, tok))
            else:
                events.append(("number", tok))
            i = j
            continue
        # 版本号 vX.Y[.Z][-suffix]：原版读 "vee two point one"（字母 v + 版本数字）
        m_ver = re.match(r"(?i)v(\d+(?:\.\d+)+)(?:[-–]([A-Za-z0-9]+))?", text[i:])
        if m_ver:
            events.append(("phones", _letter_token("v")))
            num = m_ver.group(1)
            events.append(("version" if num.count(".") >= 2 else "decimal", num))
            if m_ver.group(2):
                events.append(("word", m_ver.group(2)))
            i += m_ver.end()
            continue
        # 点分缩写（U.S.A. / i.e. / e.g.）
        m = DOT_ABBR_RE.match(text, i)
        if m:
            events.append(("dotted", m.group(0)))
            i = m.end()
            continue
        if ch.isalpha():
            m = WORD_RE.match(text, i)
            if m:
                w = m.group(0)
                i = m.end()
                # report_2024：下划线分段
                if "_" in w and any(seg.isdigit() for seg in w.split("_") if seg):
                    segs = [x for x in w.split("_") if x]
                    for si, seg in enumerate(segs):
                        if si:
                            events.append(("w", "underscore"))
                        m_a = re.fullmatch(r"([A-Za-z]+)(\d+)", seg)
                        m_d = re.fullmatch(r"(\d+)([A-Za-z]+)", seg)
                        if seg.isdigit():
                            events.append(("number", seg))
                        elif m_a:
                            letters = m_a.group(1)
                            if letters.isupper() and len(letters) > 1:
                                events.append(("acronym", letters))
                            elif _engine_phones(letters.lower()):
                                events.append(("word", letters))
                            else:
                                events.append(("letters", letters))
                            events.append(("number", m_a.group(2)))
                        elif m_d:
                            events.append(("number", m_d.group(1)))
                            letters = m_d.group(2)
                            if letters.isupper() and len(letters) > 1:
                                events.append(("acronym", letters))
                            else:
                                events.append(("letters", letters))
                        else:
                            events.append(("word", seg))
                    if i < n and text[i] == "." and i + 1 < n and text[i + 1].isalnum():
                        events.append(("w", "dot"))
                        i += 1
                    last_word = w.lower()
                    continue
                # B2B / C3PO：字母 + 数字 + 字母
                m_an3 = re.fullmatch(r"([A-Za-z]+)(\d+)([A-Za-z]+)", w)
                if m_an3:
                    def _emit_group_letters(g):
                        if g.isupper() and len(g) > 1:
                            events.append(("acronym", g))
                        elif _engine_phones(g.lower()):
                            events.append(("word", g))
                        else:
                            events.append(("letters", g))
                    _emit_group_letters(m_an3.group(1))
                    events.append(("number", m_an3.group(2)))
                    _emit_group_letters(m_an3.group(3))
                    last_word = w.lower()
                    continue
                if w.lower() == "no" and i < n and text[i] == ".":
                    events.append(("w", "number"))
                    i += 1
                    last_word = "number"
                    continue
                # HTML5 / AA123：字母串 + 数字
                m_an = re.fullmatch(r"([A-Za-z]+)(\d+)", w)
                if m_an:
                    letters = m_an.group(1)
                    if _engine_phones(letters.lower()) or letters.lower() in SPECIAL_PHONES or not letters.isupper():
                        # 已知读音的字母串/大小写混合词：按词读（MP3→"MP three", iPhone15→"iPhone fifteen"）
                        events.append(("word", letters))
                    elif letters.isupper() and len(letters) > 1:
                        events.append(("acronym", letters))
                    else:
                        events.append(("letters", letters))
                    if len(m_an.group(2)) == 3:
                        events.append(("groupnum", m_an.group(2)))
                    else:
                        events.append(("number", m_an.group(2)))
                    last_word = w.lower()
                    continue
                # 文件名/域名：word "dot" word
                if i < n and text[i] == "." and i + 1 < n and text[i + 1].isalnum():
                    events.append(("word", w))
                    events.append(("w", "dot"))
                    i += 1
                    last_word = "dot"
                    continue
                # 单大写字母 + 符号（A & B）→ 字母读音
                if len(w) == 1 and w.isupper():
                    j2 = i
                    while j2 < n and text[j2] == " ":
                        j2 += 1
                    if j2 < n and not (text[j2].isalnum() or text[j2] == "."):
                        events.append(("letters", w))
                        last_word = w.lower()
                        continue
                if i < n and text[i] == ".":
                    low = w.lower()
                    if low == "no":
                        events.append(("w", "number"))
                        i += 1
                        last_word = "number"
                        continue
                    if low in ("mr","mrs","ms","dr","st","prof","jr","sr","vs","etc","capt","col","no","inc","ltd","co","mt","saint"):
                        events.append(("word", w))
                        i += 1
                        if i >= n or not text[i:].strip():
                            events.append(("end", "punc."))
                        last_word = low
                        continue
                    if low in CNT_WORDS:
                        events.append(("word", w))
                        last_word = low
                        continue
                    if len(w) == 1 and w.isupper():
                        if last_word in CNT_WORDS:
                            events.append(("word", w))
                            last_word = low
                            continue
                        events.append(("letters", w))
                        if i + 2 < n and text[i + 1] in ",;":
                            events.append(("pause", "punc."))
                            i += 2
                            last_word = low
                            continue
                        if i + 1 >= n or text[i + 1] == " ":
                            events.append(("end", "punc."))
                        i += 1
                        last_word = low
                        continue
                    events.append(("word", w))
                    last_word = low
                    continue
                events.append(("word", w))
                last_word = w.lower()
                continue
        # 连字符：与前后的词合并为复合词（引擎把 --/--- 也当作词内连接）
        if (ch in "-–—" and events and events[-1][0] == "word"
                and i > 0 and text[i - 1] not in " 	"):
            k = i
            while k < n and text[k] in "-–—":
                k += 1
            m2 = WORD_RE.match(text, k)
            if m2:
                prev = events[-1][1]
                events[-1] = ("word", prev + "-" + m2.group(0))
                i = m2.end()
                continue
            i = k
            continue
        # 负号
        if ch == "-" and i + 1 < n and text[i + 1].isdigit() and (i == 0 or text[i - 1].isspace()):
            events.append(("minus", None))
            i += 1
            continue
        if ch == "…":
            rest = text[i + 1:].strip()
            events.append(("ellipsis_pause" if rest else "end_ellipsis", "punc…"))
            i += 1
            continue
        if ch in END_PUNCT:
            if ch == ".":
                run = 0
                while i + run < n and text[i + run] == ".":
                    run += 1
                if run >= 2:
                    rest = text[i + run:].strip()
                    if run >= 3:
                        events.append(("ellipsis_pause" if rest else "end_ellipsis", None))
                    else:
                        events.append(("end", "punc."))
                        events.append(("punc_after_end", "punc."))
                    i += run
                    continue
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt and not nxt.isspace() and nxt not in ")]":
                    if nxt.isalnum():
                        i += 1
                        continue
                    if nxt == "_":
                        events.append(("pause_br1", "punc."))
                    else:
                        events.append(("pause", "punc."))
                    i += 1
                    continue
            events.append(("end", END_PUNCT[ch]))
            if i + 1 < n and text[i + 1] in BQUOTE:
                events.append(("quote_after_end", "punc."))
                i += 2
                continue
            i += 1
            continue
        if ch in ")]":
            j2 = i + 1
            while j2 < n and text[j2] in ")]":
                j2 += 1
            if j2 < n and text[j2] == ":":
                i = j2 + 1
                continue
            if events and events[-1][0] in ("end", "punc_after_end"):
                events.append(("punc_after_end", "punc."))
                i = j2
                continue
            rest = text[j2:].lstrip()
            if not rest or rest[0] in ".!?":
                events.append(("end", "punc."))
                i = j2
                continue
        if ch in PAUSE_PUNCT:
            if ch == ":" and i + 1 < n and text[i + 1] in "-–—":
                events.append(("br1_br0", None))
                i += 1
                continue
            if ch == ":":
                j3 = i + 1
                while j3 < n and text[j3] == " ":
                    j3 += 1
                if j3 < n and text[j3] == "[":
                    events.append(("pause", "punc,"))
                    i += 1
                    continue
            if ch == "," and i + 1 < n and text[i + 1] in "-–—":
                events.append(("pause_br1", "punc,"))
                i += 1
                continue
            if ch == ":" and i + 1 < n and text[i + 1] in "\\/":
                events.append(("pause_plain", None))
            else:
                events.append(("pause", PAUSE_PUNCT[ch]))
            i += 1
            continue
        if ch == "&":
            events.append(("phones", STRONG_AND))
            i += 1
            continue
        if ch in ("×",):
            events.append(("w", "times"))
            i += 1
            continue
        if ch in ("÷",):
            events.append(("phones", ["d", "ih", "-", "v", "ay1", "-", "d", "ih", "d", "-", "b", "ay2"]))
            i += 1
            continue
        if ch in "–—":
            prev_kind = events[-1][0] if events else None
            jj = i + 1
            while jj < n and text[jj] == " ":
                jj += 1
            if prev_kind in ("number", "decimal", "percent", "thousands", "currency") and jj < n and text[jj].isdigit():
                events.append(("w", "to"))
            else:
                events.append(("pause", "punc,"))
            i += 1
            continue
        if ch == "/":
            # 单字母/单字母（A/B、I/O）：原版直接按两个字母读，不读 "slash"
            if (i > 0 and i + 1 < n and text[i - 1].isupper() and text[i + 1].isupper()
                    and (i < 2 or not text[i - 2].isalpha())
                    and (i + 2 >= n or not text[i + 2].isalpha())):
                i += 1
                continue
            events.append(("slash", None))
            i += 1
            continue
        if ch == "@":
            events.append(("w", "at"))
            i += 1
            continue
        if ch == "\\":
            i += 1
            continue
        if ch == "'":
            i += 1
            continue
        if ch == '"':
            events.append(("br2", None))
            i += 1
            continue
        if ch == "=":
            events.append(("phones", ["iy1", "-", "k", "w", "ax", "l", "z"]))
            i += 1
            continue
        if ch in SYMBOL_WORDS:
            word = SYMBOL_WORDS[ch]
            if word:
                events.append(("w", word))
            i += 1
            continue
        i += 1
    return events


# ---------------- 展开为 word/pause/phones 序列 ----------------
def _expand_number_token(tok, kind, next_kind):
    t = tok.replace(",", "")
    low = tok.lower()
    if kind == "time_range":
        parts = re.split(r"\s*[-–—]\s*", tok)
        def _one(tp):
            hh, mm = int(tp.split(":")[0]), int(tp.split(":")[1])
            return hh, mm
        h1, m1 = _one(parts[0])
        h2, m2 = _one(parts[1])
        ev = []
        if m1 == 0 and m2 == 0:
            ev += [("w", w) for w in cardinal_words(h1)]
            ev.append(("w", "to"))
            ev += [("w", w) for w in cardinal_words(h2)]
            ev.append(("w", "o'clock"))
        else:
            ev += _time_words(h1, m1, None)
            ev.append(("w", "to"))
            ev += _time_words(h2, m2, None)
        return ev
    if kind == "date_ym":
        a, b = t.split("/")
        return [("w", w) for w in year_words(int(a))] + [("w", "slash")] + [("w", w) for w in digits_words(b.zfill(2))]
    if kind == "date_my":
        a, b = t.split("/")
        return [("w", MONTH_NAMES[int(a) - 1])] + [("w", w) for w in year_words(int(b))]
    if kind == "date":
        sep = re.search(r"[/.-]", t).group(0)
        a, b, c = re.split(r"[/.-]", t)
        ai, bi, ci = int(a), int(b), int(c)
        if len(a) == 4:
            mo, d, y = bi, ci, ai
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return _date_words(mo, d, y, "year")
        elif 1 <= ai <= 12 and 1 <= bi <= 31:
            mo, d, y = ai, bi, ci
            return _date_words(mo, d, y, "cardinal" if len(c) <= 2 else "year")
        sep_word = {"/": "slash", "-": "dash", ".": "dot"}.get(sep, "slash")
        ev = [("w", w) for w in cardinal_words(ai)]
        ev.append(("w", sep_word))
        ev += [("w", w) for w in cardinal_words(bi)]
        ev.append(("w", sep_word))
        if len(c) == 4:
            ev += [("w", w) for w in year_words(ci)]
        else:
            ev += [("w", w) for w in cardinal_words(ci)]
        return ev
    if kind == "time":
        m = re.match(r"(\d{1,2})(?::(\d{2}))(?::(\d{2}))?(?:\s?([AaPp])\.?[Mm])?", tok) or             re.match(r"(\d{1,2})\s?(?:([AaPp])\.?[Mm])", tok)
        if m and m.lastindex and m.lastindex <= 2 and m.group(2) and m.group(2).isdigit() is False:
            hh2 = int(m.group(1))
            ampm2 = (m.group(2) or "").lower()
            return _time_words(hh2, 0, ampm2)
        hh = int(m.group(1))
        hh = int(m.group(1))
        mm = int(m.group(2))
        raw_ap = (m.group(4) or "").lower()
        ampm = ("am" if raw_ap.startswith("a") else "pm") if raw_ap else None
        if m.group(3):
            ss = int(m.group(3))
            ev = [("w", w) for w in cardinal_words(hh)] + [("w", "hour" if hh == 1 else "hours")]
            ev += [("w", w) for w in cardinal_words(mm)] + [("w", "minute" if mm == 1 else "minutes")]
            ev += [("w", "and")] + [("w", w) for w in cardinal_words(ss)] + [("w", "second" if ss == 1 else "seconds")]
            return ev
        return _time_words(hh, mm, ampm)
    if kind == "ratio":
        hh, mm = t.split(":")
        return [("w", w) for w in cardinal_words(int(hh)) + ["to"] + cardinal_words(int(mm))]
    if kind == "version":
        parts = t.split(".")
        ev = [("w", w) for w in cardinal_words(int(parts[0]))]
        for p in parts[1:]:
            ev.append(("w", "point"))
            ev += [("w", w) for w in digits_words(p)]
        return ev
    if kind == "ordinal":
        digits = re.sub(r"\D", "", tok)
        return [("w", w) for w in ordinal_words(int(digits))]
    if kind == "decimal":
        if t.startswith("."):
            a, b = "", t[1:]
        else:
            a, b = t.split(".")
        ev = []
        if a:
            ev += [("w", w) for w in cardinal_words(int(a))]
        ev.append(("w", "point"))
        ev += [("w", w) for w in digits_words(b)]
        return ev
    if kind == "currency":
        return None
    # number / thousands / range / phone
    if kind == "range":
        parts = [p for p in t.split("-") if p]
        sep_word = "to" if len(parts) == 2 else "dash"
        ev = []
        for pi, part in enumerate(parts):
            if pi:
                ev.append(("w", sep_word))
            if len(part) > 1 and part.startswith("0"):
                ev += [("w", w) for w in digits_words(part)]
            else:
                ev += [("w", w) for w in cardinal_words(int(part))]
        return ev
    if kind == "phone":
        return _phone_words(tok, tok)
    if kind == "ip":
        parts = t.split(".")
        ev = []
        for pi, part in enumerate(parts):
            if pi:
                ev.append(("w", "dot"))
            ev += [("w", w) for w in digits_words(part)]
        return ev
    if kind == "dashgroup":
        ev = []
        first = True
        for part in [x for x in re.split(r"-", tok) if x]:
            for ch2 in part:
                if first and ch2.isdigit():
                    ev.append(("w", ONES[int(ch2)]))
                elif first:
                    ev.append(("phones", _letter_token(ch2.lower())))
                else:
                    pass
            if first:
                ev.append(("w", "dash"))
                first = False
        # dash 后所有字符逐个读
        rest = "".join(x for x in re.split(r"-", tok) if x)[0:]
        tail = "".join([x for x in re.split(r"-", tok) if x])[len(re.split(r"-", tok)[0]):]
        ev = []
        firstpart = [x for x in re.split(r"-", tok) if x][0]
        for ch2 in firstpart:
            ev.append(("w", ONES[int(ch2)]) if ch2.isdigit() else ("phones", _letter_token(ch2.lower())))
        ev.append(("w", "dash"))
        for ch2 in tail:
            ev.append(("w", ONES[int(ch2)]) if ch2.isdigit() else ("phones", _letter_token(ch2.lower())))
        return ev
    if kind == "alnum":
        parts = tok.split("-")
        out = []
        for p in parts:
            if p.isdigit():
                if len(p) == 3:
                    n = int(p)
                    out.append(("w", ONES[int(p[0])]))
                    out += [("w", w) for w in cardinal_words(n % 100)] if n % 100 >= 10 else []
                else:
                    out += [("w", w) for w in digits_words(p)]
            else:
                if p.isupper() and len(p) > 1:
                    out.append(("phones", acronym_phones(p)))
                else:
                    out.append(("phones", _letter_token(p.lower())))
        return out
    if kind == "groupnum":
        n = int(tok)
        first = ONES[int(tok[0])]
        rest = n % 100
        if rest >= 10:
            return [("w", first)] + [("w", w) for w in cardinal_words(rest)]
        return [("w", w) for w in cardinal_words(n)]
    if kind == "fraction":
        a, b = re.split(r"\s*/\s*", tok)
        return [("w", w) for w in _fraction_words(int(a), int(b))]
    if kind == "mixed":
        m = re.match(r"(\d+)\s+(\d+)\s*/\s*(\d+)", tok)
        n, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a == 1 and b == 2:
            return [("w", w) for w in cardinal_words(n) + ["and", "a", "half"]]
        return [("w", w) for w in cardinal_words(n) + ["and"] + _fraction_words(a, b)]
    nn = int(t)
    if len(t) == 4 and 2000 <= nn <= 2099 and kind in ("number", "thousands"):
        return [("w", w) for w in year_words(nn)]
    if kind == "thousands" or len(t) > 3:
        return [("w", w) for w in cardinal_words(nn)]
    return [("w", w) for w in cardinal_words(nn)]


def _expand(events):
    out = []
    for idx, (kind, val) in enumerate(events):
        nxt = events[idx + 1] if idx + 1 < len(events) else None
        # 月份词 + 日：转序数；紧随逗号 → 日期分隔
        if kind == "ordinal" and nxt and nxt[0] == "word" and nxt[1].lower() in MONTH_NAMES:
            out.append(("w", nxt[1]))
            m = re.match(r"(\d+)", val)
            out.extend([("w", w) for w in ordinal_words(int(m.group(1)))])
            events[idx + 1] = ("consumed", None)
            continue
        if kind == "word" and val.lower() in MONTH_NAMES and nxt and nxt[0] in ("number", "ordinal"):
            try:
                day = int(re.match(r"\d+", nxt[1]).group(0))
            except Exception:
                day = 0
            if 1 <= day <= 31:
                out.append(("w", val))
                out.extend([("w", w) for w in ordinal_words(day)])
                events[idx + 1] = ("consumed", None)
                nxt2 = events[idx + 2] if idx + 2 < len(events) else None
                if nxt2 and nxt2[0] == "pause" and nxt2[1] == "punc,":
                    events[idx + 2] = ("pause_plain", None)
                continue
        if kind == "number" and val.isdigit() and len(val) >= 8:
            out.extend([("w", w) for w in digits_words(val)])
            continue
        if kind == "number" and len(val) > 1 and val.startswith("0") and val.isdigit():
            out.extend([("w", w) for w in digits_words(val)])
            continue
        if kind == "number" and len(val.replace(",", "")) == 4 and 1000 <= int(val.replace(",", "")) <= 2099:
            prev_words = [x[1] for x in out if x[0] in ("w", "word")][-2:]
            if "year" in prev_words:
                out.extend([("w", w) for w in year_words(int(val.replace(",", "")))])
                continue
        if kind in ("number", "date", "date_my", "date_ym", "time", "time_range", "ratio", "version", "ordinal",
                    "decimal", "thousands", "range", "phone", "alnum", "fraction",
                    "mixed", "ip", "dashgroup", "groupnum"):
            ev = _expand_number_token(val, kind, nxt[0] if nxt else None)
            out.extend(ev)
            # 单位：数字后紧跟单位缩写
            if nxt and nxt[0] == "word":
                raw_u = nxt[1]
                uw = _unit_words(raw_u)
                prev_word = ""
                for kk in range(len(out) - 1, -1, -1):
                    if out[kk][0] == "w":
                        prev_word = out[kk][1]
                        break
                if nxt[1].lower() == "m":
                    prev_words = [x[1] for x in out if x[0] == "w"]
                    ctx = " ".join(prev_words[-4:])
                    if "hour" in ctx or "hours" in ctx:
                        uw = ["minutes"]
                    else:
                        uw = None
                nxt3 = events[idx + 2] if idx + 2 < len(events) else None
                nxt4 = events[idx + 3] if idx + 3 < len(events) else None
                if (raw_u.lower() == "m" and nxt3 and nxt3[0] == "slash"
                        and nxt4 and nxt4[0] == "word"):
                    out.append(("phones", _letter_token("m")))
                    events[idx + 1] = ("consumed", None)
                    continue
                if (raw_u.lower() == "km" and nxt3 and nxt3[0] == "slash"
                        and nxt4 and nxt4[0] == "word" and nxt4[1].lower() == "h"):
                    out.extend([("w", "kilometers"), ("w", "per"), ("w", "hour")])
                    events[idx + 1] = ("consumed", None)
                    events[idx + 2] = ("consumed", None)
                    events[idx + 3] = ("consumed", None)
                    continue
                if raw_u.isupper() and len(raw_u) == 1:
                    uw = None  # 5G / 4K：单个大写字母读字母；MB 等多字母缩写仍按单位展开

                if uw and kind in ("number", "decimal", "thousands", "version"):
                    last_num = None
                    try:
                        last_num = float(re.sub(r"[^0-9.]", "", val))
                    except Exception:
                        last_num = None
                    words = list(uw)
                    if last_num == 1 and words and words[0] in UNIT_SINGULAR:
                        words[0] = UNIT_SINGULAR[words[0]]
                    out.extend([("w", w) for w in words])
                    events[idx + 1] = ("consumed", None)
        elif kind == "ordinal":
            digits = re.sub(r"\D", "", val)
            out.extend([("w", w) for w in ordinal_words(int(digits))])
        elif kind == "currency":
            sym, num = val
            if sym.lower() in CURRENCY:
                out.extend(_currency_words(sym, num))
            else:
                out.extend(_currency_words("$", num))
        elif kind == "percent":
            if out:
                out.append(("w", "percent"))
        elif kind in ("word",):
            if val.lower() in MONTH_ABBR:
                month_full = MONTH_ABBR[val.lower()]
                day = None
                if nxt and nxt[0] in ("number", "ordinal"):
                    try:
                        d = int(re.match(r"\d+", nxt[1]).group(0))
                    except Exception:
                        d = 0
                    if 1 <= d <= 31:
                        day = d
                        events[idx + 1] = ("consumed", None)
                out.append(("w", month_full))
                if day is not None:
                    out.extend([("w", w) for w in ordinal_words(day)])
                continue
            if val.lower() == "www":
                for ch2 in "www":
                    out.append(("phones", _letter_token(ch2)))
            else:
                out.append(("word", val))
        elif kind == "letters":
            out.append(("phones", _letter_token(val.lower())))
        elif kind == "acronym":
            out.append(("phones", acronym_phones(val)))
        elif kind == "dotted":
            letters = [c for c in val.lower() if c.isalpha()]
            out.append(("phones", _letter_token("".join(letters))))
        elif kind == "minus":
            out.append(("w", "minus"))
        elif kind == "slash":
            out.append(("w", "slash"))
        elif kind == "word_consumed":
            pass
        elif kind == "w":
            out.append(("w", val))
        elif kind == "phones":
            out.append(("phones", list(val)))
        elif kind == "consword":
            out.append(("word", val))
        elif kind == "pause_plain":
            out.append(("pause_plain", None))
        elif kind == "pause_br1":
            out.append(("pause_br1", val))
        elif kind == "br1_br0":
            out.append(("br1_br0", None))
        elif kind == "br2":
            out.append(("br2", None))
        elif kind in ("ellipsis_pause", "end_ellipsis"):
            out.append((kind, val))
        elif kind == "pause":
            out.append(("pause", val))
        elif kind == "end":
            out.append(("end", val))
        elif kind == "punc_after_end":
            out.append(("punc_after_end", val))
        elif kind == "quote_after_end":
            out.append(("quote_after_end", val))
        elif kind == "consumed":
            pass
    return out


def _flatten(events):
    """→ [('w', word) | ('pause', punc|None) | ('pause_plain', None) | ('end', punc) | ('phones', names)]"""
    flat = []
    prev_word = ""
    prev_was_name = False
    for kind, val in events:
        if kind in ("pause", "pause_plain", "end", "br2", "pause_br1"):
            prev_word = ""
            prev_was_name = False
            flat.append((kind, val))
            continue
        if kind == "word":
            w = val
            exp = _word_expansion(w, prev_word)
            if (_is_roman(w) and len(w) > 1 and prev_was_name
                    and prev_word not in CNT_WORDS):
                n = _roman_value(w)
                if 0 < n < 4000:
                    ordw = ordinal_words(n)[-1]
                    ph = ["dh", "ih1", "-"] + [x for x in pron_word(ordw) if x != "-"]
                    flat.append(("phones", ph))
                    prev_word = w.lower()
                    prev_was_name = False
                    continue
            for x in exp:
                flat.append(("w", x))
            prev_word = w.lower()
            prev_was_name = bool(w[:1].isupper())
            continue
        elif kind == "w":
            flat.append(("w", val))
        elif kind == "pause_plain":
            flat.append(("pause_plain", None))
        else:
            flat.append((kind, val))
    return flat


def _roman_value(w):
    n = 0
    prev = 0
    for ch in reversed(w):
        v = ROMAN[ch]
        n += -v if v < prev else v
        prev = max(prev, v)
    return n


ROMAN_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _is_roman(w):
    return bool(w) and bool(ROMAN_RE.fullmatch(w)) and w != "I"


def _word_expansion(w, prev_word=""):
    if w == "I" and prev_word in {"chapter", "part", "section", "book", "volume",
                                  "act", "scene", "article", "appendix"}:
        return cardinal_words(1)
    if _is_roman(w):
        n = _roman_value(w)
        if len(w) == 1 and prev_word not in CNT_WORDS:
            return [w]
        if 0 < n < 4000:
            return cardinal_words(n)
    return [w]


def _preprocess(text):
    # 温度/特殊单位
    text = re.sub(r"°\s*C", " degrees celsius", text, flags=re.I)
    text = re.sub(r"°\s*F", " degrees fahrenheit", text, flags=re.I)
    text = re.sub(r"°", " degrees", text)
    # 英制身高 6'2"
    def _ft(m):
        ft, inch = m.group(1), m.group(2)
        out = ft + " feet"
        if inch:
            out += " " + inch + " inches"
        return out
    text = re.sub(r"(\d+)\s*'\s*(\d+)?\s*\"", _ft, text)
    return text


def _push(ids, v):
    """只追加存在的符号（部分语音包没有 -bos-/br0/br4 等结构符号）"""
    if v is not None:
        ids.append(v)


def _text_to_ids_impl(text, sym, lexicon=None, variant=""):
    text = _preprocess(text)
    raw = _parse(text)
    # 日期上下文：month word + number
    ev = _expand(raw)
    flat = _flatten(ev)
    ids = [x for x in (sym.get(BOS), sym.get("br0")) if x is not None]
    first = True
    ended = False
    # 预处理：字符串词后的单位已经合并；处理 "month day" 日期
    i = 0
    while i < len(flat):
        item = flat[i]
        kind = item[0]
        if kind == "w":
            w = item[1]
            nxt = ""
            j = i + 1
            while j < len(flat) and flat[j][0] not in ("w",):
                j += 1
            if j < len(flat):
                nxt = flat[j][1]
            prev_w = flat[i - 1][1] if i > 0 and flat[i - 1][0] == "w" else ""
            if prev_w.lower() in ("not", "never", "always", "only", "also", "just", "still"):
                j2 = i - 2
                while j2 >= 0 and flat[j2][0] != "w":
                    j2 -= 1
                if j2 >= 0:
                    prev_w = flat[j2][1]
            phones = pron_word(w, nxt, prev_w)
            wl = w.lower()
            after_kind = flat[i + 1][0] if i + 1 < len(flat) else "end"
            after_w = flat[i + 1][1] if i + 1 < len(flat) and flat[i + 1][0] == "w" else ""
            if wl == "for":
                prev_pause = first or (i > 0 and flat[i - 1][0] in ("pause", "pause_plain"))
                if (prev_pause or after_w in ("underscore", "asterisk") or after_kind == "br2"
                        or after_kind in ("end", "pause", "pause_plain")):
                    phones = list(STRONG_FOR)
            elif wl == "and":
                if (after_kind in ("pause", "pause_plain", "end", "br2")
                        or after_w in ("underscore", "asterisk")):
                    phones = list(STRONG_AND)
            need_br1 = (not first) and not (i > 0 and flat[i - 1][0] == "br2")
            if need_br1:
                _push(ids, sym.get("br1"))
            for ph in phones:
                if ph == "-":
                    _push(ids, sym.get("-") if sym.get("-") is not None else sym.get("&"))
                else:
                    ids.append(_sym_id(sym, ph, variant))
            first = False
        elif kind == "ellipsis_pause":
            ids.append(sym.get("br3", 0))
            ids.append(sym.get(item[1] or "punc...", sym.get("punc.", 0)))
            ids.append(sym.get("br0", 0))
            first = True
        elif kind == "end_ellipsis":
            _push(ids, sym.get("br4"))
            ids.append(sym.get(item[1] or "punc...", sym.get("punc.", 0)))
            ids.append(sym.get("punc.", 0))
            _push(ids, sym.get(END))
            ended = True
            first = True
        elif kind == "pause_plain":
            _push(ids, sym.get("br3"))
            first = True
        elif kind == "pause_br1":
            _push(ids, sym.get("br1"))
            ids.append(sym.get(item[1], sym.get("punc.", 0)))
            _push(ids, sym.get("br0"))
            first = True
        elif kind == "br1_br0":
            _push(ids, sym.get("br1"))
            _push(ids, sym.get("br0"))
            first = True
        elif kind == "pause":
            if first:
                i += 1
                continue
            if variant == "in":
                # 印式英语实测：句中逗号只插入 br3，不使用 punc 符号
                _push(ids, sym.get("br3"))
                first = True
                i += 1
                continue
            _push(ids, sym.get("br3"))
            if item[1]:
                ids.append(sym.get(item[1], sym.get("punc,", 0)))
            _push(ids, sym.get("br0"))
            first = True
        elif kind == "end":
            # 句末：br4 + 主标点 +（紧随的附加标点）+ ~；若后面还有内容，另起一句
            if variant == "in":
                # 印式英语实测：句末只保留 br4（若有）+ ~，不带 punc 标点
                _push(ids, sym.get("br4"))
                _push(ids, sym.get(END))
                if i + 1 < len(flat):
                    ids.extend([x for x in (sym.get(BOS), sym.get("br0")) if x is not None])
                    first = True
                    ended = False
                    i += 1
                    continue
                ended = True
                first = True
                i += 1
                continue
            _push(ids, sym.get("br4"))
            j = i + 1
            extra = []
            while j < len(flat) and flat[j][0] == "punc_after_end":
                extra.append(flat[j][1])
                j += 1
            next_end = flat[j] if j < len(flat) and flat[j][0] == "end" else None
            if item[1] == "punc." and len(extra) >= 2:
                # 省略号：引擎结构 br4 + punc... + punc.（后面紧跟 ?/! 时用对应标点）
                ids.append(sym.get("punc...", sym.get("punc.", 0)))
                if next_end is not None:
                    ids.append(sym.get(next_end[1], sym.get("punc.", 0)))
                    j += 1
                else:
                    ids.append(sym.get("punc.", 0))
            else:
                ids.append(sym.get(item[1], sym.get("punc.", 0)))
                for e in extra:
                    ids.append(sym.get(e, sym.get("punc.", 0)))
            _push(ids, sym.get(END))
            if j < len(flat):
                ids.extend([x for x in (sym.get(BOS), sym.get("br0")) if x is not None])
                first = True
                ended = False
                i = j
                continue
            ended = True
            first = True
            i = j
            continue
        elif kind == "punc_after_end":
            ids.append(sym.get(item[1], sym.get("punc.", 0)))
            i += 1
            continue
        elif kind == "quote_after_end":
            ids.append(sym.get("br0", 0))
            ids.append(sym.get(item[1], sym.get("punc.", 0)))
            i += 1
            continue
        elif kind == "br2":
            _push(ids, sym.get("br2") if sym.get("br2") is not None else sym.get("br1"))
        elif kind == "phones":
            if not first:
                _push(ids, sym.get("br1"))
            for ph in item[1]:
                if ph == "-":
                    _push(ids, sym.get("-") if sym.get("-") is not None else sym.get("&"))
                else:
                    ids.append(_sym_id(sym, ph, variant))
            first = False
        i += 1
    if not ended:
        _push(ids, sym.get("br4"))
        if variant != "in" and sym.get("punc."):
            _push(ids, sym.get("punc."))
        _push(ids, sym.get(END))
    return ids


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--package", default="")
    a = ap.parse_args()
    sym = load_symbols(os.path.join(a.package, "phones.txt"))
    ids = text_to_ids(a.text, sym)
    rev = {v: k for k, v in sym.items()}
    print(json.dumps(ids))
    print(" ".join(rev.get(i, "?%d" % i) for i in ids))
