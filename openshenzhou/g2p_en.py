#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""g2p_en.py —— 英文文本 → 微软语音包音素 ID（引擎行为复刻版）

数据来源全部为引擎实测（ORT 探针标定）：
  data/lexicon_en_context.json   语料句内「词 → 音素名序列」（含弱读形态）
  data/lexicon_en_engine.json    单词隔离标定的「词 → 音素名序列」
  data/lexicon_en.names.json     早期标定词表
  data/cmudict_en.json           CMUdict 兜底
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
                r = json.loads(line)
                iso.setdefault(r["w"], r["phones"])
        _ISO = iso
    return _ISO


def load_lexicon():
    return load_engine_lexicon()


def load_symbols(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        name, idx = line.rsplit(":", 1)
        m[name] = int(idx)
    return m


# ---------------- 音素工具 ----------------
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
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD_ONES = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
            "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth"]
ORD_TENS = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
            "seventieth", "eightieth", "ninetieth"]
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
CURRENCY = {"$": ("dollar", "cent"), "€": ("euro", "cent"), "£": ("pound", "penny"),
            "¥": ("yen", "sen")}
SPECIAL_PHONES = {
    "underscore": ["ah2", "n", "-", "d", "ax", "r", "-", "s", "k", "ao1", "r"],
    "asterisk": ["ae1", "-", "s", "t", "ax", "-", "r", "ih", "s", "k"],
}
STRONG_FORMS = {"and": ["ae1", "n", "d"], "for": ["f", "ao1", "r"]}
ABBREV = {}
SYMBOL_WORDS = {"*": "asterisk", "_": "underscore", "#": "number", "@": "at",
                "&": "and", "=": "equals", "+": "plus", "<": "less", ">": "greater",
                "~": "tilde", "|": "bar"}


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


def with_syllables(phones):
    groups = split_syllables(phones)
    out = []
    for gi, g in enumerate(groups):
        if gi:
            out.append("-")
        out.extend(g)
    return out


# ---------------- 数字 / TN ----------------
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
        return w
    base = cardinal_words(n)
    if n % 100 == 0:
        return base
    return base[:-1] + [ordinal_words(n % 100)[-1]]


def digits_words(ds):
    return [ONES[int(c)] for c in ds]


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


# ---------------- 词 → 音素名 ----------------
def _engine_phones(word):
    w = word.lower().replace("’", "'")
    for src in (load_context(), load_engine_lexicon(), _VOCAB):
        v = src.get(w) if src else None
        if v:
            return list(v)
    return None


def _fallback_phones(word):
    w = word.lower()
    if pronouncing is not None:
        ph = pronouncing.phones_for_word(w)
        if ph:
            return [cmu_to_engine(p) for p in ph[0].split()]
    out = []
    for ch in w:
        if ch.isalpha():
            out.extend(LETTER_RULES.get(ch, []))
    return out


def _load_vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _load_json(os.path.join(ROOT, "data", "lexicon_en.names.json"), {})
    return _VOCAB


def pron_word(word, next_word=""):
    """词 → 引擎音素名序列（含 '-' 音节分隔）"""
    w = word.lower().replace("’", "'")
    if w == "the":
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
        flat = []
        for p in parts:
            flat.extend([x for x in pron_word(p) if x != "-"])
        return with_syllables(flat)
    return with_syllables(_fallback_phones(w))


# ---------------- 文本解析 ----------------
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*(?:-[A-Za-z]+)*")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?(?:am|pm|a\.m\.|p\.m\.)?", re.I)
END_PUNCT = {".": "punc.", "?": "punc?", "!": "punc!"}
PAUSE_PUNCT = {",": "punc,", ";": "punc;", ":": None, "(": "punc,", ")": "punc,",
               "[": "punc,", "]": "punc,"}


def _verbalize_number(tok):
    t = tok.strip()
    low = t.lower()
    if re.fullmatch(r"\d{1,2}:\d{2}", t):
        hh, mm = t.split(":")
        if mm == "00":
            return cardinal_words(int(hh)) + ["o", "clock"]
        return cardinal_words(int(hh)) + ["oh", "zero"] if mm == "00" else \
            cardinal_words(int(hh)) + cardinal_words(int(mm))
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", low)
    if m:
        return ordinal_words(int(m.group(1)))
    if re.fullmatch(r"\d+\.\d+", t):
        a, b = t.split(".")
        return cardinal_words(int(a)) + ["point"] + digits_words(b)
    core = t.replace(",", "")
    if re.fullmatch(r"\d+", core):
        n = int(core)
        if len(core) == 4 and 1000 <= n <= 2099:
            return year_words(n)
        return cardinal_words(n)
    return [t]


def _parse(text):
    events = []
    last_word = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch.isdigit():
            m = NUM_RE.match(text, i)
            tok = m.group(0)
            i = m.end()
            m2 = re.match(r"-(\d+)", text[i:i + 8])
            if m2 and len(tok) == 3:
                events.append(("phone", tok))
                i += len(m2.group(0))
                events.append(("phone2", m2.group(1)))
                continue
            events.append(("number", tok))
            continue
        if ch in CURRENCY:
            cur = ch
            i += 1
            m = NUM_RE.match(text, i)
            if m:
                events.append(("currency", (cur, m.group(0))))
                i = m.end()
            else:
                events.append(("symbol", ch))
            continue
        if ch == "%":
            events.append(("percent", None))
            i += 1
            continue
        if ch in END_PUNCT:
            events.append(("end", END_PUNCT[ch]))
            i += 1
            continue
        if ch in PAUSE_PUNCT:
            events.append(("pause", PAUSE_PUNCT[ch]))
            i += 1
            continue
        if ch in SYMBOL_WORDS:
            events.append(("symbol", ch))
            i += 1
            continue
        if ch.isalpha():
            m = WORD_RE.match(text, i)
            if m:
                w = m.group(0)
                i = m.end()
                if i < n and text[i] == ".":
                    low = w.lower()
                    if ABBREV and low in ABBREV:
                        events.append(("words", ABBREV[low].split()))
                        i += 1
                        last_word = low
                        continue
                    if len(w) == 1 and w.isupper():
                        if not (last_word in CNT_WORDS and w in ROMAN):
                            events.append(("letters", w))
                            i += 1
                            last_word = low
                            continue
                events.append(("word", w))
                last_word = w.lower()
                continue
        i += 1
    return events


CNT_WORDS = {"chapter", "part", "section", "book", "volume", "act", "scene",
             "article", "item", "figure", "table", "appendix", "page", "no"}


def _expand_events(events):
    out = []
    prev_word = ""
    for idx, (kind, val) in enumerate(events):
        nxt = events[idx + 1] if idx + 1 < len(events) else None
        if kind == "number":
            t = val.replace(",", "")
            if nxt and nxt[0] == "end" and len(t) == 4 and t.isdigit():
                out.append(("words", year_words(int(t))))
            else:
                out.append(("words", _verbalize_number(val)))
        elif kind == "phone":
            out.append(("words", digits_words(val)))
            out.append(("pause_plain", None))
        elif kind == "phone2":
            out.append(("words", digits_words(val)))
        elif kind == "currency":
            cur, num = val
            major, minor = CURRENCY[cur]
            if "." in num:
                a, b = num.split(".")
                a = a.replace(",", "")
                mn = int(a) if a else 0
                cents = int((b + "00")[:2])
                words = cardinal_words(mn) + [major + ("s" if mn != 1 else "")]
                if cents:
                    words += ["and"] + cardinal_words(cents) + [minor + ("s" if cents != 1 else "")]
            else:
                mn = int(num.replace(",", ""))
                words = cardinal_words(mn) + [major + ("s" if mn != 1 else "")]
            out.append(("words", words))
        elif kind == "percent":
            if out and out[-1][0] == "words":
                out[-1] = ("words", out[-1][1] + ["percent"])
        elif kind == "word":
            exp = _word_expansion(val, prev_word)
            out.append(("words", exp))
            if exp and exp[-1].replace("-", "").isalpha() is False:
                pass
            if len(exp) == 1 and exp[0].replace("-", "").isalpha():
                prev_word = exp[0].lower()
            elif exp:
                prev_word = exp[0].lower()
        elif kind == "pause":
            out.append(("pause", val))
        elif kind == "end":
            out.append(("end", val))
        elif kind == "symbol":
            out.append(("words", [SYMBOL_WORDS[val]]))
    return out


def _word_expansion(w, prev_word=""):
    if re.fullmatch(r"[IVXLCDM]+", w) and w != "I":
        single = len(w) == 1
        if single and prev_word not in CNT_WORDS:
            return [w]
        n = 0
        prev = 0
        for ch in reversed(w):
            v = ROMAN[ch]
            n += -v if v < prev else v
            prev = max(prev, v)
        if 0 < n < 4000:
            return cardinal_words(n)
    return [w]


def text_to_ids(text, sym, lexicon=None):
    ids = [sym[BOS], sym["br0"]]
    events = _expand_events(_parse(text))
    # letters 事件 → 字母词（A→a 等）
    ev2 = []
    for kind, val in events:
        if kind == "letters":
            for ch in val:
                ev2.append(("words", [ch.lower()]))
        else:
            ev2.append((kind, val))
    events = ev2
    flat = []
    for kind, val in events:
        if kind == "words":
            for w in val:
                flat.append(("w", w))
        elif kind == "pause_plain":
            flat.append(("pause_plain", None))
        else:
            flat.append((kind, val))
    first = True
    ended = False
    for k, item in enumerate(flat):
        if item[0] == "pause_plain":
            ids.append(sym["br3"])
            first = True
        elif item[0] == "pause":
            if first:
                continue
            ids.append(sym["br3"])
            if item[1]:
                ids.append(sym.get(item[1], sym.get("punc,", 0)))
            ids.append(sym["br0"])
            first = True
        elif item[0] == "end":
            ids.append(sym["br4"])
            ids.append(sym.get(item[1], sym.get("punc.", 0)))
            ended = True
            first = True
        else:
            w = item[1]
            nxt = ""
            for item2 in flat[k + 1:]:
                if item2[0] == "w":
                    nxt = item2[1]
                    break
            phones = pron_word(w, nxt)
            strong = STRONG_FORMS.get(w.lower()) if w else None
            if strong:
                after = flat[k + 1][0] if k + 1 < len(flat) else "end"
                if after in ("pause", "end"):
                    phones = list(strong)
            if not first:
                ids.append(sym["br1"])
            for ph in phones:
                if ph == "-":
                    ids.append(sym["-"])
                else:
                    ids.append(sym.get(ph, sym.get("<pad>", 0)))
            first = False
    if not ended:
        ids.append(sym["br4"])
    ids.append(sym[END])
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
