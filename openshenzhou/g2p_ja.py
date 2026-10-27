#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2p_ja.py —— 日文文本 → 音素 ID（假名表 + pykakasi 汉字读音）

引擎实测格式：
    今日はいい天気ですね。
    k y o1 & w u & w a / y i & y i / t e1 & nn & k i / d e & s u / n e / punc。 ~

- "&" 分莫拉，"/" 分词，结尾 "/ punc。 ~"，没有 -BOS-/br0
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import pykakasi
except ImportError:
    pykakasi = None


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
    d = load_json_prefer_names(os.path.join(ROOT, "data", "lexicon_ja"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "lexicon_ja.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_kana():
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "kana_ja"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "kana_ja.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


SMALL = set("ゃゅょぁぃぅぇぉ")
PUNCT = {"。": "punc。", "、": "punc、", "？": "punc?", "！": "punc!",
         "「": "punc「", "」": "punc」", ",": "punc,", ".": "punc."}
SPACE = set(" \u3000\t\r\n")


def to_hiragana(text):
    if pykakasi is None:
        return [text]
    k = pykakasi.kakasi()
    words = []
    for item in k.convert(text):
        h = item.get("hira") or item.get("kana") or item.get("orig")
        if h:
            words.append(h)
    return words


def split_morae(word, table):
    out = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if i + 1 < n and word[i + 1] in SMALL and (ch + word[i + 1]) in table:
            out.append(ch + word[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    phones = []
    for idx, m in enumerate(out):
        if m == "っ":
            nxt = table.get(out[idx + 1], []) if idx + 1 < len(out) else []
            if nxt:
                phones.append([nxt[0]])
            continue
        if m == "ー":
            if phones and phones[-1]:
                phones.append([phones[-1][-1]])
            continue
        phones.append(list(table.get(m, [])))
    return phones


def _resolve_names(core, sym):
    """符号名 → id；缺符号时按变体兜底（去声调数字 / hh→h / pso→p 等）"""
    out = []
    for x in core:
        if isinstance(x, int):
            out.append(x)
            continue
        if x in sym:
            out.append(sym[x])
            continue
        cand = None
        for alt in (x.rstrip("0123456789"), x.replace("hh", "h"), x.replace("pso", "p"),
                    x.replace("tsu", "ts"), x.replace("dz", "z"), x.replace("v", "b")):
            if alt in sym:
                cand = sym[alt]
                break
        out.append(cand if cand is not None else sym.get("padding", 0))
    return out


JA_PARTICLE = {"は": "わ", "へ": "え", "を": "お", "っ": ""}
JA_NUM = set("一二三四五六七八九十百千万零〇")
JA_UNIT = set("時分秒月日年歳個回度")


def _split_num_unit(word):
    """引擎会把数词+单位拆开（十時→十|時、三時→三|時）"""
    if len(word) < 2 or word[-1] not in JA_UNIT:
        return None
    head = word[:-1]
    if head and all(ch in JA_NUM for ch in head):
        return [head, word[-1]]
    return None


def _seg_words(text):
    """用复刻的官方 LSTM 断词模型在原文上分词（标点单独成项）。"""
    out = []
    buf = []

    def flush():
        if not buf:
            return
        chunk = "".join(buf)
        buf.clear()
        try:
            from . import wb_model_ja
            chars = list(chunk)
            segs = wb_model_ja.segments(chars)
            for s, L in segs:
                w = chunk[s:s + L]
                if not w:
                    continue
                parts = _split_num_unit(w)
                if parts:
                    for pp in parts:
                        out.append(("w", pp))
                else:
                    out.append(("w", w))
        except Exception:
            out.append(("w", chunk))

    for ch in text:
        if ch in PUNCT:
            flush()
            out.append(("p", PUNCT[ch]))
        elif ch in SPACE:
            flush()
        else:
            buf.append(ch)
    flush()
    return out


def _word_hiragana(word):
    """单个词 → 平假名（词尾助词 は/へ/を 按日语读音特判）。"""
    if word in JA_PARTICLE:
        return JA_PARTICLE[word]
    if len(word) > 1 and word[-1] in JA_PARTICLE:
        head = _word_hiragana(word[:-1])
        return head + JA_PARTICLE[word[-1]]
    if pykakasi is None:
        return word
    try:
        k = pykakasi.kakasi()
        h = "".join((it.get("hira") or it.get("kana") or it.get("orig") or "")
                    for it in k.convert(word))
        return h or word
    except Exception:
        return word


# 汉字后缀在「汉字词之后」时的读音（引擎上下文读音）
# 值为莫拉 token 组（含音高重音记号），直接来自引擎实测
JA_SUFFIX = {
    "中": [["ch", "y", "u1"], ["w", "u"]],
    "日": [["n", "i", "ch", "i"]],
    "人": [["n", "i", "nn"]],
    "時": [["j", "i"]],
    "分": [["f", "u", "nn"]],
    "年": [["n", "e", "nn"]],
    "月": [["g", "a", "ts", "u"]],
    "課": [["k", "a"]],
    "者": [["sh", "a"]],
    "性": [["s", "e1", "y", "i"]],
}

KANJI_RANGE = (0x4E00, 0x9FFF)


def _is_kanji(ch):
    o = ord(ch)
    return KANJI_RANGE[0] <= o <= KANJI_RANGE[1]


_CTX = None


def load_ctx():
    global _CTX
    if _CTX is None:
        try:
            _CTX = json.load(open(os.path.join(ROOT, "data", "lexicon_ja_ctx.json"),
                                  encoding="utf-8"))
        except Exception:
            _CTX = {}
    return _CTX


def text_to_ids(text, sym, table, lexicon=None):
    from .symbols import resolve_table, resolve_lexicon
    table = resolve_table(table, sym)
    lexicon = resolve_lexicon(lexicon or {}, sym)
    frame = ("-BOS-" in sym or "-bos-" in sym) and ("br0" in sym)
    amp = sym.get("&")
    mora_sep = sym.get("-") if sym.get("-") is not None else amp
    slash = sym.get("/")
    tilde = sym.get("~")
    lexicon = lexicon or {}
    seq = _seg_words(text)
    ids = []
    first = True
    ended = False
    bos = sym.get("-BOS-", sym.get("-bos-"))
    if frame and bos is not None:
        ids.extend([bos, sym.get("br0", 0)])
    prev_kanji = False
    seq_words = [v for k, v in seq]
    prev_word = "^"
    for si, (kind, val) in enumerate(seq):
        next_word = "$"
        for sj in range(si + 1, len(seq)):
            if seq[sj][0] == "w":
                next_word = seq[sj][1]
                break
        if kind == "p":
            if frame:
                ids.append(sym.get("br4", 0))
                ids.append(sym.get(val, 0))
                ended = True
                first = True
                continue
            if slash is not None:
                ids.append(slash)
            ids.append(sym.get(val, 0))
            first = True
            ended = True
            prev_word = "^"
            continue
        if frame:
            # frame 类音色（Nanami 等）：莫拉之间用 "-"，词之间 br1
            groups = []
            if False:  # frame 类音色不用 Keita 词表（结构不同）
                toks = _resolve_names(list(lexicon[val]), sym)
                delim = sym.get("&")
                cur = []
                for x in toks:
                    if x == delim:
                        if cur:
                            groups.append(cur)
                        cur = []
                    elif x in (sym.get("-"), delim):
                        continue
                    else:
                        cur.append(x)
                if cur:
                    groups.append(cur)
            else:
                for ph in split_morae(_word_hiragana(val), table):
                    g = _resolve_names(ph, sym)
                    if g:
                        groups.append(g)
            if not groups:
                continue
            if not first:
                ids.append(sym.get("br1", sym.get("br0", 0)))
            sep = sym.get("-")
            for gi, g in enumerate(groups):
                if gi > 0 and sep is not None:
                    ids.append(sep)
                ids.extend(g)
            first = False
            prev_kanji = bool(val) and _is_kanji(val[-1])
            prev_word = val
            continue
        ctx = load_ctx()
        ctx_key = "%s|%s|%s" % (prev_word, val, next_word)
        if not frame and ctx_key in ctx:
            core = _resolve_names(list(ctx[ctx_key]), sym)
        elif val in JA_PARTICLE:
            core = []
            for ph in split_morae(JA_PARTICLE[val], table):
                core.extend(ph)
        elif len(val) == 1 and val in JA_SUFFIX and prev_kanji:
            core = []
            for gi3, grp in enumerate(JA_SUFFIX[val]):
                if gi3 > 0 and mora_sep is not None:
                    core.append(mora_sep)
                core.extend(_resolve_names(grp, sym))
        elif val in lexicon:
            toks = list(lexicon[val])
            sep_ids = {v for v in (sym.get("&"), sym.get("-")) if v is not None}
            groups = []
            cur = []
            for x in toks:
                if x in sep_ids:
                    if cur:
                        groups.append(cur)
                    cur = []
                else:
                    cur.append(x)
            if cur:
                groups.append(cur)
            core = []
            for gi2, g in enumerate(groups):
                if gi2 > 0 and mora_sep is not None:
                    core.append(mora_sep)
                core.extend(g)
        else:
            core = None
            # 词尾助词 は/へ/を：前缀若在词表里，用词表读音 + 助词读音拼起来
            if len(val) > 1 and val[-1] in JA_PARTICLE and val[:-1] in lexicon:
                core = list(lexicon[val[:-1]])
                if mora_sep is not None:
                    core.append(mora_sep)
                for ph in split_morae(JA_PARTICLE[val[-1]], table):
                    core.extend(ph)
            if core is None:
                hira = _word_hiragana(val)
                morae = split_morae(hira, table)
                core = []
                for mi, ph in enumerate(morae):
                    if mi > 0 and mora_sep is not None:
                        core.append(mora_sep)
                    core.extend(ph)
        if not core:
            continue
        if not first and slash is not None:
            ids.append(slash)
        ids.extend(_resolve_names(core, sym))
        first = False
        prev_kanji = bool(val) and _is_kanji(val[-1])
        prev_word = val
    if frame:
        if not ended:
            ids.append(sym.get("br4", 0))
        ids.append(sym.get("~", tilde if tilde is not None else 0))
        return ids
    if slash is not None and not ended:
        ids.append(slash)
    if tilde is not None:
        ids.append(tilde)
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--package", default=os.path.join(
        os.environ.get("TEMP", ""), "decoded", "Keita_Male_ja-JP"))
    a = ap.parse_args()
    sym = load_symbols(os.path.join(a.package, "phones.txt"))
    ids = text_to_ids(a.text, sym, load_kana(), load_lexicon())
    rev = {v: k for k, v in sym.items()}
    print(json.dumps(ids))
    print(" ".join(rev.get(i, "?%d" % i) for i in ids))
