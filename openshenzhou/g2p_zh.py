#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2p_zh.py —— 中文文本 → am_v5 音素 ID 序列

数据来源：对引擎逐字合成，抓取 encoder 输入张量（见 docs/model-extracted.md），
得到「拼音(带声调) → 3 个音素符号」的表，再拼装成完整序列。

序列格式（实测）：
    -bos-  br0  <syl1>  -  <syl2>  -  ...  brN  [punc]  ~
每个音节 = 3 个符号：声母(+介音) / 主元音+声调 / 韵尾+声调

声调后缀规律（实测）：
    声调1: (h, h)   声调2: (l, h)   声调3: (l, l)   声调4: (h, l)   轻声5: (m, m)

  python tools/g2p_zh.py --text "你好，世界。" [--phones-dir <语音包目录>]
"""
import argparse
import json
import os
import re
import sys

from pypinyin import pinyin, Style

try:
    import jieba
    jieba.setLogLevel(60)
except ImportError:
    jieba = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 特殊符号
BOS, SEP, BR0, BR1, BR2, BR3, BR4, END = "-bos-", "-", "br0", "br1", "br2", "br3", "br4", "~"
PUNCT = {"，": "punc,", "。": "punc。", "？": "punc?", "！": "punc!", "、": "punc、", "；": "punc;", "：": "punc:"}


def load_symbols(phones_txt):
    sym2id = {}
    for line in open(phones_txt, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        name, idx = line.rsplit(":", 1)
        sym2id[name] = int(idx)
    return sym2id


_SEG_WORDS = None


def load_seg_words():
    """官方词典提取的词表（78k），用于分词候选；读音仍以标定 lexicon 为准。
    只保留 3 字及以上：官方词表里的 2 字串有较多二进制误读（如「园区」），
    分词时优先用「已标定的引擎词」来保证边界与引擎一致。"""
    global _SEG_WORDS
    if _SEG_WORDS is not None:
        return _SEG_WORDS
    p = os.path.join(ROOT, "data", "official_words_zh.json")
    try:
        words = json.load(open(p, encoding="utf-8"))
        _SEG_WORDS = set(words)
    except Exception:
        _SEG_WORDS = set()
    return _SEG_WORDS


def load_lexicon():
    """引擎标定的「词 → 音素序列」（符号名，使用时映射到本音色包 id）"""
    from .symbols import load_json_prefer_names
    d = load_json_prefer_names(os.path.join(ROOT, "data", "lexicon_zh"))
    if d:
        return d
    p = os.path.join(ROOT, "data", "lexicon_zh.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_table():
    table = {}
    for path in [os.path.join(ROOT, "data", "g2p_table.json")]:
        if os.path.exists(path):
            table.update(json.load(open(path, encoding="utf-8")))
    for name in ("g2p_extra.tsv", "g2p_samples.tsv"):
        p = os.path.join(ROOT, "data", name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            toks = parts[2].split()
            if len(toks) == 7:
                table.setdefault(parts[1], toks[2:5])
    return table


def tone_suffixes(tone):
    return {1: ("h", "h"), 2: ("l", "h"), 3: ("l", "l"), 4: ("h", "l"), 5: ("m", "m")}[tone]


def apply_sandhi(syls):
    """三声连读 3+3→2+3；「不」在四声前 bù→bú（同引擎）"""
    out = list(syls)
    for i in range(len(out) - 1):
        base_i, t_i = out[i][:-1], out[i][-1]
        base_n, t_n = out[i + 1][:-1], out[i + 1][-1]
        if t_i == "3" and t_n == "3":
            out[i] = base_i + "2"
        if base_i == "bu" and t_i == "4" and t_n == "4":
            out[i] = base_i + "2"
    return out


def segment_words(text_cn, wordset=None, maxlen=4):
    """按「引擎词表」做最长匹配分词。
    词表来自批量标定得到的 lexicon_zh（词条即引擎词，含内部 br1 的即为词组）。"""
    if not wordset:
        if jieba is None:
            return [text_cn[i:i+2] for i in range(0, len(text_cn), 2)]
        try:
            jieba.initialize()
        except Exception:
            pass
        freq = jieba.dt.FREQ
        words = []
        i, n = 0, len(text_cn)
        while i < n:
            if i + 1 < n and freq.get(text_cn[i:i+2], 0) > 0:
                words.append(text_cn[i:i+2])
                i += 2
            else:
                words.append(text_cn[i])
                i += 1
        return words
    words = []
    i, n = 0, len(text_cn)
    while i < n:
        hit = None
        for L in range(min(maxlen, n - i), 0, -1):
            cand = text_cn[i:i+L]
            if cand in wordset:
                hit = cand
                break
        if hit:
            words.append(hit)
            i += len(hit)
        else:
            words.append(text_cn[i])
            i += 1
    return words


def lexicon_groups(core, sep_id=3, br1_id=25):
    """把 lexicon 词条 ids 解析成「词内音节组」：list[list[3 个 id]]，br1 处断词"""
    groups, cur = [], []
    i = 0
    toks = []
    for x in core:
        if x == br1_id:
            groups.append(toks)
            toks = []
        else:
            toks.append(x)
    groups.append(toks)
    out = []
    for g in groups:
        if not g:
            continue
        out.append([g[i:i+3] for i in range(0, len(g), 3)])
    return out


APICAL = ("zh", "ch", "sh", "r", "z", "c", "s")


def _load_finals():
    p = os.path.join(ROOT, "data", "g2p_finals.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))["finals"]
    return {}


_FINALS = _load_finals()


def syl_to_symbols(py, table):
    """拼音(带声调) → 3 个音素符号。优先用引擎实测统计表，其次规则。"""
    tbl = load_pinyin_symbols()
    hit = tbl.get(py)
    if hit and len(hit) == 3:
        return list(hit)
    return _syl_to_symbols_rule(py, table)


def _syl_to_symbols_rule(py, table):
    """拼音(带声调) → 3 个音素符号。规则来自 448 条引擎实测样本。"""
    base = re.sub(r"[1-5]$", "", py)
    tone = int(py[-1]) if py[-1] in "12345" else 5
    mm = re.match(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]|)(.*)$", base)
    ini, fin = mm.group(1), mm.group(2)
    s1, s2 = tone_suffixes(tone)

    seg = None
    if ini == "y":
        if fin.startswith("u"):
            t1 = "yv"
            seg = {"u": ("v", "v"), "ue": ("eh", "eh"), "uan": ("a", "nn"),
                   "un": ("el", "nn")}.get(fin)
        else:
            t1 = "yi"
            seg = _FINALS.get(fin)
            if fin == "e":
                seg = ("eh", "eh")
    elif ini == "w":
        t1 = "wu"
        seg = _FINALS.get(fin)
    elif fin == "i" and ini in ("zh", "ch", "sh", "r"):
        t1 = ini
        seg = ("ib", "ib")
    elif fin == "i" and ini in ("z", "c", "s"):
        t1 = ini
        seg = ("if", "if")
    else:
        # 介音
        medial = ""
        if ini in ("j", "q", "x") and fin in ("u", "ue", "uan", "un"):
            medial = "v"
        elif fin.startswith("u"):
            medial = "u"
        elif fin.startswith("i"):
            medial = "i"
        elif fin == "ong":
            medial = "u"
        elif fin == "v":
            medial = "v"
        if ini == "":
            if fin.startswith("i"):
                t1 = "y" + medial
            elif fin.startswith("u"):
                t1 = "w" + medial
            else:
                t1 = "g" + {"a": "a", "e": "e", "o": "o", "e": "e"}.get(fin[:1], "e")
        else:
            t1 = ini + medial
        seg = _FINALS.get(fin)

    if ini in ("j", "q", "x") and fin in ("u", "ue", "uan", "un"):
        seg = {"u": ("v", "v"), "ue": ("eh", "eh"), "uan": ("a", "nn"), "un": ("el", "nn")}[fin]
    if seg is None and ini == "y" and fin == "e":
        seg = ("eh", "eh")
    if seg is None:
        if py in table:
            return table[py]
        seg = (fin, fin)
    return [t1, seg[0] + "_" + s1, seg[1] + "_" + s2]



DIGITS = "零一二三四五六七八九"
NUM_UNITS = "十百千"


def _read_digit_seq(ds):
    """按位读（年份/电话号码/小数位）"""
    return "".join(DIGITS[int(c)] for c in ds if c.isdigit())


def _read_group(n):
    """0 < n < 10000"""
    units = ["", "十", "百", "千"]
    out = ""
    s = str(n)
    L = len(s)
    zero = False
    for i, ch in enumerate(s):
        d = int(ch)
        pos = L - 1 - i
        if d == 0:
            zero = True
            continue
        if zero and out:
            out += "零"
        zero = False
        out += DIGITS[d] + units[pos]
    if out.startswith("一十"):
        out = out[1:]
    return out


def _read_digits(ds):
    """数字串 → 中文读法（整数值读法）；超长串退化为按位读"""
    if not ds:
        return ""
    ds = ds.lstrip("0") or "0"
    if len(ds) > 12:
        return _read_digit_seq(ds)
    n = int(ds)
    if n == 0:
        return "零"
    groups = ["", "万", "亿", "万亿"]
    out = ""
    gi = 0
    while n > 0:
        part = n % 10000
        n //= 10000
        if part:
            out = _read_group(part) + groups[gi] + out
        elif out and not out.startswith("零"):
            out = "零" + out
        gi += 1
    return out.rstrip("零")


# 数字表达式（引擎会把「数字+单位(年月日点)」当成一个 token）
NUM_RE = re.compile(
    r"(?P<year>\d{4})(?P<yunit>年)"
    r"|(?P<dec>\d+)\.(?P<frac>\d+)"
    r"|(?P<pct>\d+)%"
    r"|(?P<phone>\d{7,})"
    r"|(?P<md>\d+)(?P<mdunit>[月日])"
    r"|(?P<int>\d+)"
)

PUNCT_CHARS = "，。？！、；：,"


SYM_RULES = [
    ("℃", "摄氏度"),
    ("℉", "华氏度"),

    ("～", "至"),
    ("×", "乘以"),
    ("÷", "除以"),
    ("±", "正负"),
]


def tn_symbols(text):
    """符号/日期/时间/百分比的规范化（照着引擎实测行为）"""
    # 货币符号在数字前：￥100 -> 100元
    text = re.sub(r"[￥¥]\s*(\d+(?:\.\d+)?)\s*(万|亿)?", lambda m: m.group(1) + (m.group(2) or "") + "元", text)
    text = re.sub(r"[\$]\s*(\d+(?:\.\d+)?)", lambda m: m.group(1) + "美元", text)
    # 百分比：99.9% -> 百分之99.9
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", lambda m: "百分之" + m.group(1), text)
    # 日期 2026/10/26 或 2026-10-26 -> 2026年10月26日
    def _date(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return "%s年%s月%s日" % (y, str(int(mo)), str(int(d)))
    text = re.sub(r"(?<!\d)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", _date, text)
    # 时间 9:30 -> 9点30分
    def _time(m):
        h, mi = m.group(1), m.group(2)
        return "%s点%s分" % (h, mi) if int(mi) else "%s点" % h
    text = re.sub(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)", _time, text)
    for a, b in SYM_RULES:
        text = text.replace(a, b)
    return text


def normalize_segments(text):
    """文本 → 片段列表 [('t', 文本) | ('n', 数字读法短语)]

    数字片段会被当成一个词（引擎里就是如此），
    并吸收紧跟的「年/月/日/点」等量词。
    """
    text = tn_symbols(text)
    segs = []
    pos = 0
    for m in NUM_RE.finditer(text):
        if m.start() > pos:
            segs.append(("t", text[pos:m.start()]))
        if m.group("year"):
            segs.append(("n", _read_digit_seq(m.group("year")) + "年"))
        elif m.group("dec") is not None:
            txt = _read_digits(m.group("dec")) + "点" + _read_digit_seq(m.group("frac"))
            endp = m.end()
            while endp < len(text) and text[endp] in "点元年":
                txt += text[endp]
                endp += 1
            segs.append(("n", txt))
            pos = endp
            continue
        elif m.group("pct"):
            segs.append(("t", "百分之" + _read_digits(m.group("pct"))))
        elif m.group("phone"):
            segs.append(("n", _read_digit_seq(m.group("phone"))))
        elif m.group("md"):
            segs.append(("n", _read_digits(m.group("md")) + m.group("mdunit")))
        else:
            txt = _read_digits(m.group("int"))
            # 引擎会把紧跟的量词「点/元」并入数字 token
            endp = m.end()
            while endp < len(text) and text[endp] in "点元分秒":
                txt += text[endp]
                endp += 1
            segs.append(("n", txt))
            pos = endp
            continue
        pos = m.end()
    if pos < len(text):
        segs.append(("t", text[pos:]))
    return segs


_CHAR_READ = None


def load_char_readings():
    """引擎标定的「汉字最常见发音」（符号名三音素）"""
    global _CHAR_READ
    if _CHAR_READ is None:
        try:
            p = os.path.join(ROOT, "data", "zh_char_readings.json")
            _CHAR_READ = json.load(open(p, encoding="utf-8"))
        except Exception:
            _CHAR_READ = {}
    return _CHAR_READ


def correct_char_syl(ch, sym2id):
    """返回该字的纠正读音（id 三元组）或 None"""
    tbl = load_char_readings()
    corr = tbl.get(ch)
    if not corr:
        return None
    return [sym2id.get(x, sym2id.get("padding", 0)) for x in corr]


_CTX_WS = None


def load_ctx_wordset():
    global _CTX_WS
    if _CTX_WS is None:
        try:
            _CTX_WS = set(json.load(open(os.path.join(ROOT, "data", "zh_ctx_words.json"),
                                         encoding="utf-8")))
        except Exception:
            _CTX_WS = set()
    return _CTX_WS


_CORE_WS = None


def load_core_wordset():
    global _CORE_WS
    if _CORE_WS is None:
        try:
            _CORE_WS = set(json.load(open(os.path.join(ROOT, "data", "zh_seg_wordset.json"),
                                          encoding="utf-8")))
        except Exception:
            _CORE_WS = set()
    return _CORE_WS


_PINYIN_SYM = None
_CJK_ISO = None


def load_pinyin_symbols():
    global _PINYIN_SYM
    if _PINYIN_SYM is None:
        try:
            _PINYIN_SYM = json.load(open(os.path.join(ROOT, "data", "zh_pinyin_symbols.json"),
                                         encoding="utf-8"))
        except Exception:
            _PINYIN_SYM = {}
    return _PINYIN_SYM


def load_cjk_isolated():
    global _CJK_ISO
    if _CJK_ISO is None:
        try:
            _CJK_ISO = json.load(open(os.path.join(ROOT, "data", "cjk_isolated.json"),
                                      encoding="utf-8"))
        except Exception:
            _CJK_ISO = {}
    return _CJK_ISO


def isolated_char_syl(ch, sym2id):
    """引擎单字读音表（覆盖全部 CJK 基本区）"""
    tbl = load_cjk_isolated()
    hit = tbl.get(ch)
    if not hit:
        return None
    return [sym2id.get(x, sym2id.get("padding", 0)) for x in hit]


_CHAR_FIX = None


def load_char_fix():
    """语料统计的单字读音修正表（uni/tri/bi_l/bi_r）。"""
    global _CHAR_FIX
    if _CHAR_FIX is None:
        try:
            _CHAR_FIX = json.load(open(os.path.join(ROOT, "data", "zh_char_fix.json"),
                                       encoding="utf-8"))
        except Exception:
            _CHAR_FIX = {}
    return _CHAR_FIX


FIX_MODE = os.environ.get("OS_CHAR_FIX", "0")


def fix_char_syl(ch, prev_ch, next_ch, sym2id):
    fix = load_char_fix() if FIX_MODE != "0" else None
    if not fix:
        return None
    if FIX_MODE in ("tri", "oovtri"):
        cands = [(fix.get("tri"), "%s|%s|%s" % (prev_ch, ch, next_ch))]
    elif FIX_MODE == "tri2":
        cands = [(fix.get("tri"), "%s|%s|%s" % (prev_ch, ch, next_ch))]
    elif FIX_MODE == "tribi":
        cands = [
            (fix.get("tri"), "%s|%s|%s" % (prev_ch, ch, next_ch)),
            (fix.get("bi_l"), "%s|%s" % (prev_ch, ch)),
            (fix.get("bi_r"), "%s|%s" % (ch, next_ch)),
        ]
    else:
        cands = [
            (fix.get("tri"), "%s|%s|%s" % (prev_ch, ch, next_ch)),
            (fix.get("bi_l"), "%s|%s" % (prev_ch, ch)),
            (fix.get("bi_r"), "%s|%s" % (ch, next_ch)),
            (fix.get("uni"), ch),
        ]
    for tbl, key in cands:
        if not tbl:
            continue
        hit = tbl.get(key)
        if hit:
            return [sym2id.get(x, sym2id.get("padding", 0)) for x in hit]
    return None


_CHAR_CTX = None


def load_char_ctx():
    global _CHAR_CTX
    if os.environ.get("OS_CHAR_CTX", "1") == "0":
        return {}
    if _CHAR_CTX is None:
        merged = {}
        # 旧表（人工/早期采集）兜底，新表（语料统计）优先
        files = ["zh_char_ctx.json"]
        if os.environ.get("OS_CTX2", "0") == "1":
            files.append("zh_char_ctx2.json")
        for fname in files:
            try:
                d = json.load(open(os.path.join(ROOT, "data", fname), encoding="utf-8"))
            except Exception:
                continue
            for tbl in ("tri", "bi_r", "bi_l", "uni"):
                if d.get(tbl):
                    merged.setdefault(tbl, {}).update(d[tbl])
        _CHAR_CTX = merged
    return _CHAR_CTX


def ctx_char_syl(ch, prev_ch, next_ch, sym2id):
    """上下文敏感的读音（只覆盖多读音汉字），返回 id 三元组或 None"""
    ctx = load_char_ctx()
    if not ctx:
        return None
    levels = os.environ.get("OS_CTX_LEVELS", "tri,bi_r,bi_l,uni").split(",")
    cands = []
    if "tri" in levels:
        cands.append((ctx.get("tri"), "%s|%s|%s" % (prev_ch, ch, next_ch)))
    if "bi_r" in levels:
        cands.append((ctx.get("bi_r"), "%s|%s" % (ch, next_ch)))
    if "bi_l" in levels:
        cands.append((ctx.get("bi_l"), "%s|%s" % (prev_ch, ch)))
    if "uni" in levels and os.environ.get("OS_CTX_UNI", "1") == "1":
        cands.append((ctx.get("uni"), ch))
    for tbl, key in cands:
        if not tbl:
            continue
        hit = tbl.get(key)
        if hit:
            return [sym2id.get(x, sym2id.get("padding", 0)) for x in hit.split(".")]
    return None


_ZH_CTX = None
USE_ZH_CTX = os.environ.get("OS_ZH_CTX", "0") == "1"


def load_zh_ctx():
    """语料统计的「前词|词|后词 → 读音」上下文表。"""
    global _ZH_CTX
    if _ZH_CTX is None:
        try:
            _ZH_CTX = json.load(open(os.path.join(ROOT, "data", "zh_ctx_readings.json"),
                                     encoding="utf-8"))
        except Exception:
            _ZH_CTX = {}
    return _ZH_CTX


_TOKEN_FIX = None
USE_TOKEN_FIX = os.environ.get("OS_TOKEN_FIX", "0") == "1"


def load_token_fix():
    """引擎语料统计的「词 → 实测读音」修正表。"""
    global _TOKEN_FIX
    if _TOKEN_FIX is None:
        try:
            _TOKEN_FIX = json.load(open(os.path.join(ROOT, "data", "zh_token_fix.json"),
                                        encoding="utf-8"))
        except Exception:
            _TOKEN_FIX = {}
    return _TOKEN_FIX


_TOKEN_READ = None


def load_token_readings():
    """语料统计得到的「引擎词 → 逐字读音符号」表（可能随语境含变调）。"""
    global _TOKEN_READ
    if _TOKEN_READ is None:
        try:
            _TOKEN_READ = json.load(open(os.path.join(ROOT, "data", "zh_token_readings.json"),
                                         encoding="utf-8"))
        except Exception:
            _TOKEN_READ = {}
    return _TOKEN_READ


_SEG_MODEL = None


def load_seg_model():
    """加载分词边界模型（data/zh_seg_model.npz，可缺省）"""
    global _SEG_MODEL
    if os.environ.get("OS_SEG_MODEL", "1") == "0":
        return False
    if _SEG_MODEL is None:
        try:
            from .segmodel import SegModel
            _SEG_MODEL = SegModel.load(os.environ.get("OS_SEG_MODEL") if os.environ.get("OS_SEG_MODEL", "1") not in ("", "0", "1") else None) or False
        except Exception:
            _SEG_MODEL = False
    return _SEG_MODEL


_CHAR_RULES = None


def load_char_rules():
    global _CHAR_RULES
    if _CHAR_RULES is None:
        try:
            _CHAR_RULES = json.load(open(os.path.join(ROOT, "data", "zh_char_rules.json"),
                                         encoding="utf-8"))
        except Exception:
            _CHAR_RULES = {}
    return _CHAR_RULES


def _apply_char_rules(chars, syls, sym2id):
    """按语料高置信规则覆盖多音字读音（tri > prev > next）。"""
    rules = load_char_rules()
    if not rules:
        return syls
    tri = rules.get("tri") or {}
    pr = rules.get("prev") or {}
    nx = rules.get("next") or {}
    for i, ch in enumerate(chars):
        if ch in ("一", "不"):
            continue
        if i >= len(syls) or not syls[i] or len(syls[i]) != 3:
            continue
        p = chars[i - 1] if i > 0 else "^"
        q = chars[i + 1] if i + 1 < len(chars) else "$"
        mode = os.environ.get("OS_CHAR_RULES_MODE", "all")
        hit = None
        if mode in ("all", "tri"):
            hit = tri.get("%s|%s|%s" % (p, ch, q))
        if hit is None and mode in ("all", "prev"):
            hit = pr.get("%s|%s" % (p, ch))
        if hit is None and mode in ("all", "next"):
            hit = nx.get("%s|%s" % (ch, q))
        if not hit:
            continue
        rep = hit.get("rep") if isinstance(hit, dict) else hit
        if not rep or len(rep) != 3:
            continue
        syls[i] = [sym2id.get(x, sym2id.get("padding", 0)) for x in rep]
    return syls


_YI_EXCEPT = set("第唯统单之十")


def _yi_sandhi_syls(chars, syls, sym2id):
    """「一」变调：词首/一般情况变调，前字属例外表时保持 yī。

    语料统计（5.4 万句）：「一」变调率 94-100%，但前字为
    第/唯/统/单/之/十 时变调率为 0（统一、第一、之一、单一、十一…）。
    """
    if not chars or not syls:
        return syls
    rev = None
    for i, ch in enumerate(chars):
        if ch != "一" or i >= len(syls) or not syls[i]:
            continue
        if i > 0 and chars[i - 1] in _YI_EXCEPT:
            continue
        if rev is None:
            rev = {v: k for k, v in sym2id.items()}
        cur = syls[i]
        if len(cur) != 3:
            continue
        nm1 = rev.get(cur[1], "")
        nm2 = rev.get(cur[2], "")
        # 「一」无论当前是 yī/yí/yì，都按规则重算（语料统计 94-100% 遵循）
        if not (nm1.startswith("i_") and nm2.startswith("i_")):
            continue
        if i + 1 >= len(syls) or not syls[i + 1] or len(syls[i + 1]) != 3:
            continue
        nx = syls[i + 1]
        a = rev.get(nx[1], "").split("_")[-1]
        b = rev.get(nx[2], "").split("_")[-1]
        nt = {"hh": 1, "lh": 2, "ll": 3, "hl": 4, "mm": 5}.get(a + b)
        if nt == 4:
            s1, s2 = "l", "h"
        elif nt in (1, 2, 3):
            s1, s2 = "h", "l"
        else:
            continue
        base1 = nm1.split("_")[0]
        base2 = nm2.split("_")[0]
        v1 = sym2id.get(base1 + "_" + s1)
        v2 = sym2id.get(base2 + "_" + s2)
        if v1 is not None:
            syls[i][1] = v1
        if v2 is not None:
            syls[i][2] = v2
    return syls


def _char_syllables(chars, wordset, lexicon, sym2id, table, words=None, crf_ctx=None):
    """每个汉字对应的音节 token 列表（用词表读音保证多音字正确）

    words: 可选，外部给定的分词结果（如官方断词模型输出）。
    """
    if words is None:
        words = segment_words("".join(chars), wordset if wordset else None, maxlen=6)
    crf_inv = None
    if crf_ctx is not None:
        try:
            from . import crf_zh as _crf
            _mc, crf_inv = _crf.load_char_models()
        except Exception:
            crf_inv = None
    syls = [None] * len(chars)
    lex_bounds = [None] * max(0, len(chars) - 1)
    pos = 0
    for wi2, w in enumerate(words):
        n = len(w)
        if pos + n > len(chars):
            return None
        groups = None
        tfix = load_token_fix().get(w) if USE_TOKEN_FIX else None
        if tfix and len(tfix) == n:
            groups = [[sym2id.get(x, sym2id.get("padding", 0)) for x in tri] for tri in tfix]
        trep = load_token_readings().get(w) if os.environ.get("OS_TOKEN_READ", "0") == "1" else None
        if trep and len(trep) == 3 * n:
            groups = [[sym2id.get(x, sym2id.get("padding", 0)) for x in trep[k * 3:k * 3 + 3]]
                      for k in range(n)]
        core = None
        ctx_mode = os.environ.get("OS_ZH_CTX", "0")
        if ctx_mode not in ("0", "") and (ctx_mode == "1" or (ctx_mode == "single" and n == 1) or (ctx_mode == "oov")):
            ctx_tbl = load_zh_ctx()
            prev_w = words[wi2 - 1] if wi2 > 0 else "^"
            next_w = words[wi2 + 1] if wi2 + 1 < len(words) else "$"
            rep = ctx_tbl.get("%s|%s|%s" % (prev_w, w, next_w))
            if ctx_mode == "oov" and w in lexicon:
                rep = None
            if rep and len(rep) == n:
                flat = []
                for k, tri in enumerate(rep):
                    if k:
                        flat.append("-")
                    flat.extend(tri)
                core = flat
        if core is None:
            core = lexicon.get(w)
        if groups is None and core:
            parsed = lexicon_groups(core, sym2id.get("-", 3), sym2id.get("br1", 25))
            flat = [g for grp in parsed for g in grp]
            if len(flat) == n:
                groups = flat
        if groups is None:
            py = [x[0] for x in pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)]
            py = apply_sandhi(py)
            if len(py) != n:
                return None
            groups = [[sym2id.get(sym, sym2id.get("padding", 0))
                       for sym in syl_to_symbols(p2, table)] for p2 in py]
            if FIX_MODE in ("oov", "oovtri"):
                for k in range(n):
                    gi = pos + k
                    prev_ch = chars[gi - 1] if gi > 0 else "^"
                    next_ch = chars[gi + 1] if gi + 1 < len(chars) else "$"
                    corr = fix_char_syl(w[k], prev_ch, next_ch, sym2id)
                    if corr:
                        groups[k] = corr
        multi_lex = (bool(core) or bool(trep) or bool(tfix)) and n > 1
        if multi_lex and w not in load_core_wordset() and w not in load_ctx_wordset():
            # 记录词内边界：扩展词（生僻/成语）按采集到的结构固定
            parsed_groups = lexicon_groups(core, sym2id.get("-", 3), sym2id.get("br1", 25))
            tok_of_char = []
            for gi, g in enumerate(parsed_groups):
                for _ in g:
                    tok_of_char.append(gi)
            for k in range(n - 1):
                if pos + k < len(lex_bounds):
                    if k + 1 < len(tok_of_char):
                        lex_bounds[pos + k] = 1 if tok_of_char[k] != tok_of_char[k + 1] else 0
        for k in range(n):
            syls[pos + k] = groups[k]
            if not multi_lex:
                gi = pos + k
                prev_ch = chars[gi - 1] if gi > 0 else "^"
                next_ch = chars[gi + 1] if gi + 1 < len(chars) else "$"
                if os.environ.get("OS_CHAR_CORR", "1") == "0":
                    corr = None
                else:
                    order = os.environ.get("OS_CORR_ORDER", "orig")
                    # 「一/不」的变调只在词首有效；词中/词尾保持本调（引擎行为）
                    if w[k] in ("一", "不") and k > 0:
                        corr = None
                    else:
                        fixes = [fix_char_syl(w[k], prev_ch, next_ch, sym2id),
                                 ctx_char_syl(w[k], prev_ch, next_ch, sym2id)]
                        if order == "iso_first":
                            fixes += [isolated_char_syl(w[k], sym2id), correct_char_syl(w[k], sym2id)]
                        elif order == "no_correct":
                            fixes += [isolated_char_syl(w[k], sym2id)]
                        else:
                            fixes += [correct_char_syl(w[k], sym2id), isolated_char_syl(w[k], sym2id)]
                        corr = None
                        for f in fixes:
                            if f:
                                corr = f
                                break
                    if corr is not None and os.environ.get("OS_LEX_FIRST", "0") == "1" and core:
                        lex_groups = lexicon_groups(core, sym2id.get("-", 3), sym2id.get("br1", 25))
                        flat = [g for grp in lex_groups for g in grp]
                        if len(flat) == n:
                            corr = flat[k]
                if corr:
                    syls[pos + k] = corr
        if crf_inv and n == 1 and w in crf_inv and crf_ctx is not None:
            try:
                from . import crf_zh as _crf
                _sc, _tk, _base = crf_ctx
                ti = _crf.char_token_index(_tk, _base + pos)
                if ti is not None:
                    rd = _crf.reading_for_char(w, _sc, _tk, ti)
                    if rd:
                        if os.environ.get("OS_CRF_DEBUG") == "1":
                            print("[CRF]", w, "->", rd, "ctx", _tk[max(0, ti - 2):ti + 3])
                        syls[pos] = [sym2id.get(x, sym2id.get("padding", 0)) for x in rd]
            except Exception:
                pass
        pos += n
    if pos != len(chars) or any(s is None for s in syls):
        return None
    if os.environ.get("OS_CHAR_RULES", "0") == "1":
        try:
            _apply_char_rules(chars, syls, sym2id)
        except Exception:
            pass
    if os.environ.get("OS_YI_SANDHI", "1") == "1":
        try:
            _yi_sandhi_syls(chars, syls, sym2id)
        except Exception:
            pass
    return syls, lex_bounds


_TAG_IDS_CACHE = {}


def _apply_yi_sandhi_ids(ids, sym2id):
    """「一」变调（按引擎实测规则）：仅当「一」位于词首（前一个 token 是词界标签）时变调。

    - 词首「一」+ 四声 → yí；词首「一」+ 一/二/三声 → yì；
    - 非词首（统一、第一、之一…）保持 yī。
    """
    rev = _TAG_IDS_CACHE.get("rev")
    if rev is None or len(rev) != len(sym2id):
        rev = {v: k for k, v in sym2id.items()}
        _TAG_IDS_CACHE["rev"] = rev
    tag_ids = {sym2id.get(x) for x in (BOS, END, BR0, BR1, BR2, BR3, BR4)}
    tag_ids.discard(None)
    sep_id = sym2id.get(SEP)

    def tone_suffix(tok):
        nm = rev.get(tok, "")
        if nm.endswith("_h"):
            return "h"
        if nm.endswith("_l"):
            return "l"
        if nm.endswith("_m"):
            return "m"
        return None

    def tone_of(syl):
        s1, s2 = tone_suffix(syl[1]), tone_suffix(syl[2])
        return {"hh": 1, "lh": 2, "ll": 3, "hl": 4, "mm": 5}.get((s1 or "") + (s2 or ""))

    # 音节列表：[(起始下标, 三个 token)]
    syls = []
    i = 0
    n = len(ids)
    while i < n:
        x = ids[i]
        if x in tag_ids or x == sep_id or x == sym2id.get(END):
            i += 1
            continue
        nm = rev.get(x, "")
        if nm.startswith("punc"):
            i += 1
            continue
        if i + 2 < n:
            syls.append((i, [ids[i], ids[i + 1], ids[i + 2]]))
            i += 3
        else:
            i += 1
    for k, (pos, syl) in enumerate(syls):
        if rev.get(syl[0], "") != "yi":
            continue
        if tone_of(syl) != 1:
            continue
        # 词首判定：前一个 token 是标签（或句首）
        if pos == 0:
            word_initial = True
        else:
            word_initial = ids[pos - 1] in tag_ids
        if not word_initial:
            continue
        if k + 1 >= len(syls):
            continue
        nt = tone_of(syls[k + 1][1])
        if nt == 4:
            s1, s2 = "l", "h"
        elif nt in (1, 2, 3):
            s1, s2 = "h", "l"
        else:
            continue
        nm1 = rev.get(syl[1], "")
        nm2 = rev.get(syl[2], "")
        b1 = nm1.split("_")[0]
        b2 = nm2.split("_")[0]
        v1 = sym2id.get(b1 + "_" + s1)
        v2 = sym2id.get(b2 + "_" + s2)
        if v1 is not None:
            ids[pos + 1] = v1
        if v2 is not None:
            ids[pos + 2] = v2
    return ids


def _apply_t3_ids(ids, sym2id):
    """三声连读变调（引擎在词间也生效）：T3+T3… → 前字 T2，最后一个保持 T3。"""
    rev = _TAG_IDS_CACHE.get("rev")
    if rev is None or len(rev) != len(sym2id):
        rev = {v: k for k, v in sym2id.items()}
        _TAG_IDS_CACHE["rev"] = rev
    tag_ids = {sym2id.get(x) for x in (BOS, END, BR0, BR1, BR2, BR3, BR4, SEP)}
    tag_ids.discard(None)
    punc_ids = {v for k, v in sym2id.items() if k.startswith("punc")}

    def tone(tok):
        nm = rev.get(tok, "")
        if nm.endswith("_h"):
            return "h"
        if nm.endswith("_l"):
            return "l"
        if nm.endswith("_m"):
            return "m"
        return None

    def t3(toks):
        return tone(toks[1]) == "l" and tone(toks[2]) == "l"

    def set_t2(pos, toks):
        nm2 = rev.get(toks[2], "")
        base2 = nm2.split("_")[0]
        new2 = sym2id.get(base2 + "_h")
        if new2 is not None:
            ids[pos + 2] = new2

    syl = []
    buf = []
    wid = -1
    for idx, x in enumerate(ids):
        if x in tag_ids or x in punc_ids:
            if buf:
                for s0 in range(0, len(buf) - 2, 3):
                    syl.append((buf[s0][0], [buf[s0][1], buf[s0 + 1][1], buf[s0 + 2][1]], wid))
                buf = []
            wid += 1
            continue
        buf.append((idx, x))
    if buf:
        for s0 in range(0, len(buf) - 2, 3):
            syl.append((buf[s0][0], [buf[s0][1], buf[s0 + 1][1], buf[s0 + 2][1]], wid))
    i = 0
    while i + 1 < len(syl):
        # 只在跨词边界时做三声连读（引擎行为：词内读音由词表决定）
        if (t3(syl[i][1]) and t3(syl[i + 1][1]) and syl[i][2] != syl[i + 1][2]):
            j = i
            while (j + 1 < len(syl) and t3(syl[j + 1][1])
                   and syl[j][2] != syl[j + 1][2]):
                set_t2(syl[j][0], syl[j][1])
                j += 1
            i = j + 1
        else:
            i += 1
    return ids


def _apply_sandhi_ids(ids, sym2id):
    """在最终 id 序列上做「一/不/三声」变调（引擎行为）。

    - 一 + 四声 → yí；一 + 一/二/三声 → yì
    - 不 + 四声 → bú（仅词首「不」；词尾「不」保持 bù，与引擎语料一致）
    - 三声连读：同一句内 T3+T3 → 前字变 T2（引擎会在词间做，如「一起使用」）
    """
    rev = _TAG_IDS_CACHE.get("rev")
    if rev is None or len(rev) != len(sym2id):
        rev = {v: k for k, v in sym2id.items()}
        _TAG_IDS_CACHE["rev"] = rev
    tag_ids = {sym2id.get(x) for x in (BOS, END, BR0, BR1, BR2, BR3, BR4, SEP)}
    tag_ids.discard(None)
    punc_ids = set()
    for k, v in sym2id.items():
        if k.startswith("punc"):
            punc_ids.add(v)

    def tone_of(tok):
        nm = rev.get(tok, "")
        if nm.endswith("_h"):
            return "h"
        if nm.endswith("_l"):
            return "l"
        if nm.endswith("_m"):
            return "m"
        return None

    # 切出音节流（记录 ids 起点、所属词编号）
    syl_pos = []
    buf = []
    word_id = -1
    for idx, x in enumerate(ids):
        if x in tag_ids or x in punc_ids:
            if buf:
                for s0 in range(0, len(buf) - 2, 3):
                    syl_pos.append((buf[s0][0], [buf[s0][1], buf[s0 + 1][1], buf[s0 + 2][1]],
                                    word_id))
                buf = []
            word_id += 1
            continue
        buf.append((idx, x))
    if buf:
        for s0 in range(0, len(buf) - 2, 3):
            syl_pos.append((buf[s0][0], [buf[s0][1], buf[s0 + 1][1], buf[s0 + 2][1]],
                            word_id))

    def set_tone(pos, toks, s1, s2):
        nm1 = rev.get(toks[1], "")
        nm2 = rev.get(toks[2], "")
        base1 = nm1.split("_")[0]
        base2 = nm2.split("_")[0]
        new1 = sym2id.get(base1 + "_" + s1)
        new2 = sym2id.get(base2 + "_" + s2)
        if new1 is not None:
            ids[pos + 1] = new1
        if new2 is not None:
            ids[pos + 2] = new2

    def tone_num(toks):
        n1 = tone_of(toks[1])
        n2 = tone_of(toks[2])
        if n1 and n2:
            if n1 == "h" and n2 == "h":
                return 1
            if n1 == "l" and n2 == "h":
                return 2
            if n1 == "l" and n2 == "l":
                return 3
            if n1 == "h" and n2 == "l":
                return 4
            if n1 == "m" and n2 == "m":
                return 5
        return None

    for k, (pos, toks, wid) in enumerate(syl_pos):
        nm0 = rev.get(toks[0], "")
        if nm0 == "yi" and tone_num(toks) == 1:
            if k + 1 < len(syl_pos):
                nt = tone_num(syl_pos[k + 1][1])
                if nt == 4:
                    set_tone(pos, toks, "l", "h")  # yí
                elif nt in (1, 2, 3):
                    set_tone(pos, toks, "h", "l")  # yì
        elif nm0 == "bu" and tone_num(toks) == 4:
            # 引擎只在「不」处于词首（且后面还有音节）时做变调
            if k + 1 < len(syl_pos) and syl_pos[k + 1][2] == wid:
                if tone_num(syl_pos[k + 1][1]) == 4:
                    set_tone(pos, toks, "l", "h")  # bú

    # 三声连读（同一句内、词间也生效；标点/标签会打断）
    if os.environ.get("OS_T3_CROSS", "1") == "1":
        i = 0
        while i + 1 < len(syl_pos):
            if tone_num(syl_pos[i][1]) == 3 and tone_num(syl_pos[i + 1][1]) == 3:
                j = i
                # 一连续串 T3：除最后一个外都变 T2
                while j + 1 < len(syl_pos) and tone_num(syl_pos[j + 1][1]) == 3:
                    set_tone(syl_pos[j][0], syl_pos[j][1], "l", "h")
                    j += 1
                i = j + 1
            else:
                i += 1
    return ids


def _text_to_ids_sentence(text, table, sym2id, lexicon=None):
    """单句：-bos- br0 <syl> - <syl> ... [br3 punc, br0] <syl> ... br4 [punc。] ~"""
    # 句尾的非句末标点（，；：等）：引擎把它当作句末处理
    #   引擎实测（评测集 599 句）：句尾「，」→ br4 + punc,；句尾「：」→ 仅 br4（无 punc token）
    tail_punc = None
    _t = text.rstrip()
    if _t and _t[-1] in "，、；：,;:":
        tail_punc = _t[-1]
        text = _t[:-1]
    segs = normalize_segments(text)
    crf_sent_chars = list(text)
    crf_tokens = None
    crf_cursor = 0
    if os.environ.get("OS_CRF_ZH", "1") == "1":
        try:
            from . import crf_zh as _crf
            crf_tokens = _crf.build_tokens(text)
        except Exception:
            crf_tokens = None
    ids = [sym2id[BOS], sym2id[BR0]]
    first = True
    ended = False
    from .symbols import resolve_lexicon
    lexicon = resolve_lexicon(lexicon, sym2id) if lexicon else (lexicon or {})
    wordset = set(lexicon) | (load_seg_words() if os.environ.get("OS_USE_OFFICIAL_WORDS", "1") == "1" else set())
    parens = []

    def emit_word(groups):
        nonlocal first
        if not first:
            ids.append(sym2id["br1"])
        for si, toks in enumerate(groups):
            if si > 0:
                ids.append(sym2id[SEP])
            ids.extend(toks)
        first = False

    for kind, val in segs:
        if kind == "n":
            py = [x[0] for x in pinyin(val, style=Style.TONE3, neutral_tone_with_five=True)]
            py = apply_sandhi(py)
            grp = []
            for p2 in py:
                grp.append([sym2id.get(sym, sym2id.get("padding", 0))
                            for sym in syl_to_symbols(p2, table)])
            if grp:
                emit_word(grp)
            continue
        chunks = re.split(r"([" + PUNCT_CHARS + "])", val)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in PUNCT or chunk == ",":
                key = PUNCT.get(chunk) or "punc,"
                is_end = chunk in ("。", "？", "！")
                if is_end:
                    ids.append(sym2id[BR4])
                    ids.append(sym2id.get(key, sym2id.get("punc?", 32)))
                    ended = True
                else:
                    ids.append(sym2id.get("br3", 39))
                    ids.append(sym2id.get(key, sym2id.get("punc?", 32)))
                    ids.append(sym2id[BR0])
                    first = True
                continue
            crf_base = None
            if crf_tokens is not None:
                _b = text.find(chunk, crf_cursor)
                if _b >= 0:
                    crf_base = _b
                    crf_cursor = _b + len(chunk)
            crf_ctx = ((crf_sent_chars, crf_tokens, crf_base)
                       if (crf_tokens is not None and crf_base is not None) else None)
            chars = [c for c in chunk if "\u4e00" <= c <= "\u9fff"]
            if not chars:
                continue
            text_cn = "".join(chars)
            model = load_seg_model()
            if model:
                parsed = _char_syllables(chars, wordset, lexicon, sym2id, table,
                                         crf_ctx=crf_ctx)
                if parsed is not None:
                    syls, lex_bounds = parsed
                    bounds = None
                    try:
                        from . import wb_model
                        _wb_words = [''.join(chars[s0:s0 + L0]) for s0, L0 in wb_model.segments(chars)]
                        bounds = wb_model.bounds_from_words(chars, _wb_words)
                        parsed = _char_syllables(chars, wordset, lexicon, sym2id, table,
                                                 words=_wb_words, crf_ctx=crf_ctx)
                        if parsed is not None:
                            syls, lex_bounds = parsed
                    except Exception:
                        bounds = None
                    if bounds is None:
                        from .segmodel import baseline_words, baseline_labels
                        bwords = baseline_words(chars)
                        blabs = baseline_labels(len(chars), bwords)
                        bounds = model.predict(chars, blabs, bwords)
                    for bi, lb in enumerate(lex_bounds):
                        if lb is not None and bi < len(bounds):
                            bounds[bi] = lb
                    if not first:
                        ids.append(sym2id["br1"])
                    for k, syl in enumerate(syls):
                        if k > 0:
                            ids.append(sym2id["br1"] if bounds[k - 1] else sym2id[SEP])
                        ids.extend(syl)
                    first = False
                    continue
            words = segment_words(text_cn, wordset if wordset else None, maxlen=6)
            merged = []
            i = 0
            while i < len(words):
                if (i + 1 < len(words) and len(words[i]) == 1 and words[i] == words[i + 1]
                        and words[i] not in lexicon):
                    merged.append(words[i] * 2)
                    i += 2
                else:
                    merged.append(words[i])
                    i += 1
            words = merged
            for w in words:
                core = lexicon.get(w)
                if core:
                    parsed = lexicon_groups(core, sym2id.get("-", 3), sym2id.get("br1", 25))
                    if parsed:
                        for g in parsed:
                            emit_word(g)
                        continue
                py = [x[0] for x in pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)]
                py = apply_sandhi(py)
                grp = []
                for p2 in py:
                    grp.append([sym2id.get(sym, sym2id.get("padding", 0))
                                for sym in syl_to_symbols(p2, table)])
                if grp:
                    emit_word(grp)
    if tail_punc is not None:
        ids.append(sym2id[BR4])
        key = PUNCT.get(tail_punc) or "punc,"
        if tail_punc not in ("：", ":"):
            ids.append(sym2id.get(key, sym2id.get("punc?", 32)))
        ended = True
    if not ended:
        ids += [sym2id[BR4]]
    ids += [sym2id[END]]
    if os.environ.get("OS_SANDHI_IDS", "0") == "1":
        try:
            ids = _apply_sandhi_ids(ids, sym2id)
        except Exception:
            pass
    if os.environ.get("OS_YI_SANDHI_IDS", "0") == "1":
        try:
            ids = _apply_yi_sandhi_ids(ids, sym2id)
        except Exception:
            pass
    if os.environ.get("OS_T3_IDS", "0") == "1":
        try:
            ids = _apply_t3_ids(ids, sym2id)
        except Exception:
            pass
    return ids


_SENT_END = "。！？!?"


def split_sentences(text):
    """按引擎行为切句：句末标点 。！？!? 结尾算一句（标点保留），余下部分单独成句。"""
    parts = []
    start = 0
    for m in re.finditer(r"[^。！？!?]*[。！？!?]", text):
        parts.append(m.group(0))
        start = m.end()
    if start < len(text):
        rest = text[start:]
        if rest:
            parts.append(rest)
    return [p for p in parts if p]


def text_to_ids(text, table, sym2id, lexicon=None):
    """引擎对每个句子单独调用一次 encoder，输出是各句序列首尾相接。

    因此这里按句切分，逐句生成 -bos- ... ~ 序列并拼接。
    """
    if not text:
        return []
    out = []
    for sent in split_sentences(text):
        if not any(ch.strip() for ch in sent):
            continue
        out.extend(_text_to_ids_sentence(sent, table, sym2id, lexicon))
    return out


def ids_to_symbols(ids, sym2id):
    rev = {v: k for k, v in sym2id.items()}
    return [rev.get(i, "?%d" % i) for i in ids]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--phones-dir", default=r"D:\Voice\Xiaoxiao_Female_zh-CN")
    ap.add_argument("--symbols", action="store_true")
    a = ap.parse_args()

    sym2id = load_symbols(os.path.join(a.phones_dir, "phones.txt"))
    table = load_table()
    lexicon = load_lexicon()
    ids = text_to_ids(a.text, table, sym2id, lexicon)
    print(json.dumps(ids, ensure_ascii=False))
    if a.symbols:
        print(" ".join(ids_to_symbols(ids, sym2id)))
