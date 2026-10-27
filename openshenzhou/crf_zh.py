#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crf_zh.py —— 引擎中文多音字 CRF 的离线复刻

数据来源：
  - data/zh_crf_models.json      : 87 个 CRF 模型的标签/模板/偏移（来自语音包 .dat）
  - data/zh_crf_model_chars.json : 模型→汉字 映射
  - <语音包>/MSTTSLocZhCN.dat     : A 段（trie 数据）与 B 段（权重）按偏移读取

实现依据（反编译 embedded.tts）：
  FUN_18011967c varint 解码 / FUN_180119850 descend / FUN_18011975c skip
  FUN_1801196c8 commit / FUN_180119a28 主循环
"""
import json
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_MODELS = None
_MODEL_CHARS = None
_CHAR_MODELS = None
_DAT_PATH = os.environ.get("OS_ZH_DAT", "")
_TRUST = None
_TRUST_TH = float(os.environ.get("OS_CRF_ACC", "0.9"))
_TRUST_MIN_N = int(os.environ.get("OS_CRF_MIN_N", "3"))
_MARGIN = float(os.environ.get("OS_CRF_MARGIN", "0"))
_DAT = None
_SLICES = {}


def set_dat_path(path):
    """由 TTS 层传入语音包里的 MSTTSLocZhCN.dat 路径"""
    global _DAT_PATH, _DAT, _SLICES
    if path and path != _DAT_PATH:
        _DAT_PATH = path
        _DAT = None
        _SLICES = {}


def load_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = json.load(open(os.path.join(DATA, "zh_crf_models.json"),
                                 encoding="utf-8"))
    return _MODELS


def load_char_models():
    global _MODEL_CHARS, _CHAR_MODELS
    if _MODEL_CHARS is None:
        _MODEL_CHARS = json.load(open(os.path.join(DATA, "zh_crf_model_chars.json"),
                                      encoding="utf-8"))
        inv = {}
        for idx, o in _MODEL_CHARS.items():
            if o.get("char"):
                inv.setdefault(o["char"], []).append(int(idx))
        _CHAR_MODELS = inv
    return _MODEL_CHARS, _CHAR_MODELS


def _find_dat():
    cands = []
    if _DAT_PATH:
        cands.append(_DAT_PATH)
    env = os.environ.get("OS_ZH_DAT")
    if env:
        cands.append(env)
    root = os.environ.get("TEMP", "")
    if root:
        base = os.path.join(root, "AllVoice")
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                cands.append(os.path.join(base, name, "MSTTSLocZhCN.dat"))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return ""


def load_trust():
    """每模型在引擎语料上的验证准确率（validate_crf.py 生成）"""
    global _TRUST
    if _TRUST is None:
        try:
            _TRUST = json.load(open(os.path.join(DATA, "zh_crf_model_trust.json"),
                                    encoding="utf-8"))
        except Exception:
            _TRUST = {}
    return _TRUST


def _model_margin(idx):
    t = load_trust().get(str(idx)) or {}
    return float(t.get("margin", _MARGIN))


def _model_trusted(idx):
    t = load_trust().get(str(idx))
    if not t:
        return False
    return t.get("n", 0) >= _TRUST_MIN_N and t.get("acc", 0.0) >= _TRUST_TH


def _dat():
    global _DAT, _DAT_PATH
    if _DAT is None:
        path = _find_dat()
        if not path:
            raise FileNotFoundError("MSTTSLocZhCN.dat 未找到（设置 OS_ZH_DAT）")
        _DAT_PATH = path
        _DAT = open(path, "rb").read()
    return _DAT


def model_slices(idx):
    """返回 (A, B) 两个 bytes 片段（带缓存）"""
    if idx in _SLICES:
        return _SLICES[idx]
    dat = _dat()
    m = load_models()[idx]
    A = dat[m["A_off"]:m["A_off"] + m["lenA"]]
    B = dat[m["B_off"]:m["B_off"] + m["lenB"]]
    _SLICES[idx] = (A, B)
    return A, B


class Trie:
    """CVocabTrie 的离线实现（A 段）"""

    def __init__(self, A):
        self.A = A
        self.nfeat = struct.unpack_from("<I", A, 20)[0]
        self.count_b = struct.unpack_from("<I", A, 24)[0]
        self.node_count = struct.unpack_from("<I", A, 28)[0]
        self.count_a = struct.unpack_from("<I", A, 36)[0]
        self.stream_off = struct.unpack_from("<I", A, 40)[0]
        self.stream_size = struct.unpack_from("<I", A, 44)[0]
        self.tbl1 = 0x30
        self.tbl2 = 0x34
        self.tbl3 = 0x38
        self.nodes = 0x3C
        self.arr_a = self.nodes + self.node_count * 4
        self.arr_b = self.arr_a + self.count_a * 4
        self.stream = self.stream_off
        self.base = self.stream

    def u16(self, off):
        return struct.unpack_from("<H", self.A, off)[0]

    def u32(self, off):
        return struct.unpack_from("<I", self.A, off)[0]

    def node(self, idx):
        off = self.nodes + idx * 4
        return self.u16(off), self.u16(off + 2)

    def decode(self, tbl, p):
        b = self.A[p]
        val = 0
        i = 0
        while True:
            base = self.u16(tbl + i * 2)
            if base > b:
                break
            val += base
            i += 1
            p += 1
            b = (b - base) * 256 + self.A[p]
        return val + b, i + 1

    def descend(self, st):
        p = st["p10"]
        idx, used = self.decode(self.tbl1, p)
        st["ch"], flags = self.node(idx)
        st["flags"] = flags
        v = 0
        q = p + used
        if flags & 4:
            b0 = self.A[q]
            if b0 < 0xC0:
                if b0 > 0x7F:
                    v = ((b0 & 0x7F) << 8) | self.A[q + 1]
                    q += 2
                else:
                    v = b0
                    q += 1
            else:
                v = (((self.A[q + 1] & 0x7F) | ((b0 & 0x3F) << 7)) << 8) | self.A[q + 2]
                q += 3
        st["val8"] = v
        st["p10"] = q

    def skip(self, st):
        mask = st["flags"] & 0x1D0
        if not mask:
            return
        p = st["p10"]
        if mask == 0x10:
            idx, used = self.decode(self.tbl3, p)
            p2 = p + used
            q = st["p28"]
            if q == 0:
                st["p28"] = p2
                q = p2
            st["p28"] = q + self.u32(self.arr_b + idx * 4)
        elif mask == 0x50:
            st["p28"] = p
            p2 = p
        elif mask == 0x90:
            i = 0
            b = self.A[p]
            base = self.u16(self.tbl2)
            q = p
            if base <= b:
                while True:
                    i += 1
                    q += 1
                    b = (b - base) * 256 + self.A[q]
                    base = self.u16(self.tbl2 + i * 2)
                    if base > b:
                        break
            p2 = p + (i + 1)
        elif mask == 0x110:
            p2 = p + 3
        else:
            p2 = p
        st["p10"] = p2

    def commit(self, st):
        mask = st["flags"] & 0x1C0
        if mask == 0:
            v = st["p28"]
        elif mask == 0x40:
            v = st["p_desc"]
        elif mask == 0x80:
            idx, _ = self.decode(self.tbl2, st["p_desc"])
            v = self.u32(self.arr_a + idx * 4) + self.base
        elif mask == 0x100:
            p = st["p_desc"]
            v = ((self.A[p] << 16) | (self.A[p + 1] << 8) | self.A[p + 2]) + self.base
        else:
            v = 0
        st["p18"] = v

    def lookup(self, text):
        """返回 (feature_id, consumed_len)；未找到返回 (-1, 0)"""
        s = [ord(c) for c in text]
        n = len(s)
        st = {"ch": 0, "flags": 0x10, "val8": 0, "p10": self.base,
              "p18": self.base, "p28": 0, "p20": 0, "p_desc": self.base}
        acc = -1
        best_id = -1
        best_len = 0
        i = 0
        guard_total = 0
        while True:
            saved = acc
            if (st["flags"] & 0x10) == 0:
                break
            st["p28"] = 0
            st["p10"] = st["p18"]
            self.descend(st)
            st["p_desc"] = st["p10"]
            self.skip(st)
            ch = s[i] if i < n else 0
            acc = acc + 1
            if (st["flags"] & 1) == 0:
                acc = saved
            guard = 0
            while st["ch"] != ch:
                guard += 1
                guard_total += 1
                if guard > 200000 or guard_total > 400000:
                    return -1, 0
                acc += st["val8"]
                if st["flags"] & 2:
                    if (st["flags"] & 0x200) == 0:
                        if (st["flags"] & 0x20) == 0:
                            return -1, 0
                        st["p10"] = st["p20"]
                    st["p28"] = 0
                self.descend(st)
                st["p_desc"] = st["p10"]
                if st["flags"] & 1:
                    acc += 1
                self.skip(st)
            st["p10"] = st["p_desc"]
            self.commit(st)
            i += 1
            if st["flags"] & 1:
                best_id = acc
                best_len = i
            if i >= n:
                break
        return best_id, best_len


def _label_parts(model):
    """返回 [[symbol,...], ...]；非 3 段标签返回 None"""
    out = []
    for lab in model["labels"]:
        parts = [x.lower() for x in lab.split("%")]
        if not parts or any(not p for p in parts):
            return None
        out.append(parts)
    return out


def _weights(model, B):
    L = model["L"]
    nf = model["nfeat"]
    return struct.unpack("<%df" % (len(B) // 4), B), L, nf


VIRT_PRE = ("_B", "_B-2", "_B-1")
VIRT_POST = ("_B+1", "_B+2", "_B+3", "_B+4")


def _features(model, tok_text, sent_chars, tokens, ti):
    """按模板生成特征串列表（token 上下文用扩展虚拟 token 表索引）"""
    tmpls = model["templates"]
    t_start, t_end = tokens[ti][1], tokens[ti][2]
    sents = sent_chars
    n = len(sents)
    ext = list(VIRT_PRE) + [t[0] for t in tokens] + list(VIRT_POST)
    ei = len(VIRT_PRE) + ti

    def prev_char(k):
        p = t_start - 1 - k
        return sents[p] if 0 <= p < n else "^"

    def next_char(k):
        p = t_end + k
        return sents[p] if 0 <= p < n else "^"

    def prev_tok(k):
        j = ei - k
        return ext[j] if 0 <= j < len(ext) else "_B"

    def next_tok(k):
        j = ei + k
        return ext[j] if 0 <= j < len(ext) else "_B+2"

    out = []
    for tm in tmpls:
        expr = tm.split(":", 1)[1] if ":" in tm else tm
        parts = []
        for part in expr.split("/"):
            part = part.strip()
            if part.startswith("Token.PreviousCharacters.GetCharacter("):
                k = int(part.rsplit("(", 1)[1].rstrip(")"))
                parts.append(prev_char(k))
            elif part.startswith("Token.NextCharacters.GetCharacter("):
                k = int(part.rsplit("(", 1)[1].rstrip(")"))
                parts.append(next_char(k))
            elif "PrevToken" in part and part.endswith("CaseInsensitiveText"):
                k = part.count("PrevToken")
                parts.append(prev_tok(k))
            elif "NextToken" in part and part.endswith("CaseInsensitiveText"):
                k = part.count("NextToken")
                parts.append(next_tok(k))
            else:
                parts.append("")
        out.append(tm.split(":", 1)[0] + ":" + "/".join(parts))
    return out


def reading_for_char(ch, sent_chars, tokens, ti):
    """对单字 token 给出 CRF 判定读音（symbol 名称列表）或 None"""
    models, inv = load_char_models()
    idxs = inv.get(ch)
    if not idxs:
        return None
    for idx in idxs:
        if not _model_trusted(idx):
            continue
        model = load_models()[idx]
        if model["T"] != 24:
            continue
        labels = _label_parts(model)
        if labels is None:
            continue
        try:
            A, B = model_slices(idx)
            trie = Trie(A)
            feats = _features(model, tokens[ti][0], sent_chars, tokens, ti)
            vals, L, nf = _weights(model, B)
        except Exception:
            continue
        if L != len(labels):
            continue
        scores = [0.0] * L
        hit = 0
        for fs in feats:
            try:
                fid, _ = trie.lookup(fs)
            except Exception:
                continue
            if fid is None or fid < 0 or fid >= nf:
                continue
            hit += 1
            base = fid * L
            for l in range(L):
                scores[l] += vals[base + l]
        if hit == 0:
            continue
        order = sorted(range(L), key=lambda l: -scores[l])
        margin = scores[order[0]] - scores[order[1]] if L > 1 else 99.0
        if margin < _model_margin(idx):
            continue
        return list(labels[order[0]])
    return None


def build_tokens(text):
    """把整句切成 token 列表 [(text,start,end)]，标点单字成 token，CJK 连续段用断词模型"""
    tokens = []
    i = 0
    n = len(text)
    punct = set("，。！？、；：,.;!?…—－《》「」『』（）()【】 \t")
    while i < n:
        c = text[i]
        if c in punct:
            tokens.append((c, i, i + 1))
            i += 1
            continue
        j = i
        while j < n and text[j] not in punct:
            j += 1
        run = text[i:j]
        try:
            from . import wb_model
            words = ["".join(run[s0:s0 + L0]) for s0, L0 in wb_model.segments(list(run))]
        except Exception:
            words = list(run)
        pos = i
        for w in words:
            if not w:
                continue
            tokens.append((w, pos, pos + len(w)))
            pos += len(w)
        i = j
    return tokens


def char_token_index(tokens, pos):
    for k, (_, s, e) in enumerate(tokens):
        if s <= pos < e:
            if e - s == 1:
                return k
            return None
    return None
