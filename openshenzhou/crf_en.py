#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crf_en.py —— 英文异读词 CRF 复刻（引擎 MSTTSLocEnUS.dat 内的 29 个 CRF 模型）

引擎侧行为（反编译 FUN_180119a28/FUN_180119cac + 运行期 hook 验证）：
  - 每个异读词对应一个 CRF 模型（2 标签 = 两种读音音素序列）；
  - 特征模板形如 U00:Token.PrevToken.PrevToken.PrevToken.CaseInsensitiveText，
    特征串为全大写 token 文本，'/' 连接多个上下文槽，虚拟边界 token 为
    '_B-2'、'_B-1'（前）与 '_B+1'..'_B+4'（后）；
  - 打分 = sum_f w[f, label]（B 段 float32，行主序），取 argmax。

本模块只负责「词 → 读音音素列表」的判定；模型从语音包 .dat 按需读取。
"""
import json
import os
import struct

from . import crf_zh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_MODELS = None
_DAT = None
_DAT_PATH = ""
_SLICES = {}
_FALLBACK = set()

VIRT_PRE = ("_B-2", "_B-1")
VIRT_POST = ("_B+1", "_B+2", "_B+3", "_B+4")


def _find_dat():
    global _DAT_PATH
    if _DAT_PATH and os.path.exists(_DAT_PATH):
        return _DAT_PATH
    env = os.environ.get("OS_EN_DAT")
    if env and os.path.exists(env):
        _DAT_PATH = env
        return _DAT_PATH
    roots = []
    v = os.environ.get("OS_VOICE_ROOT")
    if v:
        roots.append(v)
    if os.environ.get("TEMP"):
        roots.append(os.path.join(os.environ["TEMP"], "AllVoice"))
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        names = sorted(os.listdir(root))
        # 优先 en-US 语音包（其 MSTTSLocEnUS.dat 是该语种的权威数据）
        names.sort(key=lambda n: (("en-us" not in n.lower()), n))
        for name in names:
            p = os.path.join(root, name, "MSTTSLocEnUS.dat")
            if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
                _DAT_PATH = p
                return p
    return ""


def set_dat_path(path):
    global _DAT_PATH, _DAT, _SLICES
    if path and path != _DAT_PATH:
        _DAT_PATH = path
        _DAT = None
        _SLICES = {}


def load_models():
    global _MODELS
    if _MODELS is None:
        path = os.path.join(DATA, "en_crf_models.json")
        if not os.path.exists(path):
            _MODELS = []
        else:
            _MODELS = json.load(open(path, encoding="utf-8"))
    return _MODELS


def model_for_word(word):
    w = (word or "").lower().strip()
    for m in load_models():
        if m.get("word") == w:
            return m
    return None


def _dat():
    global _DAT
    if _DAT is None:
        p = _find_dat()
        if not p:
            raise FileNotFoundError("MSTTSLocEnUS.dat 未找到（设置 OS_VOICE_ROOT）")
        _DAT = open(p, "rb").read()
    return _DAT


def _features(m, tokens, ti):
    n = len(tokens)
    ext = list(VIRT_PRE) + [t.upper() for t in tokens] + list(VIRT_POST)
    ei = len(VIRT_PRE) + ti

    def get(j):
        if j < 0:
            return VIRT_PRE[0]
        if j >= len(ext):
            return VIRT_POST[0]
        return ext[j]

    out = []
    for tm in m["templates"]:
        expr = tm.split(":", 1)[1] if ":" in tm else tm
        parts = []
        for part in expr.split("/"):
            part = part.strip()
            if "PrevToken" in part and part.endswith("CaseInsensitiveText"):
                parts.append(get(ei - part.count("PrevToken")))
            elif "NextToken" in part and part.endswith("CaseInsensitiveText"):
                parts.append(get(ei + part.count("NextToken")))
            else:
                parts.append("")
        out.append(tm.split(":", 1)[0] + ":" + "/".join(parts))
    return out


def _label_phones(lab):
    parts = lab.split("%")
    out = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            out.append((p + parts[i + 1]).lower())
            i += 2
        elif p:
            out.append(p.lower())
            i += 1
        else:
            i += 1
    return out


def score_word(model, tokens, ti):
    dat = _dat()
    if model["idx"] not in _SLICES:
        A = dat[model["A_off"]:model["A_off"] + model["lenA"]]
        B = dat[model["B_off"]:model["B_off"] + model["lenB"]]
        _SLICES[model["idx"]] = (crf_zh.Trie(A), struct.unpack("<%df" % (len(B) // 4), B))
    trie, vals = _SLICES[model["idx"]]
    L = model["L"]
    nf = len(vals) // max(L, 1)
    scores = [0.0] * L
    for fs in _features(model, tokens, ti):
        fid, _ = trie.lookup(fs)
        if fid is None or fid < 0 or fid >= nf:
            continue
        for l in range(L):
            scores[l] += vals[fid * L + l]
    return scores


def select(word, tokens, ti, min_margin=None):
    """对 tokens[ti] 的异读词给出读音音素列表；无模型或置信不足返回 None"""
    if min_margin is None:
        min_margin = _MARGIN
    m = model_for_word(word)
    if m is None:
        return None
    if not (0 <= ti < len(tokens)):
        return None
    try:
        scores = score_word(m, tokens, ti)
    except Exception:
        return None
    order = sorted(range(len(scores)), key=lambda l: -scores[l])
    if len(scores) > 1 and min_margin:
        margin = scores[order[0]] - scores[order[1]]
        if margin < min_margin:
            return None
    return _label_phones(m["labels"][order[0]])


# 低置信度回退阈值：margin 小于该值时不启用 CRF 覆盖（交给既有规则）。
# 文档记录见 findings：引擎在近平分场景存在额外消歧，CRF 直接 argmax 会在
# 少数近分句上偏离；阈值由独立探针语料标定，避免破坏既有规则的正确判定。
_MARGIN = float(os.environ.get("OS_CRF_EN_MARGIN", "0.5"))
