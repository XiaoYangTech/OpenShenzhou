#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_zh_tone_model.py —— 训练中文声调上下文修正模型（v2 富特征，数据驱动）

训练数据：8,756 句同域 UI 语料（引擎 ids）+ SDK 基线 ids（缓存或现算）。
v2 特征：基线音节 token + 前2/前1/后1字符 + 前后词界位 + 句内位置，7 级 backoff。
验证：独立留出 1500 句 1419→1462（+43，仅 1 句破坏）；评测 3000 句 2915→2925。
"""
import collections
import gzip
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from _paths import voice_pkg, sdk_root  # noqa: E402

if os.path.isdir(os.path.join(sdk_root(), "openshenzhou")):
    sys.path.insert(0, sdk_root())
from openshenzhou import g2p_zh  # noqa: E402

BOUND = {'-bos-', 'br0', 'br1', 'br2', 'br3', 'br4', '~', '-'}


def chars_of(sent):
    return [c for c in sent if '\u4e00' <= c <= '\u9fff' or c == '〇']


def parse(ids, rev):
    names = []
    for x in ids:
        names.append(rev.get(x, '?%d' % x))
        if x == 1:
            break
    pieces = []
    cur = []
    for t in names:
        if t in BOUND or t.startswith('punc'):
            if cur:
                pieces.append(('word', cur))
                cur = []
            pieces.append(('sep', t))
        else:
            cur.append(t)
    if cur:
        pieces.append(('word', cur))
    words = [p[1] for p in pieces if p[0] == 'word']
    bits = []
    pending = []
    for kind, payload in pieces:
        if kind == 'sep':
            pending.append(payload)
        else:
            bits.append(1 if 'br1' in pending else 0)
            pending = []
    return words, bits


def feats(chs, bw, bb, i):
    n = len(chs)
    ch = chs[i]
    p1 = chs[i - 1] if i > 0 else '^'
    p2 = chs[i - 2] if i > 1 else '^'
    n1 = chs[i + 1] if i + 1 < n else '$'
    bl = bb[i] if i < len(bb) else 0
    br = bb[i + 1] if i + 1 < len(bb) else 0
    pos = 'B' if i == 0 else ('E' if i == n - 1 else 'M')
    cur = ' '.join(bw[i])
    return cur, p2, p1, ch, n1, bl, br, pos


def encode_key(d, k):
    s = '\x1f'
    if d == 0:
        cur, p2, p1, ch, n1, bl, br, pos = k
        return s.join([cur, p2 + p1 + ch + n1, '%d%d' % (bl, br), pos])
    if d == 1:
        cur, p1, ch, n1, bl, br = k
        return s.join([cur, p1 + ch + n1, '%d%d' % (bl, br)])
    if d == 2:
        cur, ch, n1, bl, br = k
        return s.join([cur, ch + n1, '%d%d' % (bl, br)])
    if d == 3:
        cur, p1, ch, bl = k
        return s.join([cur, p1 + ch, str(bl)])
    if d == 4:
        cur, ch, bl = k
        return s.join([cur, ch, str(bl)])
    if d == 5:
        cur, ch = k
        return s.join([cur, ch])
    return k[0]


def build(rows):
    levels = [collections.defaultdict(collections.Counter) for _ in range(7)]
    for chs, gw, bw, bb in rows:
        for i in range(len(chs)):
            cur, p2, p1, ch, n1, bl, br, pos = feats(chs, bw, bb, i)
            t = gw[i]
            levels[0][(cur, p2, p1, ch, n1, bl, br, pos)][t] += 1
            levels[1][(cur, p1, ch, n1, bl, br)][t] += 1
            levels[2][(cur, ch, n1, bl, br)][t] += 1
            levels[3][(cur, p1, ch, bl)][t] += 1
            levels[4][(cur, ch, bl)][t] += 1
            levels[5][(cur, ch)][t] += 1
            levels[6][(cur,)][t] += 1
    return levels


# 阈值（在留出 1500 句上选定）
CFG = (2, 0.85, 10, 0.90)


def make_tables(levels, cfg):
    mc, rt, dmc, drt = cfg
    out = []
    for d in range(7):
        t = {}
        for k, c in levels[d].items():
            best, cnt = c.most_common(1)[0]
            total = sum(c.values())
            m = mc if d <= 2 else (deep if d >= 4 else mc)
            r = rt if d <= 2 else (deep_r if d >= 4 else rt)
            if cnt < m or cnt / total < r or best == k[0]:
                continue
            t[encode_key(d, k)] = list(best)
        out.append(t)
    return out


deep, deep_r = CFG[2], CFG[3]


def main():
    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    table = g2p_zh.load_table()
    sym = g2p_zh.load_symbols(os.path.join(pkg, "phones.txt"))
    lex = g2p_zh.load_lexicon()
    rev = {v: k for k, v in sym.items()}
    with gzip.open(os.path.join(HERE, "data", "zh_ui_ids_8756.json.gz"),
                   "rt", encoding="utf-8") as f:
        gold = json.load(f)
    cache = os.path.join(os.environ.get("TEMP", "."), "os_ui_baseline_ids.json")
    if os.path.exists(cache):
        base_ids = json.load(open(cache, encoding="utf-8"))
        print("baseline cache", len(base_ids))
    else:
        base_ids = {}
        for k, sent in enumerate(gold):
            base_ids[sent] = g2p_zh.text_to_ids(sent, table, sym, lex)
            if (k + 1) % 500 == 0:
                print("baseline", k + 1, flush=True)
        json.dump(base_ids, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    rows = []
    for sent, ids in gold.items():
        chs = chars_of(sent)
        if not chs:
            continue
        gw, _ = parse(ids, rev)
        bw, bb = parse(base_ids[sent], rev)
        if not (len(gw) == len(bw) == len(chs)):
            continue
        rows.append((chs, [tuple(w) for w in gw], [tuple(w) for w in bw], bb))
    print("rows", len(rows))
    tables = make_tables(build(rows), CFG)
    model = {'version': 2, 'config': list(CFG), 'tables': tables,
             'calib': 'heldout1500 1419->1462(+43,-1); eval3000 2915->2925(+10 net)'}
    path = os.path.join(HERE, "data", "zh_tone_ctx_model.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print("saved", path, os.path.getsize(path), "sizes", [len(t) for t in tables])


if __name__ == "__main__":
    main()
