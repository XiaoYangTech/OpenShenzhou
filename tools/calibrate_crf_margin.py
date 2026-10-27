#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_crf_margin.py —— 标定「CRF 置信度门限」：引擎可能只在 margin 足够大时才用 CRF 读音

对每个语料上下文记录 (margin, CRF 预测读音, 无 CRF 基线读音, 引擎真实读音)，
统计不同阈值下 (CRF vs 基线回退) 的准确率，输出 data/zh_crf_margin.json
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, HERE)
import openshenzhou.g2p_zh as gz  # noqa: E402
from openshenzhou import crf_zh as C  # noqa: E402
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402

TAGS = {17, 18, 25, 39, 68}
MAX_SENT = int(os.environ.get("OS_CALIB_SENT", "12000"))


def load_symbols(path):
    sym = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line or ":" not in line or line.startswith("#"):
            continue
        n, i = line.rsplit(":", 1)
        sym[int(i)] = n
    return sym


def main():
    pkg = _voice_pkg("Xiaoxiao_Female_zh-CN")
    C.set_dat_path(os.path.join(pkg, "MSTTSLocZhCN.dat"))
    id2sym = load_symbols(os.path.join(pkg, "phones.txt"))
    sym2id = {v: k for k, v in id2sym.items()}
    table = gz.load_table()
    lex = gz.load_lexicon()
    mp, inv = C.load_char_models()
    models = C.load_models()
    rows = []
    feat_cache = {}
    n = 0
    files = sorted(glob.glob(os.path.join(ENGINE, "data", "seg_*.jsonl")))
    for fn in files:
        if "seg_test" in fn:
            continue
        for line in open(fn, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("text") or ""
            ids = r.get("ids") or []
            if not text or len(text) > 40:
                continue
            if not all(0x4E00 <= ord(c) <= 0x9FCC for c in text):
                continue
            syl = []
            for x in ids:
                if x in TAGS or x in (1, 2, 3):
                    continue
                nm = id2sym.get(x)
                if not nm or nm.startswith("punc") or nm == "padding":
                    continue
                syl.append(nm)
            if len(syl) != 3 * len(text):
                continue
            n += 1
            if n > MAX_SENT:
                break
            chars = list(text)
            try:
                tokens = C.build_tokens(text)
            except Exception:
                continue
            for ti, (t, s, e) in enumerate(tokens):
                if e - s != 1 or t not in inv:
                    continue
                eng = tuple(syl[s * 3:s * 3 + 3])
                for idx in inv[t]:
                    if mp[str(idx)].get("reason") in ("conflict", "label-pair"):
                        continue
                    model = models[idx]
                    labels = C._label_parts(model)
                    if not labels:
                        continue
                    A, B = C.model_slices(idx)
                    trie = C.Trie(A)
                    vals, L, nf = C._weights(model, B)
                    feats = C._features(model, t, chars, tokens, ti)
                    sc = [0.0] * L
                    hit = 0
                    for fs in feats:
                        k = (idx, fs)
                        fid = feat_cache.get(k)
                        if fid is None:
                            try:
                                fid, _ = trie.lookup(fs)
                            except Exception:
                                fid = -1
                            feat_cache[k] = fid
                        if not (0 <= fid < nf):
                            continue
                        hit += 1
                        for l in range(L):
                            sc[l] += vals[fid * L + l]
                    if hit == 0:
                        continue
                    order = sorted(range(L), key=lambda l: -sc[l])
                    margin = sc[order[0]] - sc[order[1]]
                    pred = tuple(labels[order[0]])
                    rows.append({"m": round(margin, 4), "pred": pred, "eng": eng,
                                 "char": t, "idx": idx, "ok": pred == eng})
            if n > MAX_SENT:
                break
        if n > MAX_SENT:
            break
    print("contexts", len(rows))
    base_ok = sum(0 for _ in rows)
    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for th in grid:
        acc = sum(1 for r in rows if (r["ok"] and r["m"] >= th) or (not r["ok"] and r["m"] < th)) / len(rows)
        used = sum(1 for r in rows if r["m"] >= th)
        print("TH=%.2f  CRF-used=%-5d  per-token acc=%.4f" % (th, used, acc))
    json.dump(rows, open(os.path.join(HERE, "data", "zh_crf_margin_rows.json"),
                         "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
