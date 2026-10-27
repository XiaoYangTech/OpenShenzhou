#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_crf.py —— 用引擎语料校验离线 CRF 读音预测

对语料中每个「单字 token 且该字有映射模型」的位置，
比较 CRF argmax 读音与引擎实际读音（来自 seg_*.jsonl）。
输出 data/zh_crf_model_trust.json：{model_idx: {n, ok, acc, char}}
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, HERE)
from openshenzhou import crf_zh as C  # noqa: E402
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402

TAGS = {17, 18, 25, 39, 68}
MAX_SENT = int(os.environ.get("OS_VALID_SENT", "6000"))


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
    mp, inv = C.load_char_models()
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "char": None})
    feat_cache = {}
    n_sent = 0
    n_ctx = 0
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
            ok = True
            for x in ids:
                if x in TAGS or x in (1, 2, 3):
                    continue
                name = id2sym.get(x)
                if not name or name.startswith("punc") or name == "padding":
                    continue
                syl.append(name)
            if len(syl) != 3 * len(text):
                continue
            n_sent += 1
            if n_sent > MAX_SENT:
                break
            chars = list(text)
            try:
                tokens = C.build_tokens(text)
            except Exception:
                continue
            for ti, (t, s, e) in enumerate(tokens):
                if e - s != 1 or t not in inv:
                    continue
                idxs = [i for i in inv[t] if mp[str(i)].get("reason") != "conflict"]
                if not idxs:
                    continue
                eng = tuple(syl[s * 3:s * 3 + 3])
                for idx in idxs:
                    model = C.load_models()[idx]
                    labels = C._label_parts(model)
                    if labels is None:
                        continue
                    # 使用带缓存的预测
                    key = (idx, text, s)
                    if key in feat_cache:
                        pred = feat_cache[key]
                    else:
                        pred = _predict(idx, model, chars, tokens, ti, labels)
                        feat_cache[key] = pred
                    if pred is None:
                        continue
                    st = stats[idx]
                    st["char"] = t
                    st["n"] += 1
                    n_ctx += 1
                    if tuple(pred) == eng:
                        st["ok"] += 1
                    pred2 = C._LAST[1]
                    if tuple(pred2) == eng:
                        st["ok2"] = st.get("ok2", 0) + 1
            if n_sent >= MAX_SENT:
                break
        if n_sent >= MAX_SENT:
            break
    out = {}
    for idx, st in sorted(stats.items(), key=lambda kv: -(kv[1]["n"])):
        n = st["n"]
        acc = st["ok"] / n if n else 0.0
        acc2 = st.get("ok2", 0) / n if n else 0.0
        out[idx] = {"char": st["char"], "n": n, "ok": st["ok"], "acc": round(acc, 4),
                    "ok2": st.get("ok2", 0), "acc2": round(acc2, 4)}
        print("model %3s %s  n=%-5d acc=%.3f acc2=%.3f" % (idx, st["char"], n, acc, acc2))
    json.dump(out, open(os.path.join(HERE, "data", "zh_crf_model_trust.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved; sentences", n_sent, "contexts", n_ctx)


def _predict(idx, model, chars, tokens, ti, labels):
    try:
        A, B = C.model_slices(idx)
        trie = C.Trie(A)
        feats = C._features(model, tokens[ti][0], chars, tokens, ti)
        vals, L, nf = C._weights(model, B)
    except Exception:
        return None
    if L != len(labels):
        return None
    scores = [0.0] * L
    scores2 = [0.0] * L
    hit = 0
    cache = getattr(C, "_FEAT_CACHE", None)
    if cache is None:
        cache = {}
        C._FEAT_CACHE = cache
    for fs in feats:
        k = (idx, fs)
        fid = cache.get(k)
        if fid is None:
            try:
                fid, _ = trie.lookup(fs)
            except Exception:
                fid = -1
            cache[k] = fid
        if fid is None or fid < 0 or fid >= nf:
            continue
        hit += 1
        for l in range(L):
            scores[l] += vals[fid * L + l]
            scores2[l] += vals[l * nf + fid]
    if hit == 0:
        return None
    b1 = max(range(L), key=lambda l: scores[l])
    b2 = max(range(L), key=lambda l: scores2[l])
    C._LAST = (list(labels[b1]), list(labels[b2]))
    return list(labels[b1])


if __name__ == "__main__":
    main()
