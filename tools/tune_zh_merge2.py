#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_zh_merge2.py —— 用大规模引擎语料校准中文断词「合并率」参数

训练统计：tts-engine/data/seg_*.jsonl（排除 eval_cap.jsonl 与留出集 seg_B）
调参集：data/eval/zh_seg_cap*.jsonl（新抓的 3000 句引擎实测）
留出测试：seg_B 抽样
评估指标：词边界完全一致率（不涉及读音，纯断词后处理质量）

用法::
    OS_SDK_ROOT=<python-sdk> python tools/tune_zh_merge2.py
"""
import glob
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "tts-engine"))
sys.path.insert(0, os.path.join(ENGINE, "tools"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402

CACHE = os.path.join(ENGINE, "data", "wb_classes_cache.json")
COUNTS_CACHE = os.path.join(os.environ.get("TEMP", HERE), "zh_merge_counts_v3.json")


def load_sym(pkg):
    m = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        m[n] = int(i)
    return m


def decode_bounds(text, ids, sym):
    """→ (chars_ok?, engine boundary set over CJK char gaps)"""
    rev = {v: k for k, v in sym.items()}
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    bounds = set()
    pos = 0
    syl = 0
    started = False
    for x in ids:
        nm = rev.get(x)
        if nm is None:
            continue
        if nm in ("-bos-", "br0", "br2", "br3", "br4", "~") or nm.startswith("punc"):
            continue
        if nm == "br1":
            if started:
                pos += syl + 1
                if 0 < pos < len(chars):
                    bounds.add(pos)
            syl = 0
            started = False
            continue
        if nm == "-":
            syl += 1
            started = True
        else:
            started = True
    if started:
        pos += syl + 1
    if pos != len(chars):
        return None, None
    return chars, bounds


def r1_segs(cls):
    segs = []
    st = 0
    for j in range(1, len(cls)):
        if cls[j] in (0, 3):
            segs.append((st, j - st))
            st = j
    segs.append((st, len(cls) - st))
    return segs


def merge_with_rates(text, segs, words, rates, th, max_len=12):
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(segs):
            best = None
            for k in range(len(segs) - 1, i, -1):
                end = segs[k][0] + segs[k][1]
                cand = text[segs[i][0]:end]
                if len(cand) > max_len or cand not in words:
                    continue
                rate = rates.get(cand, 0.0)
                if rate >= th:
                    best = k
                    break
            if best is not None:
                out.append((segs[i][0], segs[best][0] + segs[best][1] - segs[i][0]))
                i = best + 1
                changed = True
            else:
                out.append(segs[i])
                i += 1
        segs = out
    return segs


def main():
    from openshenzhou import wb_model

    pkg = voice_pkg("Xiaoxiao_Female_zh-CN")
    sym = load_sym(pkg)
    official = set(json.load(open(os.path.join(HERE, "data", "official_words_zh.json"),
                                  encoding="utf-8")))

    # 1) 统计 m/s（可缓存）
    if os.path.exists(COUNTS_CACHE):
        d = json.load(open(COUNTS_CACHE, encoding="utf-8"))
        m_cnt = Counter(d["m"])
        s_cnt = Counter(d["s"])
        eng_words = Counter(d.get("eng", {}))
        print("counts loaded", len(m_cnt), len(s_cnt), "engine words", len(eng_words))
    else:
        import wb_infer3 as M  # noqa: E402
        try:
            cache = json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            cache = {}
        Wd = M.load()

        def cls_of(text):
            c = cache.get(text)
            if c is None:
                pred = M.infer(Wd, text, "chunk")
                c = [int(x) for x in pred[1:1 + len(text)]]
                cache[text] = c
            return c

        m_cnt, s_cnt = Counter(), Counter()
        eng_words = Counter()
        files = sorted(glob.glob(os.path.join(ENGINE, "data", "seg_*.jsonl")))
        n_sent = 0
        for fn in files:
            base = os.path.basename(fn)
            if base in ("eval_cap.jsonl", "seg_B.jsonl"):
                continue
            for line in open(fn, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r["text"]
                if not (4 <= len(text) <= 40):
                    continue
                if not all("\u4e00" <= c <= "\u9fff" for c in text):
                    continue
                labs = M.engine_labels(r["ids"])
                if len(labs) != len(text):
                    continue
                eb = set()
                for i, l in enumerate(labs):
                    if l in "ES" and i + 1 < len(labs):
                        eb.add(i + 1)
                cls = cls_of(text)
                segs = r1_segs(cls)
                n_sent += 1
                # 引擎自己的词表：引擎输出中的 token 词
                st = 0
                for b in sorted(eb):
                    w = text[st:b]
                    if 2 <= len(w) <= 12:
                        eng_words[w] += 1
                    st = b
                if st < len(text):
                    w = text[st:]
                    if 2 <= len(w) <= 12:
                        eng_words[w] += 1
                for i in range(len(segs)):
                    for k in range(i + 1, min(len(segs), i + 5)):
                        end = segs[k][0] + segs[k][1]
                        cand = text[segs[i][0]:end]
                        if cand not in eng_words:
                            continue
                        inner = [segs[j][0] for j in range(i + 1, k + 1)]
                        if all(b not in eb for b in inner):
                            m_cnt[cand] += 1
                        else:
                            s_cnt[cand] += 1
        print("train sentences", n_sent, "cands", len(m_cnt), len(s_cnt))
        json.dump({"m": dict(m_cnt), "s": dict(s_cnt), "eng": dict(eng_words)},
                  open(COUNTS_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        json.dump(dict(eng_words), open(os.path.join(HERE, "data", "zh_engine_words.json"),
                                        "w", encoding="utf-8"), ensure_ascii=False)
        print("engine words", len(eng_words))
        try:
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass

    def make_rates(alpha):
        rates = {}
        for w in set(m_cnt) | set(s_cnt):
            m = m_cnt.get(w, 0)
            s = s_cnt.get(w, 0)
            denom = m + s + 2 * alpha
            rates[w] = ((m + alpha) / denom) if denom > 0 else 0.0
        return rates

    # 2) 评估集：seg_B 切三段（调参 1500 / 留出验证 1500 / 最终评测 3000），互不重叠
    import random
    rows_all = []
    with open(os.path.join(ENGINE, "data", "seg_B.jsonl"), encoding="utf-8") as fh:
        allb = [json.loads(l) for l in fh if l.strip()]
    random.Random(11).shuffle(allb)
    for r in allb:
        chars, eng = decode_bounds(r["text"], r["ids"], sym)
        if chars and len(chars) >= 4:
            rows_all.append((r["text"], chars, eng, r))
        if len(rows_all) >= 6000:
            break
    rows_heldout = rows_all[:1500]
    tune_rows = rows_all[1500:3000]
    final_rows = rows_all[3000:6000]
    json.dump([x[3] for x in final_rows],
              open(os.path.join(HERE, "data", "eval", "zh_large_eval.jsonl"), "w",
                   encoding="utf-8"), ensure_ascii=False)
    print("tune rows", len(tune_rows), "heldout", len(rows_heldout),
          "final eval", len(final_rows))

    cache_cls = {}

    def eval_rows(rows, rates, th):
        exact = 0
        for row in rows:
            text, chars, eng = row[0], row[1], row[2]
            key = "".join(chars)
            cls = cache_cls.get(key)
            if cls is None:
                cls = wb_model.classes(chars)
                cache_cls[key] = cls
            segs = r1_segs(cls)
            out = merge_with_rates("".join(chars), segs, official, rates, th)
            our = set(st for st, _ in out[1:])
            if our == eng:
                exact += 1
        return exact / max(1, len(rows))

    best = None
    for alpha in (0.1, 0.3, 0.5, 1.0):
        rates = make_rates(alpha)
        for th in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            acc_t = eval_rows(tune_rows, rates, th)
            acc_h = eval_rows(rows_heldout, rates, th)
            print("alpha=%.1f th=%.1f tune=%.4f heldout=%.4f" % (alpha, th, acc_t, acc_h))
            score = acc_h
            if best is None or score > best[0]:
                best = (score, alpha, th, acc_t, acc_h)
    print("BEST", best)
    score, alpha, th, acc_t, acc_h = best
    rates = make_rates(alpha)
    out = {"alpha": alpha, "threshold": th, "rates": rates}
    json.dump(out, open(os.path.join(HERE, "data", "zh_merge_rates.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    print("saved data/zh_merge_rates.json words", len(rates))


if __name__ == "__main__":
    main()
