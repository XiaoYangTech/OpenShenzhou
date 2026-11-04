"""build_zh_tone_model.py —— 训练中文声调上下文校正模型（数据驱动）

训练数据：8,756 句同域 UI 语料（引擎 ids）+ SDK 基线 ids（缓存或现算）。
模型：以「基线音节 token + 前2/前1/后1字符」为键的 5 级 backoff 查表；
      只保留满足 (count>=5, ratio>=0.90) 且与基线不同的覆盖项。
留出验证（1500 句）：基线 1419/1500 → 1448/1500（+29）；评测 3000：2915 → 2922（+7, 0 破坏）。
"""
import collections
import gzip
import json
import os
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
CFG = {'min_count': 5, 'ratio': 0.90}


def parse(ids, rev):
    words = []
    cur = []
    for x in ids:
        t = rev.get(x, '?%d' % x)
        if x == 1:
            break
        if t in BOUND or t.startswith('punc'):
            if cur:
                words.append(tuple(cur))
                cur = []
        else:
            cur.append(t)
    if cur:
        words.append(tuple(cur))
    return words


def chars_of(sent):
    return [c for c in sent if '\u4e00' <= c <= '\u9fff' or c == '〇']


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

    levels = [collections.defaultdict(collections.Counter) for _ in range(5)]
    used = 0
    for sent, ids in gold.items():
        chs = chars_of(sent)
        syl = parse(ids, rev)
        bw = parse(base_ids[sent], rev)
        if not chs or len(chs) != len(syl) or len(bw) != len(chs):
            continue
        used += 1
        n = len(chs)
        for i, ch in enumerate(chs):
            p1 = chs[i - 1] if i > 0 else '^'
            p2 = chs[i - 2] if i > 1 else '^'
            n1 = chs[i + 1] if i + 1 < n else '$'
            t = syl[i]
            cur = bw[i]
            levels[0][(cur, p2, p1, ch, n1)][t] += 1
            levels[1][(cur, p1, ch, n1)][t] += 1
            levels[2][(cur, ch, n1)][t] += 1
            levels[3][(cur, ch)][t] += 1
            levels[4][(cur,)][t] += 1
    print("used sentences", used, "tables", [len(t) for t in levels])

    out_tables = []
    for d in range(5):
        t = {}
        for key, c in levels[d].items():
            best, cnt = c.most_common(1)[0]
            total = sum(c.values())
            if cnt < CFG["min_count"] or cnt / total < CFG["ratio"] or best == key[0]:
                continue
            cur_s = ' '.join(key[0])
            if d in (0, 1, 2):
                kstr = cur_s + '\x1f' + ''.join(key[1:])
            elif d == 3:
                kstr = cur_s + '\x1f' + key[1]
            else:
                kstr = cur_s
            t[kstr] = list(best)
        out_tables.append(t)
        print("level", d, "override entries", len(t))
    model = {'version': 1, 'config': CFG, 'tables': out_tables,
             'corpus': 'zh_ui_ids_8756', 'calib': 'heldout1500: 1419->1448; eval3000: 2915->2922'}
    path = os.path.join(HERE, "data", "zh_tone_ctx_model.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print("saved", path, os.path.getsize(path))


if __name__ == "__main__":
    main()
