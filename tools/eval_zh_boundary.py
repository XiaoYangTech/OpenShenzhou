#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_zh_boundary.py —— 中文词界单独评测（只比较词边界，忽略读音/音素差异）。

与 eval_first_sentence.py / eval_zh_large.py 的「逐 id 一致」互补：
本工具把引擎 golden 与 SDK 输出都投影成「中文汉字之间的词边界集合」，
用于分离「词界错误」与「读音（多音字）错误」两类残差。

用法::
    python tools/eval_zh_boundary.py data/eval/zh_eval_599.jsonl
    python tools/eval_zh_boundary.py data/eval/zh_large_eval.jsonl
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])

from _paths import voice_pkg  # noqa: E402

PUNCT = '，。！？、；：,.!?;:「」『』（）()《》〈〉—“”‘’…．·－-　 '


def first_sentence(ids):
    out = []
    for x in ids:
        out.append(x)
        if x == 1:  # '~'
            break
    return out


def content_bounds(ids, rev):
    """返回 (内容 token 边界集合, 内容 token 数)。

    引擎 id 序列里 br1 出现在「下一个词」的 token 组之前，因此与
    `eval_first_sentence` 的遍历方式保持一致：pending='br1' 时在下一个
    内容 token 处记账。
    """
    bounds = set()
    content = 0
    pending = False
    for x in ids:
        name = rev.get(x)
        if name in ('-bos-', 'br0'):
            continue
        if name == '~':
            break
        if name and name.startswith('punc'):
            pending = True
            continue
        if name in ('br1', '-'):
            pending = (name == 'br1')
            continue
        if name in ('br2', 'br3', 'br4'):
            pending = True
            continue
        if pending and content > 0:
            bounds.add(content)
        content += 1
        pending = False
    return bounds, content


def is_clean_cn(sent):
    """只看纯中文 + 标点/空格的句子，避免数字/字母/乱码行干扰词界统计。"""
    for c in sent:
        if '\u4e00' <= c <= '\u9fff' or c == '〇':
            continue
        if c in PUNCT:
            continue
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus')
    ap.add_argument('--show', type=int, default=5, help='打印前 N 个差异句')
    args = ap.parse_args()

    from openshenzhou import g2p_zh as gz
    pkg = voice_pkg('Xiaoxiao_Female_zh-CN')
    sym = gz.load_symbols(os.path.join(pkg, 'phones.txt'))
    rev = {v: k for k, v in sym.items()}
    table = gz.load_table()
    lex = gz.load_lexicon()

    rows = [json.loads(l) for l in open(args.corpus, encoding='utf-8') if l.strip()]
    total = ok = clean_total = clean_ok = 0
    shown = 0
    for r in rows:
        sents = gz.split_sentences(r['text'])
        sent = sents[0] if sents else r['text']
        cn = [c for c in sent if '\u4e00' <= c <= '\u9fff' or c == '〇']
        if len(cn) < 2:
            continue
        n = len(cn)
        gb, gc = content_bounds(first_sentence(r['ids']), rev)
        pb, pc = content_bounds(first_sentence(gz.text_to_ids(sent, table, sym, lex)), rev)
        if gc != 3 * n or pc != 3 * n:
            continue
        gbc = {c // 3 for c in gb}
        pbc = {c // 3 for c in pb}
        good = gbc == pbc
        total += 1
        ok += int(good)
        if is_clean_cn(sent):
            clean_total += 1
            clean_ok += int(good)
        elif not good and shown < args.show:
            shown += 1
            print('差异:', sent)
            print('  gold:', '|'.join(
                ''.join(cn[a:b]) for a, b in
                list(zip([0] + sorted(gbc), sorted(gbc) + [n]))))
            print('  sdk :', '|'.join(
                ''.join(cn[a:b]) for a, b in
                list(zip([0] + sorted(pbc), sorted(pbc) + [n]))))

    def pct(a, b):
        return '%.1f%%' % (100.0 * a / b) if b else '-'

    print('中文词界一致（全部）: %d/%d = %s' % (ok, total, pct(ok, total)))
    print('中文词界一致（纯中文句）: %d/%d = %s'
          % (clean_ok, clean_total, pct(clean_ok, clean_total)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
