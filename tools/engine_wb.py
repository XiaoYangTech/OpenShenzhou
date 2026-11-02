"""engine_wb.py —— 引擎 CBlstmWordBreaker 完整复刻（LSTM 类别 + FUN_1802543e0 规则后处理）。

LSTM 逐字类别由 wb_model.classes(full_text) 给出（已与引擎逐位一致）；
本模块复刻引擎 Process 的类别修正/分词规则（来自 FUN_1802543e0 反编译），
输出引擎最终的词区间列表 [(start, len)]（字符下标）。

规则要点（lang = 0x804 zh-CN）：
  字符类型 type(c): 1=汉字, 2=数字, 3=字母, 4/5=标点/符号两类
  逐字规则会改写 class[i]（0=B 1=M 2=E 3=S），并在 class∈{2,3} 时结词。
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("OS_SDK_ROOT"):
    sys.path.insert(0, os.environ["OS_SDK_ROOT"])
sys.path.insert(0, HERE)

from openshenzhou import wb_model

PUNC = {}


def _bit(mask, v, mod=0x40):
    return (mask >> (v & (mod - 1))) & 1


def char_type(c, lang=0x804):
    """FUN_180254894 → 1..5"""
    # type1: FUN_1802548ec
    if lang in (0x804, 0xe04):
        if 0x4E00 <= c <= 0x9FCC:
            return 1
    elif ((lang - 0x404) & 0xF6FF) == 0 and lang != 0x504:
        # 未使用（zh-CN 走上面分支）
        pass
    # type2: FUN_180254ab8
    if ((c - 0x30) <= 9 or (c + 0xDBA0) <= 0x3B or (c + 0xDB16) <= 0x15
            or (c + 0xF0) <= 9):
        return 2
    # type3: FUN_180254afc
    t3 = False
    if 0x41 <= c <= 0x41 + 0x39:
        v = c - 0x41
        t3 = _bit(0x3FFFFFF03FFFFFF, v)
    if not t3 and 0xC0 <= c <= 0xC0 + 0x36:
        v = c - 0xC0
        t3 = _bit(0x7FFFFFFF7FFFFF, v)
    if not t3 and not ((c - 0xF8) > 0x157 and (c + 0xDB64) > 0x4D
                      and (c + 0xDF) > 0x19 and (c + 0xBF) > 0x19):
        t3 = True
    if t3:
        return 3
    # type4/5: FUN_180254b80
    t5 = False
    if not ((c - 0x20) <= 0x20 or not _bit(0x1FC00FFFF, c - 0x20)):
        t5 = True
    elif not ((c - 0x5B) <= 0x23 or not _bit(0xF0000003F, c - 0x5B)):
        t5 = True
    elif ((c - 0xA1) > 0x1E and c != 0xD7 and c != 0xF7 and (c + 0xDFEE) > 0x15
          and ((c + 0xDFD0) > 0x34 or not _bit(0x1C7FFFFFFFFFFFF, c + 0xDFD0))
          and (c + 0xCFFF) > 0x2E and c != 0x30FB
          and (c - 0x1B0) > 0x1B and ((c - 0xFF) > 0x3F or not _bit(0xFC000000FE007FFF, c - 0xFF))):
        t5 = True
    return 5 if t5 else 4


def _separator(c):
    """FUN_180254848"""
    return (c < 0x20 or (0x80 <= c < 0xA0) or (c - 0x2000) % 0x10000 < 0xC
            or c in (0x20, 0x3000, 0x7F))


def _trim_word(chars, start, length):
    """FUN_180254e1c：去掉词首尾分隔符"""
    end = start + length
    while end > start and _separator(ord(chars[start])):
        start += 1
    while end > start and _separator(ord(chars[end - 1])):
        end -= 1
    return start, end - start


def word_spans(text, classes=None):
    """复刻 FUN_1802543e0：返回引擎最终词区间 [(start,len)]（字符下标）。"""
    n = len(text)
    if n == 0:
        return []
    if classes is None:
        classes = wb_model.classes(list(text))
    h = [int(x) for x in classes]
    chars = list(text)
    spans = []
    i = 0
    word_len = 0
    word_start = 0
    prev_type = 0
    while i < n:
        if i < n - 1 and h[i + 1] in (1, 2):
            if h[i] == 2:
                h[i] = 1
            elif h[i] == 3:
                h[i] = 0
        ctype = char_type(ord(chars[i]))
        if _separator(ord(chars[i])) and word_len != 0:
            s, L = _trim_word(chars, word_start, word_len)
            if L > 0:
                spans.append((s, L))
            word_start = word_start + word_len
            word_len = 0
            h[i] = 3
        # ---- 按字符类型修正 class[i]（FUN_1802543e0 主循环） ----
        if word_len == 0:
            if i < n - 1:
                if ctype == 4:
                    c0 = ord(chars[i])
                    if (chars[i] == chars[i + 1]
                            and ((c0 - 0x2D) < 2 or c0 == 0x2026 or (c0 + 0xDFEC) < 2)):
                        h[i] = 0
                elif ((ctype - 2) & 0xFFFFFFFC) == 0 and ctype != 4:
                    h[i] = 0
        elif ctype == 1:
            if prev_type != 1:
                s, L = _trim_word(chars, word_start, word_len)
                if L > 0:
                    spans.append((s, L))
                word_start += word_len
                word_len = 0
        elif ctype == 2:
            if prev_type != 4:
                h[i] = 1 if prev_type == 2 else 0
            else:
                if ord(chars[i - 1]) not in (0x25, 0x2E):
                    h[i] = 0
                else:
                    h[i] = 1
        elif ctype == 3:
            if prev_type == 4:
                if ord(chars[i - 1]) in (0x27, 0x2D) and (
                        ('A' <= chars[i] <= 'Z') or ('a' <= chars[i] <= 'z')):
                    h[i] = 1
                else:
                    h[i] = 0
            else:
                h[i] = 1 if prev_type == 3 else 0
        elif ctype == 5:
            if prev_type == 4:
                if ord(chars[i - 1]) in (0x27, 0x2D):
                    h[i] = 1
                else:
                    h[i] = 0
            else:
                h[i] = 1 if prev_type == 5 else 0
        else:  # ctype == 4
            if prev_type == 1:
                h[i] = 0
            elif prev_type == 3:
                if ord(chars[i]) in (0x27, 0x2D) and i < n - 1:
                    if char_type(ord(chars[i + 1])) == 3:
                        h[i] = 1
                    else:
                        h[i] = 3
                else:
                    h[i] = 3
            elif prev_type == 2:
                if chars[i] == '%':
                    h[i] = 2
                elif chars[i] == '.' and i < n - 1 and char_type(ord(chars[i + 1])) == 2:
                    h[i] = 1
                else:
                    h[i] = 3
            else:  # prev_type == 4 or 0
                if prev_type != 4 or chars[i] != chars[i - 1] if i > 0 else True:
                    h[i] = 0
                elif ord(chars[i]) in (0x2D, 0x2E, 0x2026) or (ord(chars[i]) + 0xDFEC) < 2:
                    h[i] = 1
                else:
                    h[i] = 0
        # 遇到新的词首（class 0/3）且上一词未结束：先结词
        if h[i] in (0, 3) and word_len != 0:
            s, L = _trim_word(chars, word_start, word_len)
            if L > 0:
                spans.append((s, L))
            word_start += word_len
            word_len = 0
        word_len += 1
        # class E(2)/S(3) 结词
        if h[i] in (2, 3):
            s, L = _trim_word(chars, word_start, word_len)
            if L > 0:
                spans.append((s, L))
            word_start += word_len
            word_len = 0
        i += 1
        prev_type = ctype
    if word_len != 0:
        s, L = _trim_word(chars, word_start, word_len)
        if L > 0:
            spans.append((s, L))
    return spans


def boundaries(text, classes=None):
    """返回中文汉字之间的边界集合（char 下标，按汉字序列编号）。"""
    spans = word_spans(text, classes)
    # 汉字序列
    han = [(i, c) for i, c in enumerate(text) if '\u4e00' <= c <= '\u9fff' or c == '〇']
    pos_of_han = {orig: k for k, (orig, c) in enumerate(han)}
    bounds = set()
    # 每个 span 起点：若该位置是汉字且不是第一个汉字 → 边界
    for s, L in spans:
        if s in pos_of_han and pos_of_han[s] > 0:
            bounds.add(pos_of_han[s])
    return bounds


if __name__ == '__main__':
    import json
    texts = sys.argv[1:]
    for t in texts:
        print(repr(t), word_spans(t))
