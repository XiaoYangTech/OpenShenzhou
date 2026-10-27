#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crf_trie.py —— 离线实现 CVocabTrie 的字符串→特征ID 查找（A 段）

依据反编译：
  FUN_18011967c : varint 解码（u16 表 + 字节流）
  FUN_180119850 : descend（解码节点索引→读 node[char,flags]，可选读 node 值）
  FUN_18011975c : skip siblings（按 flags 调整字节流/指针）
  FUN_1801196c8 : commit（计算 state+0x18）
  FUN_180119a28 : 主循环（逐字符下行，返回累计 ID 与长度）
"""
import struct
import sys


class Trie:
    def __init__(self, A, base_off=0):
        self.A = A
        self.o = base_off
        # 头部字段（A 起始）
        self.nfeat = struct.unpack_from('<I', A, self.o + 20)[0]
        self.count_b = struct.unpack_from('<I', A, self.o + 24)[0]     # 0x37c0 数组个数
        self.node_count = struct.unpack_from('<I', A, self.o + 28)[0]  # 节点数
        self.count_a = struct.unpack_from('<I', A, self.o + 36)[0]     # 0x35bc 数组个数
        self.stream_off = struct.unpack_from('<I', A, self.o + 40)[0]  # 0x3934
        self.stream_size = struct.unpack_from('<I', A, self.o + 44)[0]
        self.tbl1 = self.o + 0x30    # inner+8
        self.tbl2 = self.o + 0x34    # inner+0x18
        self.tbl3 = self.o + 0x38    # inner+0x20
        self.nodes = self.o + 0x3c   # inner+0x28
        self.arr_a = self.nodes + self.node_count * 4          # 0x35bc
        self.arr_b = self.arr_a + self.count_a * 4             # 0x37c0
        self.stream = self.o + self.stream_off                 # 0x3934
        self.base = self.stream                                # inner+0x48

    # ---------- 基础读取 ----------
    def u16(self, off):
        return struct.unpack_from('<H', self.A, off)[0]

    def u32(self, off):
        return struct.unpack_from('<I', self.A, off)[0]

    def node(self, idx):
        off = self.nodes + idx * 4
        return self.u16(off), self.u16(off + 2)

    # ---------- varint 解码 ----------
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
        # *out = val + b ; consumed = i + 1
        return val + b, i + 1

    # ---------- descend ----------
    def descend(self, st):
        p = st['p10']
        idx, used = self.decode(self.tbl1, p)
        ch, flags = self.node(idx)
        st['ch'] = ch
        flags &= 0xFFFF
        st['flags'] = flags
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
        st['val8'] = v
        st['p10'] = q

    # ---------- skip ----------
    def skip(self, st):
        mask = st['flags'] & 0x1D0
        if mask:
            p = st['p10']
            if mask == 0x10:
                idx, used = self.decode(self.tbl3, p)
                p2 = p + used
                q = st['p28']
                if q == 0:
                    st['p28'] = p2
                    q = p2
                st['p28'] = q + self.u32(self.arr_b + idx * 4)
            elif mask == 0x50:
                st['p28'] = p
                p2 = p
            elif mask == 0x90:
                tbl = self.tbl2
                i = 0
                b = self.A[p]
                base = self.u16(tbl)
                q = p
                if base <= b:
                    while True:
                        tbl += 2
                        q += 1
                        i += 1
                        b = (b - base) * 256 + self.A[q]
                        base = self.u16(tbl)
                        if base > b:
                            break
                p2 = p + (i + 1)
            elif mask == 0x110:
                p2 = p + 3
            else:
                p2 = p
            st['p10'] = p2

    # ---------- commit ----------
    def commit(self, st):
        mask = st['flags'] & 0x1C0
        if mask == 0:
            v = st['p28']
        elif mask == 0x40:
            v = st['p10']
        elif mask == 0x80:
            idx, used = self.decode(self.tbl2, st['p10'])
            v = self.u32(self.arr_a + idx * 4) + self.base
        elif mask == 0x100:
            p = st['p10']
            v = (self.A[p] | (self.A[p + 1] << 8) | (self.A[p + 2] << 16)) + self.base
        else:
            v = 0
        st['p18'] = v

    # ---------- 主查找 ----------
    def lookup(self, text):
        s = [ord(c) for c in text]
        st = {'ch': 0, 'flags': 0, 'val8': 0, 'p10': 0, 'p18': 0, 'p28': 0}
        i = 0
        st['flags'] = 0x10
        st['p18'] = self.base
        iVar6 = -1
        iVar10 = -1
        lVar8 = 0
        n = len(s)
        while True:
            if (st['flags'] & 0x10) == 0:
                break
            st['p28'] = 0
            st['p10'] = st['p18']
            self.descend(st)
            uVar5 = st['p10']
            self.skip(st)
            ch = s[i] if i < n else 0
            iVar7 = iVar6 + 1
            uVar3 = st['p10']
            if (st['flags'] & 1) == 0:
                iVar7 = iVar6
            while True:
                st['p10'] = uVar5
                iVar6 = iVar7
                if st['ch'] == ch:
                    break
                iVar7 = iVar6 + st['val8']
                st['p10'] = uVar3
                if st['flags'] & 2:
                    if (st['flags'] & 0x200) == 0:
                        if (st['flags'] & 0x20) == 0:
                            iVar10 = -1
                            return -1, 0
                        st['p10'] = st.get('p20', 0)
                    st['p28'] = 0
                self.descend(st)
                uVar5 = st['p10']
                if st['flags'] & 1:
                    iVar7 += 1
                self.skip(st)
                uVar3 = st['p10']
            self.commit(st)
            i += 1
            iVar7 = iVar6
            lVar4 = i
            if (st['flags'] & 1) == 0:
                iVar7 = iVar10
                lVar4 = lVar8
            lVar8 = lVar4
            iVar10 = iVar7
            if i == n:
                break
        if iVar10 == -1:
            return -1, 0
        return iVar10, lVar8


def load_model79(dat_path):
    """读取语音包 .dat 中第 79 个 CRF 段的 A 区（按 find_sections 顺序）"""
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    import parse_crf_models as P
    data = open(dat_path, 'rb').read()
    secs = P.find_sections(data)
    sec = secs[79]
    return data[sec['A_off']:sec['A_off'] + sec['lenA']], sec


if __name__ == '__main__':
    import json
    import os
    from parse_crf_models import find_sections
    dat = sys.argv[1]
    A = None
    data = open(dat, 'rb').read()
    secs = find_sections(data)
    sec = secs[79]
    A = data[sec['A_off']:sec['A_off'] + sec['lenA']]
    t = Trie(A)
    print('nodes', t.node_count, 'nfeat', t.nfeat, 'stream', hex(t.stream_off), t.stream_size)
    pairs = json.load(open(sys.argv[2], encoding='utf-8'))
    ok = 0
    for s, fid in pairs['pairs']:
        got, ln = t.lookup(s)
        flag = 'OK ' if got == fid else 'BAD'
        if got == fid:
            ok += 1
        if flag == 'BAD' and ok + 5 > len(pairs['pairs']):
            pass
        print(flag, repr(s), 'expect', fid, 'got', got, 'len', ln)
    print('match %d/%d' % (ok, len(pairs['pairs'])))
