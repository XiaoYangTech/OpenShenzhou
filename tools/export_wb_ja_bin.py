#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_wb_ja_bin.py —— 把日语断词权重导出成 Node SDK 用的二进制

格式（与中文 wb_bilstm.bin 相同）：
  "OSWB0001" | u32 rows | u32 cols | emb(rows*cols) | Wf(128*256) | Wb(128*256)
  | bf(256) | bb(256) | Wo(128*4) | bo(4)       全部 float32 小端
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from openshenzhou import wb_model_ja as W  # noqa: E402


def main():
    w = W._load()
    rows = w["emb"].shape[0]
    cols = w["emb"].shape[1]
    out = os.path.join(HERE, "data", "wb_ja_bilstm.bin")
    with open(out, "wb") as f:
        f.write(b"OSWB0001")
        f.write(np.uint32(rows).tobytes())
        f.write(np.uint32(cols).tobytes())
        for key in ("emb", "w_fwd", "w_bwd", "b_fwd", "b_bwd", "w_out", "b_out"):
            arr = np.ascontiguousarray(w[key].reshape(-1), dtype="<f4")
            f.write(arr.tobytes())
            print(key, arr.shape)
    print("written", out, os.path.getsize(out))


if __name__ == "__main__":
    main()
