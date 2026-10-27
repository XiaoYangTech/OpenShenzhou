#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""projection.py —— 按引擎分块调度做 mel 投影（复刻逐块量化）

引擎 decoder 分块运行，最后的 DynamicQuantizeLinear 是按「块内张量」现算 scale 的。
整段推理只有一个 scale，会带来约 0.5% 的残差。
这里改成：整段跑出 hidden（input.52），再按引擎的块大小逐块量化投影。
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _find_projection(m):
    """通用定位 mel 投影：最后一个 MatMulInteger（int8 权重）+ 前面的 DynamicQuantizeLinear"""
    inits = {i.name: i for i in m.graph.initializer}
    mm = [n for n in m.graph.node if n.op_type == "MatMulInteger"]
    if not mm:
        raise RuntimeError("找不到 MatMulInteger")
    node = mm[-1]
    w_name = node.input[1]
    zp_name = node.input[3] if len(node.input) > 3 else None
    base = w_name.replace("_quantized", "")
    if zp_name is None:
        zp_name = base + "_zero_point"
    sc_name = base + "_scale"
    if sc_name not in inits or zp_name not in inits:
        raise RuntimeError("找不到投影 scale/zero_point: " + w_name)
    W = numpy_helper.to_array(inits[w_name]).astype(np.int32) - int(numpy_helper.to_array(inits[zp_name]))
    W_scale = float(numpy_helper.to_array(inits[sc_name]))
    # hidden 张量 = 喂给 DynamicQuantizeLinear 的那个
    dq = [n for n in m.graph.node if n.op_type == "DynamicQuantizeLinear"]
    hidden_name = dq[-1].input[0] if dq else None
    # bias：最后输出链上的 Add 的 1 维初始化器
    bias = None
    for n in reversed(m.graph.node):
        if n.op_type == "Add":
            for x in n.input:
                if x in inits and numpy_helper.to_array(inits[x]).ndim == 1:
                    bias = numpy_helper.to_array(inits[x]).astype(np.float32)
                    break
        if bias is not None:
            break
    if bias is None:
        raise RuntimeError("找不到投影 bias")
    return hidden_name, W, W_scale, bias


def load_projection(decoder_onnx_path, hidden_onnx_path):
    """从 decoder 模型里取出 mel 投影权重；生成暴露 hidden 的变体"""
    import os
    m = onnx.load(decoder_onnx_path)
    hidden_name, W, W_scale, bias = _find_projection(m)
    if not os.path.exists(hidden_onnx_path):
        m.graph.node.append(helper.make_node("Identity", [hidden_name], ["hidden_out"], name="expose_hidden"))
        m.graph.output.append(helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT, None))
        onnx.save(m, hidden_onnx_path)
    return W, W_scale, bias


def project(hidden, W, W_scale, bias):
    """hidden: [T,256] → [T,80]，复刻 DynamicQuantizeLinear + MatMulInteger"""
    mn, mx = float(hidden.min()), float(hidden.max())
    scale = (mx - mn) / 255.0 if mx > mn else 1.0
    zp = max(0, min(255, int(round(-mn / scale)) if scale > 0 else 0))
    q = np.clip(np.round(hidden / scale) + zp, 0, 255).astype(np.int32)
    acc = (q - zp) @ W
    return (acc.astype(np.float32) * (scale * W_scale) + bias[None, :]).astype(np.float32)


def decoder_chunk_sizes(T):
    """复刻引擎 decoder 的块调度：首块 64（不足则 32），之后 32 一块，最后一块为余数"""
    sizes = []
    first = 64 if T > 64 else 32
    rest = T
    if rest > first:
        sizes.append(first)
        rest -= first
        while rest > 32:
            sizes.append(32)
            rest -= 32
        sizes.append(rest)
    else:
        sizes.append(min(T, first))
        if rest > first:
            pass
    # 修正：T 很小时直接一块
    if T <= 32:
        return [T]
    return sizes


def mel_from_hidden(hidden, sizes, W, W_scale, bias):
    outs, st = [], 0
    for s in sizes:
        seg = hidden[st:st + s]
        if len(seg) == 0:
            continue
        outs.append(project(seg, W, W_scale, bias))
        st += s
    if st < len(hidden):
        outs.append(project(hidden[st:], W, W_scale, bias))
    return np.concatenate(outs, axis=0)
