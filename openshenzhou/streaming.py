#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming.py —— 按引擎块调度逐块重放 decoder（与官方引擎逐采样一致）

原理：
  StreamingConv = 因果卷积 + 右看 1 帧：首块每层少 1 帧、中间块等长、末块每层多 1 帧
  StreamingAdd  = 延迟对齐：低延迟分支先补缓存（首块直接裁头），末块尾部补零
  量化算子按块内张量现算 scale —— 逐块执行时自动与引擎一致

用法：
    dec = StreamingDecoder(decoder_onnx, cache_dir)
    mel = dec.decode(decoder_inputs)      # [1,T,256] -> [T,80]
"""
import hashlib
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

CONTROL = "streaming_control"


def _delay_map(m):
    delay = {}
    for i in m.graph.initializer:
        delay[i.name] = 0
    for i in m.graph.input:
        delay.setdefault(i.name, 0)
    for node in m.graph.node:
        if node.domain == "test.customop":
            d = delay.get(node.input[0], 0)
            if node.op_type == "StreamingConv":
                delay[node.output[0]] = d + 1
            elif node.op_type == "StreamingAdd":
                delay[node.output[0]] = max(delay.get(node.input[0], 0), delay.get(node.input[1], 0))
            else:
                delay[node.output[0]] = d
            continue
        ds = [delay.get(i, 0) for i in node.input if i]
        d = max(ds) if ds else 0
        for o in node.output:
            delay[o] = d
    return delay


def _plain_shapes(plain_model, n):
    mm = onnx.ModelProto()
    mm.CopyFrom(plain_model)
    for i in mm.graph.input:
        if i.name == "decoder_inputs":
            dims = i.type.tensor_type.shape.dim
            dims[0].dim_value = 1
            dims[1].dim_value = n
            dims[2].dim_value = 256
    inf = shape_inference.infer_shapes(mm)
    shp = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else None)
        shp[vi.name] = dims
    return shp


def _seq_axis(shape):
    if not shape:
        return 2
    rank = len(shape)
    if rank == 4 or rank == 2:
        return 0
    if rank == 3:
        if shape[1] in (128, 256, 512) and shape[2] not in (128, 256, 512):
            return 2
        return 1
    return 0


def _desired(n, delay, mode):
    if mode == "first":
        return n - delay
    if mode == "middle":
        return n
    return n + delay


def chunk_plan(T):
    """引擎 decoder 分块调度（由引擎实测块长归纳）：
       首块 64（不足 64 时取 min(32,T)）；之后剩余 >64 发 64、>32 发 32；最后一块为 flush。
    """
    if T <= 0:
        return []
    first = 64 if T > 64 else min(32, T)
    chunks = [(first, "first")]
    rest = T - first
    while rest > 0:
        if rest > 64:
            chunks.append((64, "middle"))
            rest -= 64
        elif rest > 32:
            chunks.append((32, "middle"))
            rest -= 32
        else:
            chunks.append((rest, "flush"))
            rest = 0
    if chunks[-1][1] != "flush":
        chunks[-1] = (chunks[-1][0], "flush")
    return chunks


class StreamingDecoder:
    def __init__(self, decoder_onnx, plain_onnx, cache_dir):
        self.decoder_onnx = decoder_onnx
        self.plain_onnx = plain_onnx
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._m = onnx.load(decoder_onnx)
        self._plain = onnx.load(plain_onnx)
        self._sessions = {}
        self._templates = {}

    # ---------- 生成某一 (mode, n) 的静态形状模型 ----------
    def _build(self, mode, n):
        key = (mode, n)
        path = os.path.join(self.cache_dir, "dec_%s_%d.onnx" % (mode, n))
        if not os.path.exists(path):
            model = onnx.ModelProto()
            model.CopyFrom(self._m)
            delay = _delay_map(model)
            shp = _plain_shapes(self._plain, n)
            g = model.graph
            inits = {i.name: i for i in g.initializer}
            new_nodes, new_inits, caches, snaps = [], [], [], []
            for node in g.node:
                if node.domain != "test.customop":
                    new_nodes.append(node)
                    continue
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                ins = [i for i in node.input if i != CONTROL]
                op = node.op_type
                if op == "StreamingConv":
                    X, W = ins[0], ins[1]
                    B = ins[2] if len(ins) > 2 else None
                    group = attrs.get("group", 1)
                    wsh = numpy_helper.to_array(inits[W]).shape
                    c_in = wsh[1] * group
                    d_out = delay[node.output[0]]
                    cache = X + "_cache_in"
                    cat = X + "_cat"
                    full = node.output[0] + "_full"
                    new_nodes.append(helper.make_node("Concat", [cache, X], [cat], name=node.name + "_cat", axis=2))
                    pads = [0, 1] if mode == "flush" else [0, 0]
                    new_nodes.append(helper.make_node(
                        "Conv", [cat, W] + ([B] if B else []), [full], name=node.name + "_conv",
                        kernel_shape=attrs.get("kernel_shape"), pads=pads,
                        strides=attrs.get("strides"), dilations=attrs.get("dilations"), group=group))
                    if mode == "first":
                        st, en, ax = full + "_st", full + "_en", full + "_ax"
                        new_inits += [
                            numpy_helper.from_array(np.array([1], np.int64), st),
                            numpy_helper.from_array(np.array([_desired(n, d_out, mode) + 1], np.int64), en),
                            numpy_helper.from_array(np.array([2], np.int64), ax),
                        ]
                        new_nodes.append(helper.make_node("Slice", [full, st, en, ax], list(node.output), name=node.name + "_slice"))
                    else:
                        new_nodes.append(helper.make_node("Identity", [full], list(node.output), name=node.name + "_id"))
                    snap = X + "_cache_out"
                    new_nodes.append(helper.make_node("Identity", [X], [snap], name=node.name + "_snap"))
                    caches.append(helper.make_tensor_value_info(cache, TensorProto.FLOAT, [1, c_in, 2]))
                    snaps.append(helper.make_tensor_value_info(snap, TensorProto.FLOAT, None))
                elif op == "StreamingAdd":
                    A, Bn = ins[0], ins[1]
                    out = node.output[0]
                    if len(shp.get(A) or []) < 3 or len(shp.get(Bn) or []) < 3:
                        new_nodes.append(helper.make_node("Add", [A, Bn], [out], name=node.name + "_add"))
                        continue
                    dA, dB = delay[A], delay[Bn]
                    d = abs(dA - dB)
                    if d == 0:
                        new_nodes.append(helper.make_node("Add", [A, Bn], [out], name=node.name + "_add"))
                        continue
                    low = A if dA < dB else Bn
                    high = Bn if dA < dB else A
                    shp_low = shp.get(low) or [1, 1, 1]
                    axis = _seq_axis(shp_low)
                    target = _desired(n, max(dA, dB), mode)
                    cache_name = out + "_addcache_in"
                    cat = out + "_addcat"
                    if mode == "first":
                        cat = low
                    else:
                        new_nodes.append(helper.make_node("Concat", [cache_name, low], [cat], name=node.name + "_acat", axis=axis))
                    st, en, ax_n = out + "_st2", out + "_en2", out + "_ax2"
                    new_inits += [
                        numpy_helper.from_array(np.array([0], np.int64), st),
                        numpy_helper.from_array(np.array([target], np.int64), en),
                        numpy_helper.from_array(np.array([axis], np.int64), ax_n),
                    ]
                    cropped = out + "_aligned"
                    new_nodes.append(helper.make_node("Slice", [cat, st, en, ax_n], [cropped], name=node.name + "_aslice"))
                    new_nodes.append(helper.make_node("Add", [high, cropped], [out], name=node.name + "_add"))
                    snap = out + "_addcache_out"
                    cst, cen, cax = out + "_cs", out + "_ce", out + "_ca"
                    new_inits += [
                        numpy_helper.from_array(np.array([-d], np.int64), cst),
                        numpy_helper.from_array(np.array([2 ** 31 - 1], np.int64), cen),
                        numpy_helper.from_array(np.array([axis], np.int64), cax),
                    ]
                    new_nodes.append(helper.make_node("Slice", [low, cst, cen, cax], [snap], name=node.name + "_csnap"))
                    cshape = list(shp_low)
                    cshape[axis] = d
                    caches.append(helper.make_tensor_value_info(cache_name, TensorProto.FLOAT, cshape))
                    snaps.append(helper.make_tensor_value_info(snap, TensorProto.FLOAT, None))
                else:
                    new_nodes.append(node)
            del g.node[:]
            g.node.extend(new_nodes)
            for ni in new_inits:
                g.initializer.append(ni)
            for i in g.input:
                if i.name == "decoder_inputs":
                    dims = i.type.tensor_type.shape.dim
                    dims[0].dim_value = 1
                    dims[1].dim_value = n
                    dims[2].dim_value = 256
            keep = [i for i in g.input if i.name != CONTROL]
            del g.input[:]
            g.input.extend(keep)
            g.input.extend(caches)
            have = {o.name for o in g.output}
            for s in snaps:
                if s.name not in have:
                    g.output.append(s)
            onnx.save(model, path)
        return path

    def session(self, mode, n):
        import onnxruntime as ort
        key = (mode, n)
        if key not in self._sessions:
            path = self._build(mode, n)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._sessions[key] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return self._sessions[key]

    @staticmethod
    def _zero_for(inp, n):
        return np.zeros([d if isinstance(d, int) else n for d in inp.shape], np.float32)

    def decode(self, decoder_inputs):
        """decoder_inputs: [1,T,256] → mel [T,80]（与引擎逐采样一致）"""
        x = np.asarray(decoder_inputs, dtype=np.float32)
        T = x.shape[1]
        chunks = chunk_plan(T)
        states = [np.zeros((1, 1, 128), np.float32) for _ in range(4)]
        caches = {}
        mels = []
        pos = 0
        for idx, (n, mode) in enumerate(chunks):
            S = self.session(mode, n)
            cin = [i for i in S.get_inputs() if "cache_in" in i.name]
            feeds = {
                "decoder_inputs": x[:, pos:pos + n, :].reshape(1, n, 256) if n > 0 else np.zeros((1, 0, 256), np.float32),
                "h1": states[0], "c1": states[1], "h2": states[2], "c2": states[3],
            }
            for i in cin:
                feeds[i.name] = caches.get(i.name, self._zero_for(i, n))
            outs = S.run(None, feeds)
            d = dict(zip([o.name for o in S.get_outputs()], outs))
            mels.append(d["output_mels"].reshape(-1, 80))
            states = [d["h1_new"].astype(np.float32), d["c1_new"].astype(np.float32),
                      d["h2_new"].astype(np.float32), d["c2_new"].astype(np.float32)]
            for i in cin:
                snap = np.ascontiguousarray(d[i.name.replace("cache_in", "cache_out")]).astype(np.float32)
                target = [dd if isinstance(dd, int) else n for dd in i.shape]
                axis = 2 if (len(target) == 3 and target[1] in (128, 256, 512) and target[2] == 2) else 1
                want = target[axis]
                if snap.shape[axis] != want:
                    sl = [slice(None)] * snap.ndim
                    sl[axis] = slice(-want, None)
                    snap = snap[tuple(sl)]
                caches[i.name] = snap
            pos += n
        return np.concatenate(mels, axis=0) if mels else np.zeros((0, 80), np.float32)
