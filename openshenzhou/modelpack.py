#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modelpack.py —— 语音包加载：.bin → ONNX → 标准算子，带缓存"""
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np

HEADER = 94
KEY = b"ZCjZ7nHDSLvf4gpELteM4AnzaWUjTpn7UkV7D@vvksl0w1SNgon6d1905WANbktDc9S39oaA4r29HJNayXvTq8fJsq"
MODELS = ["am_v5_encoder", "am_v5_decoder", "device_vocoder_v6_streaming"]
CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".openshenzhou", "models")


def decode_bin(data: bytes) -> bytes:
    payload = np.frombuffer(data[HEADER:], dtype=np.uint8)
    key = np.frombuffer(KEY, dtype=np.uint8)
    ks = np.tile(key, len(payload) // len(key) + 1)[:len(payload)]
    return (payload ^ ks).tobytes()


def _ensure_onnx():
    global onnx, helper
    import onnx as _onnx
    from onnx import helper as _helper
    onnx = _onnx
    helper = _helper


def as_list(v):
    if v is None: return None
    if isinstance(v, (list, tuple)): return list(v)
    return [v]


def strip_control_inputs(node, control_names=("streaming_control",)):
    """按名字去掉 streaming_control 输入（不同算子位置不同）"""
    return [i for i in node.input if i not in control_names]


CAUSAL = False


def _causal_pads(attrs):
    k = attrs.get("kernel_shape") or [1]
    d = attrs.get("dilations") or [1]
    left = (k[0] - 1) * d[0]
    return [left, 0]


def rewrite_customops(src, dst):
    _ensure_onnx()
    m = onnx.load(src)
    g = m.graph
    removed = 0
    new_nodes = []
    for node in g.node:
        if node.domain != "test.customop":
            new_nodes.append(node)
            continue
        removed += 1
        op = node.op_type
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        ins = strip_control_inputs(node)
        if op == "StreamingConv":
            pads = _causal_pads(attrs) if CAUSAL else as_list(attrs.get("pads"))
            nn = helper.make_node("Conv", ins, list(node.output),
                                  name=node.name + "_conv",
                                  kernel_shape=as_list(attrs.get("kernel_shape")),
                                  pads=pads,
                                  strides=as_list(attrs.get("strides")),
                                  dilations=as_list(attrs.get("dilations")),
                                  group=attrs.get("group", 1))
        elif op == "StreamingConvTranspose":
            pads = as_list(attrs.get("pads"))
            if CAUSAL:
                k = attrs.get("kernel_shape") or [1]
                d = attrs.get("dilations") or [1]
                st = attrs.get("strides") or [1]
                # 转置卷积的因果形式：左侧不补，右侧补足剩余
                pads = [0, max(0, (k[0] - 1) * d[0] - (st[0] - 1))]
            nn = helper.make_node("ConvTranspose", ins, list(node.output),
                                  name=node.name + "_convt",
                                  kernel_shape=as_list(attrs.get("kernel_shape")),
                                  pads=pads,
                                  strides=as_list(attrs.get("strides")),
                                  dilations=as_list(attrs.get("dilations")),
                                  group=attrs.get("group", 1),
                                  output_padding=as_list(attrs.get("output_padding")))
        elif op == "StreamingPad":
            # 输入: (x, control, pads_tensor) -> Pad(x, pads, constant_value=0)
            nn = helper.make_node("Pad", ins, list(node.output),
                                  name=node.name + "_pad",
                                  mode=attrs.get("mode", b"constant").decode() if isinstance(attrs.get("mode", b"constant"), bytes) else attrs.get("mode", "constant"))
        elif op == "StreamingAdd":
            nn = helper.make_node("Add", ins, list(node.output), name=node.name + "_add")
        else:
            print("未知私有算子:", op)
            new_nodes.append(node)
            continue
        new_nodes.append(nn)

    del g.node[:]
    g.node.extend(new_nodes)

    # 清理不再使用的 graph input（如 streaming_control）
    used = set()
    for n in g.node:
        for i in n.input:
            used.add(i)
    inputs = [i for i in g.input if i.name in used]
    del g.input[:]
    g.input.extend(inputs)

    onnx.save(m, dst)
    print("%s -> %s  (替换 %d 个私有算子，剩余输入 %s)" % (src, dst, removed, [i.name for i in g.input]))
    return 0



def materialize_voice_dir(path):
    """语音包路径 → 可直接加载的目录

    支持：
      * 已解包的 appx 目录（含 phones.txt / AppxManifest.xml）
      * .appx / .msix / .zip 文件（自动解包到 ~/.openshenzhou/appx 缓存）
    """
    path = os.path.abspath(str(path))
    if os.path.isdir(path):
        # 允许传 appx 解包后的上层目录（一层子目录里才是语音包）
        if os.path.exists(os.path.join(path, "phones.txt")):
            return path
        subs = [d for d in sorted(os.listdir(path))
                if os.path.isdir(os.path.join(path, d))]
        if len(subs) == 1 and os.path.exists(os.path.join(path, subs[0], "phones.txt")):
            return os.path.join(path, subs[0])
        return path
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".appx", ".msix", ".zip"):
            h = hashlib.md5(("%s:%s" % (path, os.path.getmtime(path))).encode()).hexdigest()[:12]
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(CACHE_ROOT, "appx", stem + "-" + h)
            if not os.path.exists(os.path.join(out, ".ok")):
                os.makedirs(out, exist_ok=True)
                import zipfile
                with zipfile.ZipFile(path) as z:
                    z.extractall(out)
                open(os.path.join(out, ".ok"), "w").write("ok")
            return materialize_voice_dir(out)
        raise FileNotFoundError("不是语音包目录或 appx: %s" % path)
    raise FileNotFoundError(path)


class VoicePack:
    """一个语音包目录：自动解码 .bin、重写私有算子、加载音素表"""

    def __init__(self, pkg_dir):
        self.dir = materialize_voice_dir(pkg_dir)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(self.dir)
        self.name = os.path.basename(self.dir.rstrip("\\/"))
        self.phones_path = os.path.join(self.dir, "phones.txt")
        self.symbols = self._load_symbols(self.phones_path)
        self.ids = {v: k for k, v in self.symbols.items()}
        self._sessions = None
        self._lang = self._guess_lang()

    def _load_symbols(self, path):
        m = {}
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            name, idx = line.rsplit(":", 1)
            m[name] = int(idx)
        return m

    _LANG_TAGS = (("zh-cn", "zh"), ("en-us", "en"), ("en-gb", "en"),
                  ("en-in", "en"), ("ja-jp", "ja"), ("ko-kr", "ko"),
                  ("es-es", "es"), ("es-mx", "es"), ("pt-br", "pt"),
                  ("fr-fr", "fr"), ("de-de", "de"))

    def _guess_lang(self):
        candidates = [self.name.lower()]
        # appx 解包目录：从 AppxManifest.xml / MSTTSLoc*.dat 识别语言
        try:
            man = os.path.join(self.dir, "AppxManifest.xml")
            if os.path.exists(man):
                txt = open(man, encoding="utf-8", errors="replace").read().lower()
                candidates.append(txt[:200000])
        except Exception:
            pass
        try:
            for f in os.listdir(self.dir):
                fl = f.lower()
                if fl.startswith("msttsloc") and fl.endswith(".dat"):
                    candidates.append(fl)
        except Exception:
            pass
        for c in candidates:
            for tag, lang in self._LANG_TAGS:
                if tag in c:
                    return lang
        return "unknown"

    @property
    def lang(self):
        return self._lang

    def _cache_dir(self):
        h = hashlib.md5((self.dir + str(os.path.getmtime(self.phones_path))).encode()).hexdigest()[:12]
        d = os.path.join(CACHE_ROOT, self.name + "-" + h)
        os.makedirs(d, exist_ok=True)
        return d

    def models(self):
        """返回 3 个已解码、已重写的 ONNX 路径"""
        cache = self._cache_dir()
        out = []
        for base in MODELS:
            dst = os.path.join(cache, base + ".onnx")
            if not os.path.exists(dst):
                binp = os.path.join(self.dir, base + ".bin")
                if not os.path.exists(binp):
                    onnxp = os.path.join(self.dir, base + ".onnx")
                    if os.path.exists(onnxp):
                        shutil.copy(onnxp, dst)
                    else:
                        raise FileNotFoundError(base + ".bin")
                else:
                    tmp = os.path.join(cache, base + ".raw.onnx")
                    open(tmp, "wb").write(decode_bin(open(binp, "rb").read()))
                    if base == "am_v5_encoder":
                        os.replace(tmp, dst)
                    else:
                        if base == "am_v5_decoder":
                            import shutil as _sh
                            _sh.copy(tmp, os.path.join(cache, "am_v5_decoder.custom.onnx"))
                        rewrite_customops(tmp, dst)
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
            out.append(dst)
        return out

    def decoder_paths(self):
        """返回 (自定义算子版 decoder, 标准算子版 decoder)"""
        models = self.models()
        plain = models[1]
        custom = os.path.join(os.path.dirname(plain), "am_v5_decoder.custom.onnx")
        return custom, plain

    def hidden_model(self):
        """返回暴露 hidden（input.52）的 decoder 变体路径"""
        cache = self._cache_dir()
        dst = os.path.join(cache, 'am_v5_decoder.hidden.onnx')
        if not os.path.exists(dst):
            plain = self.models()[1]
            import onnx
            from onnx import helper, TensorProto
            from .projection import _find_projection
            m = onnx.load(plain)
            hidden_name, _, _, _ = _find_projection(m)
            m.graph.node.append(helper.make_node('Identity', [hidden_name], ['hidden_out'], name='expose_hidden'))
            m.graph.output.append(helper.make_tensor_value_info('hidden_out', TensorProto.FLOAT, None))
            onnx.save(m, dst)
        return dst

    def sessions(self):
        if self._sessions is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 4
            paths = self.models()
            self._sessions = [
                ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for p in paths
            ]
        return self._sessions
