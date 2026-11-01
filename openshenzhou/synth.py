#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""synth.py —— 三段模型推理：音素 ID → 24kHz 波形"""
import os
import wave

import numpy as np

MEL_SCALE = 0.25   # 引擎实测：decoder 输出 × 0.25 后转置成 [1,80,T] 给 vocoder
END_ID = 1         # '~' 序列结束符；引擎每句一个序列


def split_sentences(ids, sym=None, end_id=END_ID):
    """把多句音素序列拆成逐句序列（每句以 ~ 结束）。

    引擎对每个句子单独跑一次 encoder：句间边界可能是 ~（中文逐句流），
    也可能是 br4 + 标点后直接接下一句（英文单流），两种都要拆开。
    """
    if sym:
        # 每个语音包的 '~' id 可能不同；部分语音包（如 Francisca pt-BR）符号表里没有 '~'，
        # 此时整段 ids 就是一个序列，不做切分也不补 END。
        end_id = sym.get("~")
        bos_id = sym.get("-bos-")
        br0_id = sym.get("br0")
        br4_id = sym.get("br4")
        punc_ids = {v for k, v in sym.items() if k.startswith("punc")}
        punc_ids.add(sym.get("punc.", -1))
        # '/'：多语种前端用��的分隔/收尾符号，视为结构符号，不切开句子
        if sym.get("/") is not None:
            punc_ids.add(sym["/"])
        punc_ids.discard(None)
    else:
        bos_id = None; br0_id = None; br4_id = None; punc_ids = set()
    def _close(cur):
        if end_id is None:
            return cur
        # 句尾补 br4：若末尾是 punc token 且前面没有 br4，则插入 br4
        if cur and cur[-1] == end_id:
            j = len(cur) - 2
            if (br4_id is not None and j >= 0 and cur[j] in punc_ids
                    and br4_id not in cur[max(0, j - 2):j]):
                cur.insert(j, br4_id)
        if not cur or cur[-1] != end_id:
            cur.append(end_id)
        return cur

    out, cur = [], []
    n = len(ids)
    i = 0
    while i < n:
        x = ids[i]
        cur.append(x)
        i += 1
        if end_id is not None and x == end_id:
            out.append(_close(cur)); cur = []
            continue
        if br4_id is not None and x == br4_id:
            # 句末 br4：把紧跟的标点 token 一并收进本句，然后按句收尾
            while i < n and ids[i] in punc_ids:
                cur.append(ids[i]); i += 1
            if i < n and ids[i] != end_id:
                out.append(_close(cur)); cur = []
                # 下一句补齐 -bos- br0 前缀
                if ids[i] != bos_id and bos_id is not None:
                    cur = [x for x in (bos_id, br0_id) if x is not None]
            continue
    if cur:
        out.append(_close(cur))
    return out


def write_wav(audio, out_path, sample_rate=24000):
    import numpy as np
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def synthesize(sessions, ids, out_path=None, speed=1.0, sample_rate=24000, projection=None, streaming=None):
    """ids: 音素 ID 列表；speed: 1.0 正常，<1 更慢，>1 更快
    projection=(hidden_session, W, W_scale, bias) 时按引擎分块调度做 mel 投影"""
    enc, dec, voc = sessions
    n = len(ids)
    x = np.array([ids], dtype=np.int32)
    sp = np.full((1, n), float(speed), dtype=np.float32)
    dec_inputs, durations = enc.run(None, {"inputs": x, "speed_ratios": sp})

    h = np.zeros((1, 1, 128), dtype=np.float32)
    c = np.zeros((1, 1, 128), dtype=np.float32)
    feeds = {"decoder_inputs": dec_inputs.astype(np.float32), "h1": h, "c1": c, "h2": h, "c2": c}

    if streaming is not None:
        mels = streaming.decode(dec_inputs)[None, :, :]
    elif projection is not None:
        hidden_sess, W, W_scale, bias_w = projection
        from .projection import decoder_chunk_sizes, mel_from_hidden
        outs = hidden_sess.run(None, feeds)
        names = [o.name for o in hidden_sess.get_outputs()]
        d = dict(zip(names, outs))
        H = np.asarray(d["hidden_out"]).reshape(-1, 256)
        T = H.shape[0]
        mels = mel_from_hidden(H, decoder_chunk_sizes(T), W, W_scale, bias_w)[None, :, :]
    else:
        mels, *_ = dec.run(None, feeds)

    mel = (np.transpose(mels, (0, 2, 1)) * MEL_SCALE).astype(np.float32)
    wav, *_ = voc.run(None, {
        "mel": mel,
        "state1_in": np.zeros((1, 128), dtype=np.float32),
        "state2_in": np.zeros((1, 64), dtype=np.float32),
    })
    audio = np.asarray(wav).reshape(-1).astype(np.float32)

    if sample_rate != 24000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * sample_rate / 24000))
        audio = np.interp(idx, np.arange(len(audio)), audio)

    if out_path:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
    return audio
