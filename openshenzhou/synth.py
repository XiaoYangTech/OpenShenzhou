#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""synth.py —— 三段模型推理：音素 ID → 24kHz 波形"""
import os
import wave

import numpy as np

MEL_SCALE = 0.25   # 引擎实测：decoder 输出 × 0.25 后转置成 [1,80,T] 给 vocoder
END_ID = 1         # '~' 序列结束符；引擎每句一个序列


def split_sentences(ids, end_id=END_ID):
    """把「多句首尾相接」的音素序列拆成逐句序列（每句以 ~ 结束）。"""
    out, cur = [], []
    for x in ids:
        cur.append(x)
        if x == end_id:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
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
