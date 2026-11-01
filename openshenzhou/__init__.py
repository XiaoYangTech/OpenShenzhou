#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenShenzhou —— 纯本地、跨平台的微软神经语音 SDK（不注册 SAPI、不依赖 Windows 引擎）

用法：

    from openshenzhou import TTS
    tts = TTS('D:/Voice/Xiaoxiao_Female_zh-CN')
    tts.speak('你好，世界。', 'out.wav')

命令行：

    python -m openshenzhou --voice <语音包目录> --text "你好，世界。" --out out.wav
"""
import os
import sys

from .modelpack import VoicePack
from .synth import synthesize

__all__ = ["TTS", "VoicePack", "synthesize"]
__version__ = "0.1.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
_ZH_PHONES = os.environ.get("OPENSHENZHOU_ZH_PHONES", r"D:\Voice\Xiaoxiao_Female_zh-CN\phones.txt")


def _load_g2p(lang):
    if lang == "zh":
        from .g2p_zh import load_symbols, load_table, load_lexicon, text_to_ids
        return ("zh", load_symbols(_ZH_PHONES), load_table(), load_lexicon())
    if lang == "en":
        from .g2p_en import load_symbols, text_to_ids
        return ("en", None, None, None)
    return (lang, None, None, None)


class TTS:
    """一个语音包的朗读器"""

    def __init__(self, voice_dir, phones_path=None):
        self.pack = VoicePack(voice_dir)
        self.lang = self.pack.lang
        self.phones_path = phones_path
        self._sessions = None
        self._g2p = None
        self._proj = None
        self._stream = None
        # 英文变体：en-GB（英式）/ en-IN（印式）使用引擎实测的专用发音词表
        self._en_variant = self._detect_en_variant() if self.lang == "en" else ""

    def _detect_en_variant(self):
        d = (self.pack.dir or "").lower()
        if "en-gb" in d:
            return "gb"
        if "en-in" in d:
            return "in"
        syms = self.pack.symbols
        try:
            if any(str(k).endswith("_hin") for k in syms) or "vw" in syms:
                return "in"
            if "ir1" in syms:
                return "gb"
        except Exception:
            pass
        return ""

    def _ensure(self):
        if self._sessions is None:
            self._sessions = self.pack.sessions()
        if self._g2p is None:
            self._g2p = self._build_g2p()

    def _build_g2p(self):
        if self.lang == "zh":
            from .g2p_zh import load_symbols, load_table, load_lexicon, text_to_ids
            try:
                from . import crf_zh
                crf_zh.set_dat_path(os.path.join(self.pack.dir, "MSTTSLocZhCN.dat"))
            except Exception:
                pass
            return (text_to_ids,
                    load_table(),
                    load_symbols(self.phones_path or self.pack.phones_path),
                    load_lexicon())
        if self.lang == "en":
            from .g2p_en import load_lexicon, text_to_ids
            return (text_to_ids, self.pack.symbols, load_lexicon(self._en_variant), None)
        if self.lang in ("es", "fr", "de", "pt"):
            from .g2p_latin import load_lexicon, text_to_ids
            return (text_to_ids, self.pack.symbols, load_lexicon(self.lang), self.lang)
        if self.lang == "ja":
            from .g2p_ja import load_kana, load_lexicon, text_to_ids
            return (text_to_ids, self.pack.symbols, load_kana(), load_lexicon())
        if self.lang == "ko":
            from .g2p_ko import load_lexicon, load_syllables, text_to_ids
            return (text_to_ids, self.pack.symbols, load_lexicon(), load_syllables())
        return (None, None, None, None)

    def speak_ssml(self, ssml, out_path=None):
        """按 SSML 合成（volume 逐采样精确；rate/pitch 见 ssml 模块说明）"""
        from .ssml import speak_ssml as _speak_ssml
        audio = _speak_ssml(self, ssml)
        # 与引擎一致：整段音频首尾各 300 采样静音（Vocoder 前导/收尾帧）
        if len(audio) > 800:
            import numpy as _np
            audio = _np.asarray(audio, dtype=_np.float32).copy()
            audio[:300] = 0.0
            audio[-300:] = 0.0
        if out_path:
            import wave
            import numpy as _np
            data = (_np.clip(audio, -1.0, 1.0) * 32767.0).astype('<i2')
            w = wave.open(out_path, 'wb')
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(data.tobytes())
            w.close()
        return audio

    def text_to_ids(self, text):
        self._ensure()
        fn, a1, a2, a3 = self._g2p
        if self.lang == "zh":
            return fn(text, a1, a2, a3)
        if self.lang == "en":
            return fn(text, a1, a2, self._en_variant)
        if self.lang in ("es", "fr", "de", "pt"):
            return fn(text, a1, a2, a3)
        if self.lang in ("ja", "ko"):
            return fn(text, a1, a2, a3)
        raise NotImplementedError("暂不支持的语言: " + self.lang)

    def _streaming(self):
        """按引擎块调度逐块重放 decoder（与引擎逐采样一致）"""
        if self._stream is not None:
            return self._stream
        from .streaming import StreamingDecoder
        try:
            custom, plain = self.pack.decoder_paths()
            if not os.path.exists(custom):
                self._stream = False
                return False
            cache = os.path.join(self.pack._cache_dir(), "stream")
            self._stream = StreamingDecoder(custom, plain, cache)
        except Exception:
            self._stream = False
        return self._stream

    def _projection(self):
        """按引擎分块调度复刻 mel 投影（消除逐块量化残差）"""
        if self._proj is not None:
            return self._proj
        import onnxruntime as ort
        from .projection import load_projection
        try:
            dec_plain = self.pack.models()[1]
            hidden_path = self.pack.hidden_model()
            W, W_scale, bias = load_projection(dec_plain, hidden_path)
            so = ort.SessionOptions(); so.log_severity_level = 3
            sess = ort.InferenceSession(hidden_path, so, providers=["CPUExecutionProvider"])
            self._proj = (sess, W, W_scale, bias)
        except Exception:
            self._proj = False
        return self._proj

    def word_events(self, text, speed=1.0):
        """复刻引擎 WordBoundary 事件（audio_ticks / text_offset / text_length / word / type）"""
        from .events import word_events as _we
        return _we(self, text, speed=speed)

    def speak(self, text, out_path=None, speed=1.0):
        """文本 → 24kHz 音频（返回 numpy 数组，可选写 WAV）

        引擎对每个句子单独跑一次 encoder，因此多句文本要逐句合成再拼接
        （否则把多段 -bos-...~ 塞进一次 encoder 会得到错误音频）。
        """
        ids = self.text_to_ids(text)
        if self._sessions is None:
            self._sessions = self.pack.sessions()
        from .synth import split_sentences, write_wav
        chunks = split_sentences(ids, self.pack.symbols)
        stream = self._streaming()
        proj = None if stream else self._projection()

        def run(chunk):
            if stream:
                return synthesize(self._sessions, chunk, None, speed=speed, streaming=stream)
            return synthesize(self._sessions, chunk, None, speed=speed,
                              projection=proj if proj else None)

        if len(chunks) <= 1:
            audio = run(ids)
        else:
            import numpy as np
            audio = np.concatenate([run(ch) for ch in chunks])
        import numpy as _np
        audio = _np.asarray(audio, dtype=_np.float32)
        # 引擎音频首尾各有一段静音：逆向后确认引擎自然音频前 300、后 300 采样为 0
        # （来自 Vocoder 前导帧/收尾帧处理），复刻后自然音频与引擎逐采样一致率由
        # 98% 提升到 99.7%+，且 RateAdjustment 的基音检测得以与引擎对齐。
        if len(audio) > 800:
            audio = audio.copy()
            audio[:300] = 0.0
            audio[-300:] = 0.0
        # 引擎 RateAdjustment：语音包 INI 中非 100 的音色，引擎会对音频做
        # 基音同步拼接的保音高变速（已完全逆向，见 rateadjust.py）。
        rate = getattr(self.pack, "rate_adjust", 100) or 100
        if rate != 100:
            from .rateadjust import adjust_float
            audio = adjust_float(audio, rate / 100.0)
        if out_path:
            write_wav(audio, out_path)
        return audio


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="openshenzhou")
    ap.add_argument("--voice", required=True, help="语音包目录（含 .bin 模型与 phones.txt）")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args(argv)
    tts = TTS(a.voice)
    tts.speak(a.text, a.out, speed=a.speed)
    print("已写入 " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
