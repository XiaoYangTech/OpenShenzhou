#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_all.py —— OpenShenzhou 端到端自检

覆盖：
  1. 数据完整性（8 语种词表/符号名、音色包模型）
  2. Python G2P 冒烟（8 语种）
  3. 合成冒烟（8 语种，检查采样数与有声性）
  4. Node≡Python G2P 逐 id 一致
  5. （可选）与引擎逐采样比对：--engine 时用探针数据

  python tests/test_all.py --voice-root "$TEMP/AllVoice"
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VOICES = {
    "zh": "Xiaoxiao_Female_zh-CN",
    "en": "Aria_Female_en-US",
    "es": "Alvaro_Male_es-ES",
    "fr": "Denise_Female_fr-FR",
    "de": "Conrad_Male_de-DE",
    "pt": "Francisca_Female_pt-BR",
    "ja": "Keita_Male_ja-JP",
    "ko": "InJoon_Male_ko-KR",
}
SAMPLES = {
    "zh": "你好，世界。",
    "en": "hello world.",
    "es": "hola mundo.",
    "fr": "bonjour le monde.",
    "de": "hallo welt.",
    "pt": "olá mundo.",
    "ja": "こんにちは。",
    "ko": "안녕하세요.",
}

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print("  [OK] %s" % name)
    else:
        failed += 1
        print("  [FAIL] %s %s" % (name, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-root", default=os.path.join(os.environ.get("TEMP", ""), "AllVoice"))
    ap.add_argument("--skip-node", action="store_true")
    ap.add_argument("--quick", action="store_true", help="只测 zh")
    a = ap.parse_args()

    langs = ["zh"] if a.quick else list(VOICES)

    print("== 1. 数据完整性 ==")
    for lang in langs:
        names = os.path.join(ROOT, "data", "lexicon_%s.names.json" % lang)
        has = os.path.exists(names)
        check("%s 词表（符号名版）" % lang, has, names + " 缺失")
        if has:
            d = json.load(open(names, encoding="utf-8"))
            check("%s 词表条目 > 500" % lang, len(d) > 500, "实际 %d" % len(d))
    for f in ["kana_ja.names.json", "syllables_ko.names.json", "ja_readings.json"]:
        check(f, os.path.exists(os.path.join(ROOT, "data", f)))

    print("== 2. Python G2P 冒烟 ==")
    from openshenzhou import TTS
    tts = {}
    for lang in langs:
        pkg = os.path.join(a.voice_root, VOICES[lang])
        if not os.path.exists(pkg):
            check("%s 音色包存在" % lang, False, pkg)
            continue
        t = TTS(pkg)
        tts[lang] = t
        ids = t.text_to_ids(SAMPLES[lang])
        check("%s G2P 产出 id（%d 个）" % (lang, len(ids)),
              len(ids) > 5 and all(isinstance(x, int) for x in ids))
        tilde = t.pack.symbols.get("~")
        if tilde is not None:
            check("%s 序列以 ~ 结尾" % lang, tilde == ids[-1])

    print("== 3. 合成冒烟 ==")
    import numpy as np
    for lang in langs:
        if lang not in tts:
            continue
        audio = np.asarray(tts[lang].speak(SAMPLES[lang]))
        check("%s 音频长度（%.2f 秒）" % (lang, len(audio) / 24000.0), len(audio) > 4800)
        check("%s 音频非静音" % lang, float(np.abs(audio).max()) > 0.05)

    # 引擎自然音频首尾各 300 采样静音（Vocoder 前导/收尾帧），SDK 已复刻
    if "en" in tts:
        _a = np.asarray(tts["en"].speak("hello world."))
        if len(_a) > 800:
            check("自然音频首 300 采样为静音", float(np.abs(_a[:300]).max()) == 0.0)
            check("自然音频末 300 采样为静音", float(np.abs(_a[-300:]).max()) == 0.0)

    print("== 3b. SSML 冒烟 ==")
    if "en" in tts:
        import numpy as _np
        t = tts["en"]
        ns = 'xmlns="http://www.w3.org/2001/10/synthesis"'
        base = _np.asarray(t.speak_ssml('<speak %s>Hello world.</speak>' % ns))
        check("SSML 基础合成", len(base) > 4800)
        loud = _np.asarray(t.speak_ssml('<speak %s><prosody volume="80">Hello world.</prosody></speak>' % ns))
        rms = lambda x: float(_np.sqrt((x ** 2).mean()))
        ratio = rms(loud) / (rms(base) + 1e-9)
        check("SSML volume=80 增益 ≈0.5623", abs(ratio - 0.5623) < 0.02, "实际 %.4f" % ratio)
        slow = _np.asarray(t.speak_ssml('<speak %s><prosody rate="-20%%">Hello world.</prosody></speak>' % ns))
        check("SSML rate=-20% 时长 ×1.25", abs(len(slow) / len(base) - 1.25) < 0.03,
              "实际 %.4f" % (len(slow) / len(base)))
        brk = _np.asarray(t.speak_ssml('<speak %s>Hello<break time="500ms"/>world.</speak>' % ns))
        check("SSML break=500ms 插入 12000 采样", abs((len(brk) - len(base)) - 12000) < 600,
              "实际差 %d" % (len(brk) - len(base)))

    print("== 3c. RateAdjustment 精确复刻（引擎算法自检） ==")
    from openshenzhou.rateadjust import adjust_int16
    import numpy as _np2
    rng = _np2.random.RandomState(0)
    # 用带周期的合成信号检查 int16 输出与长度关系（不依赖音色包）
    tsec = 0.5
    n = int(24000 * tsec)
    per = 120
    sig = (_np2.sin(2 * _np2.pi * _np2.arange(n) / per) * 8000).astype(_np2.int16)
    for rate, expect in ((110, 1 / 1.1), (90, 1 / 0.9), (95, 1 / 0.95)):
        out = adjust_int16(sig, rate / 100.0)
        ratio = len(out) / len(sig)
        check("rate=%d 输出长度比 ≈%.3f" % (rate, expect), abs(ratio - expect) < 0.02,
              "实际 %.4f（%d/%d）" % (ratio, len(out), len(sig)))
        check("rate=%d 输出为 int16" % rate, out.dtype == _np2.int16)
    # 同一输入的确定性
    o1 = adjust_int16(sig, 1.1)
    o2 = adjust_int16(sig, 1.1)
    check("rate复刻确定性", _np2.array_equal(o1, o2))

    print("== 3d. SSML pitch 语义（引擎链路自检） ==")
    if "en" in tts:
        import numpy as _np3
        _t = tts["en"]
        _ns = 'xmlns="http://www.w3.org/2001/10/synthesis"'
        def _pitch_ctx(pct):
            return _np3.asarray(_t.speak_ssml(
                '<speak %s><prosody pitch="%s">hello world.</prosody></speak>' % (_ns, pct)),
                dtype=_np3.float32)
        def _f0_est(x, sr=24000):
            x = _np3.asarray(x, dtype=_np3.float64)
            vals = []
            for st in range(0, len(x) - 2048, 512):
                seg = x[st:st + 2048]
                if _np3.sqrt((seg ** 2).mean()) < 0.02:
                    continue
                seg = seg - seg.mean()
                ac = _np3.correlate(seg, seg, 'full')[len(seg) - 1:]
                lo, hi = int(sr / 500), int(sr / 70)
                if hi >= len(ac):
                    continue
                k = lo + int(_np3.argmax(ac[lo:hi]))
                if ac[k] > 0.3 * ac[0]:
                    vals.append(sr / k)
            return float(_np3.median(vals)) if vals else 0.0
        _base = _pitch_ctx("0%")
        _up = _pitch_ctx("50%")
        _dn = _pitch_ctx("-50%")
        check("pitch +50% 时长保持", abs(len(_up) / len(_base) - 1.0) < 0.02,
              "实际 %.4f" % (len(_up) / len(_base)))
        check("pitch -50% 时长保持", abs(len(_dn) / len(_base) - 1.0) < 0.02,
              "实际 %.4f" % (len(_dn) / len(_base)))
        _r_up = _f0_est(_up) / (_f0_est(_base) + 1e-9)
        _r_dn = _f0_est(_dn) / (_f0_est(_base) + 1e-9)
        check("pitch +50% F0 比 ≈ 2^(5/12)=1.3348", abs(_r_up - 1.3348) < 0.08,
              "实际 %.4f" % _r_up)
        check("pitch -50% F0 比 ≈ 2^(-5/12)=0.7492", abs(_r_dn - 0.7492) < 0.06,
              "实际 %.4f" % _r_dn)

    print("== 3e. SSML break 语义自检 ==")
    if "en" in tts:
        import numpy as _np4
        _t = tts["en"]
        _ns = 'xmlns="http://www.w3.org/2001/10/synthesis"'
        _base = _np4.asarray(_t.speak_ssml('<speak %s>hello world.</speak>' % _ns), dtype=_np4.float32)
        _brk = _np4.asarray(_t.speak_ssml(
            '<speak %s><break time="500ms"/>hello world.</speak>' % _ns), dtype=_np4.float32)
        check("break 500ms 总长 +12000 采样", len(_brk) == len(_base) + 12000,
              "%d vs %d" % (len(_brk), len(_base) + 12000))
        _runs = []
        _st = None
        for _i, _v in enumerate(_brk):
            if abs(_v) < 1e-7 and _st is None:
                _st = _i
            if abs(_v) >= 1e-7 and _st is not None:
                _runs.append((_st, _i))
                _st = None
        if _st is not None:
            _runs.append((_st, len(_brk)))
        _runs.sort(key=lambda x: x[1] - x[0], reverse=True)
        check("break 在开头插入位置=1200（引擎块边界）",
              bool(_runs) and _runs[0][0] == 1200 and _runs[0][1] - _runs[0][0] == 12000,
              str(_runs[0]) if _runs else "None")

    if not a.skip_node and os.path.exists(os.path.join(ROOT, "node", "node_modules")):
        print("== 4. Node≡Python G2P 一致 ==")
        script = os.path.join(os.environ.get("TEMP", "."), "os_node_ids.js")
        with open(script, "w", encoding="utf-8") as f:
            f.write("const { TTS } = require(process.argv[2] + '/index.js');\n"
                    "const t = new TTS(process.argv[3]);\n"
                    "console.log(JSON.stringify(t.textToIds(process.argv[4])));\n")
        node_dir = os.path.join(ROOT, "node").replace(os.sep, "/")
        for lang in langs:
            if lang not in tts:
                continue
            pkg = os.path.join(a.voice_root, VOICES[lang])
            r = subprocess.run(["node", script, node_dir, pkg, SAMPLES[lang]],
                               capture_output=True, encoding="utf-8", errors="replace")
            try:
                other = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                check("%s Node G2P" % lang, False, (r.stderr or "")[-120:])
                continue
            check("%s Node≡Python id 一致" % lang, other == tts[lang].text_to_ids(SAMPLES[lang]))

    # 5. 前端 golden（引擎实测的 24 句，跨 8 语种）
    gold_path = os.path.join(ROOT, "tests", "golden_frontend.json")
    if os.path.exists(gold_path):
        print("== 5. 前端 golden（引擎实测） ==")
        gold = json.load(open(gold_path, encoding="utf-8"))
        cache = {}
        for lang in langs:
            rows = [r for r in gold.get(lang, []) if r.get("match")]
            for r in rows:
                pkg = os.path.join(a.voice_root, r["pkg"])
                if not os.path.exists(pkg):
                    check("%s golden 音色包 %s" % (lang, r["pkg"]), False)
                    continue
                if pkg not in cache:
                    cache[pkg] = TTS(pkg)
                ids = cache[pkg].text_to_ids(r["text"])
                check("%s golden %s" % (lang, r["text"][:12]),
                      ids == r["ids"], "len %d vs %d" % (len(ids), len(r["ids"])))
        if not a.skip_node and os.path.exists(os.path.join(ROOT, "node", "node_modules")):
            print("== 6. Node 前端 golden 一致 ==")
            script = os.path.join(os.environ.get("TEMP", "."), "os_node_gold.js")
            with open(script, "w", encoding="utf-8") as f:
                f.write("const { TTS } = require(process.argv[2] + '/index.js');\n")
                f.write("const t = new TTS(process.argv[3]);\n")
                f.write("console.log(JSON.stringify(t.textToIds(process.argv[4])));\n")
            node_dir = os.path.join(ROOT, "node").replace(os.sep, "/")
            for lang in langs:
                rows = [r for r in gold.get(lang, []) if r.get("match")]
                for r in rows:
                    pkg = os.path.join(a.voice_root, r["pkg"])
                    rr = subprocess.run(["node", script, node_dir, pkg, r["text"]],
                                        capture_output=True, encoding="utf-8", errors="replace")
                    try:
                        other = json.loads(rr.stdout.strip().splitlines()[-1])
                    except Exception:
                        check("%s Node golden %s" % (lang, r["text"][:12]), False,
                              (rr.stderr or "")[-100:])
                        continue
                    check("%s Node golden %s" % (lang, r["text"][:12]),
                          other == r["ids"], "len %d vs %d" % (len(other), len(r["ids"])))

    print("\n结果: %d 通过, %d 失败" % (passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
