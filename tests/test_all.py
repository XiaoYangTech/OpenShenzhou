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
