#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_golden.py —— 采集「前端 golden 用例」：引擎 ids + SDK ids 对比，写入仓仓库 tests 数据"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", "OpenShenzhou"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "tools"))

import ml_eval as E  # noqa: E402

CASES = {
    "zh": ["你好，世界。", "网络连接已断开。", "会议将在下午三点开始。"],
    "en": ["Hello world.", "This is a test sentence.", "The system is updating now."],
    "fr": ["Bonjour le monde.", "Le système est en cours de mise à jour.", "Comment allez-vous?"],
    "de": ["Hallo Welt.", "Bitte öffnen Sie die Einstellungen.", "Die Datei wurde gespeichert."],
    "es": ["Hola mundo.", "El sistema se está actualizando.", "Esta es una frase de prueba."],
    "pt": ["Olá mundo.", "O sistema está sendo atualizado.", "Terminaremos o teste na sexta-feira."],
    "ja": ["こんにちは。", "会議は十時に始まります。", "これはテストの文章です。"],
    "ko": ["안녕하세요.", "시스템이 업데이트 중입니다.", "회의는 열 시에 시작합니다."],
}


def main():
    out = {}
    for lang, texts in CASES.items():
        pkg, voice, _ = E.CASES[lang]
        rows = []
        for t in texts:
            eng = E.speak_capture(t, voice)
            ours = E.synth(lang, t, pkg)
            rows.append({"text": t, "pkg": pkg, "ids": eng, "sdk": ours,
                         "match": eng == ours if eng else False})
        out[lang] = rows
        n = sum(1 for r in rows if r["match"])
        print(lang, "%d/%d" % (n, len(rows)))
    path = os.path.join(ROOT, "tests", "golden_frontend.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
