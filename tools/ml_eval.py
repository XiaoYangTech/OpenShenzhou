#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ml_eval.py —— 8 语种前端「逐 id 一致率」评测

做法同中文评测：用 ORT 探针抓引擎 encoder 输入 ids，与 SDK 输出整句对比。
用法::
    python tools/ml_eval.py                # 全部语种
    python tools/ml_eval.py ja             # 只测日语
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "tools"))

PROBE = os.path.join(os.environ["TEMP"], "probe_dbg")
PORT = int(os.environ.get("OS_ENGINE_PORT", "17461"))
PKG = os.path.join(os.environ["TEMP"], "AllVoice")


SENTINEL = {"zh": "测试哨兵甲", "ja": "てすとせんてい", "ko": "테스트센티널"}


def _ids_of(fn):
    return np.fromfile(os.path.join(PROBE, fn), dtype=np.int32).tolist()


def _cidx(fn):
    try:
        return int(fn.rsplit("_c", 1)[1].split(".")[0])
    except Exception:
        return -1


def _speak(text, voice):
    req = urllib.request.Request("http://127.0.0.1:%d/speak" % PORT,
                                 data=json.dumps({"text": text, "voice": voice}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=40).read()
    except Exception:
        pass


_SENTINEL_IDS = {}


def calibrate_sentinel(voice, sentinel="zzzqqq"):
    """取哨兵文本的 ids 作为签名（同一次进程内缓存）。"""
    if voice in _SENTINEL_IDS:
        return _SENTINEL_IDS[voice]
    for _ in range(5):
        before = set(os.listdir(PROBE))
        _speak(sentinel, voice)
        time.sleep(0.5)
        new = [f for f in (set(os.listdir(PROBE)) - before) if f.startswith("in_n2_i0")]
        if not new:
            continue
        new.sort(key=_cidx)
        sig = _ids_of(new[-1])
        _SENTINEL_IDS[voice] = sig
        return sig
    return None


def speak_capture(text, voice, rev=None, sentinel="zzzqqq"):
    """哨兵屏障抓取：请求 text → 请求哨兵 → 取「哨兵文件」前一个文件的 ids。

    引擎推理是异步的，批量评测时探针文件会乱序，必须用屏障才能正确归属。
    """
    sig = calibrate_sentinel(voice, sentinel)
    if sig is None:
        return None
    for _ in range(3):
        before = set(os.listdir(PROBE))
        _speak(text, voice)
        time.sleep(0.30)
        _speak(sentinel, voice)
        time.sleep(0.55)
        new = [f for f in (set(os.listdir(PROBE)) - before) if f.startswith("in_n2_i0")]
        if len(new) < 2:
            continue
        new.sort(key=_cidx)
        data = [_ids_of(fn) for fn in new]
        idx = None
        for k in range(len(data) - 1, -1, -1):
            if data[k] == sig:
                idx = k
                break
        if idx is None or idx == 0:
            continue
        return data[idx - 1]
    return None


def load_sym(pkg):
    sym = {}
    for line in open(os.path.join(PKG, pkg, "phones.txt"), encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        n, i = line.rsplit(":", 1)
        sym[n] = int(i)
    return sym


def synth(module, text, pkg):
    sym = load_sym(pkg)
    if module == "zh":
        import openshenzhou.g2p_zh as m
        return m.text_to_ids(text, m.load_table(), sym, m.load_lexicon())
    if module == "ja":
        import openshenzhou.g2p_ja as m
        return m.text_to_ids(text, sym, m.load_kana(), m.load_lexicon())
    if module == "ko":
        import openshenzhou.g2p_ko as m
        return m.text_to_ids(text, sym, m.load_lexicon(), m.load_syllables())
    if module == "en":
        import openshenzhou.g2p_en as m
        return m.text_to_ids(text, sym, m.load_lexicon())
    import openshenzhou.g2p_latin as m
    lang = {"fr": "fr", "de": "de", "es": "es", "pt": "pt"}[module]
    return m.text_to_ids(text, sym, m.load_lexicon(lang), lang)


CASES = {
    "zh": ("Xiaoxiao_Female_zh-CN", "Microsoft Xiaoxiao", [
        "你好，世界。", "今天天气很好。", "请帮我打开设置。", "系统正在更新中。",
        "这是一段测试文本。", "我们需要保存文件。", "会议将在下午三点开始。",
    ]),
    "en": ("Aria_Female_en-US", "Microsoft Aria", [
        "Hello world.", "How are you today?", "Please open the settings.",
        "The system is updating now.", "This is a test sentence.",
    ]),
    "fr": ("Denise_Female_fr-FR", "Microsoft Denise", [
        "Bonjour le monde.", "Comment allez-vous?", "Veuillez ouvrir les paramètres.",
        "Le système est en cours de mise à jour.", "Ceci est une phrase de test.",
    ]),
    "de": ("Katja_Female_de-DE", "Microsoft Katja", [
        "Hallo Welt.", "Wie geht es dir?", "Bitte öffnen Sie die Einstellungen.",
        "Das System wird gerade aktualisiert.", "Dies ist ein Testsatz.",
    ]),
    "es": ("Alvaro_Male_es-ES", "Microsoft Alvaro", [
        "Hola mundo.", "¿Cómo estás?", "Por favor abre la configuración.",
        "El sistema se está actualizando.", "Esta es una frase de prueba.",
    ]),
    "pt": ("Francisca_Female_pt-BR", "Microsoft Francisca", [
        "Olá mundo.", "Como você está?", "Por favor abra as configurações.",
        "O sistema está sendo atualizado.", "Esta é uma frase de teste.",
    ]),
    "ja": ("Keita_Male_ja-JP", "Microsoft Keita", [
        "こんにちは。", "今日はいい天気ですね。", "設定を開いてください。",
        "システムは更新中です。", "これはテストの文章です。",
    ]),
    "ko": ("InJoon_Male_ko-KR", "Microsoft InJoon", [
        "안녕하세요.", "오늘 날씨가 좋네요.", "설정을 열어 주세요.",
        "시스템이 업데이트 중입니다.", "이것은 테스트 문장입니다.",
    ]),
}


EXTRA = {'zh': ['请把文件保存到桌面。', '明天的会议改到上午十点。', '这个软件需要更新。', '网络连接已断开。', '请重新启动计算机。', '电池电量不足，请充电。', '打印机没有响应。', '请在设置中启用自动更新。', '我们将在周五完成测试。', '谢谢你帮我解决问题。', '数据备份已完成。', '请输入正确的用户名和密码。'], 'en': ['Please restart your computer.', 'The network connection was lost.', 'Your battery is running low.', 'The printer is not responding.', 'Enable automatic updates in settings.', 'We will finish testing on Friday.', 'Thank you for your help.', 'Enter the correct password.', 'The file has been saved.', 'The meeting starts at ten.', 'My phone is charging.', 'The application is not installed.'], 'fr': ["Veuillez redémarrer l'ordinateur.", 'La connexion réseau est perdue.', 'La batterie est presque vide.', "L'imprimante ne répond pas.", 'Activez les mises à jour automatiques.', 'Nous terminerons le test vendredi.', 'Merci de votre aide.', 'Entrez le mot de passe correct.', 'Le fichier a été enregistré.', 'La réunion commence à dix heures.', 'Mon téléphone est en charge.', "L'application n'est pas installée."], 'de': ['Bitte starten Sie den Computer neu.', 'Die Netzwerkverbindung wurde getrennt.', 'Der Akku ist fast leer.', 'Der Drucker antwortet nicht.', 'Aktivieren Sie die automatischen Updates.', 'Wir werden die Tests am Freitag beenden.', 'Danke für Ihre Hilfe.', 'Geben Sie das richtige Passwort ein.', 'Die Datei wurde gespeichert.', 'Die Besprechung beginnt um zehn Uhr.', 'Mein Telefon wird geladen.', 'Die Anwendung ist nicht installiert.'], 'es': ['Reinicia el ordenador.', 'Se ha perdido la conexión de red.', 'La batería está casi vacía.', 'La impresora no responde.', 'Activa las actualizaciones automáticas.', 'Terminaremos la prueba el viernes.', 'Gracias por tu ayuda.', 'Introduce la contraseña correcta.', 'El archivo se ha guardado.', 'La reunión empieza a las diez.', 'Mi teléfono se está cargando.', 'La aplicación no está instalada.'], 'pt': ['Reinicie o computador.', 'A conexão de rede foi perdida.', 'A bateria está quase vazia.', 'A impressora não responde.', 'Ative as atualizações automáticas.', 'Terminaremos o teste na sexta-feira.', 'Obrigado pela sua ajuda.', 'Digite a senha correta.', 'O arquivo foi salvo.', 'A reunião começa às dez horas.', 'O meu telefone está a carregar.', 'A aplicação não está instalada.'], 'ja': ['コンピューターを再起動してください。', 'ネットワーク接続が切断されました。', 'バッテリーがほとんどありません。', 'プリンターが応答しません。', '設定で自動更新を有効にしてください。', '金曜日にテストを終えます。', '助けてくれてありがとう。', '正しいパスワードを入力してください。', 'ファイルは保存されました。', '会議は十時に始まります。', '携帯電話を充電しています。', 'アプリケーションはインストールされていません。'], 'ko': ['컴퓨터를 다시 시작하세요.', '네트워크 연결이 끊어졌습니다.', '배터리가 거의 없습니다.', '프린터가 응답하지 않습니다.', '설정에서 자동 업데이트를 설정하세요.', '금요일에 테스트를 마치겠습니다.', '도와주셔서 감사합니다.', '올바른 비밀번호를 입력하세요.', '파일이 저장되었습니다.', '회의는 열 시에 시작합니다.', '휴대 전화를 충전하고 있습니다.', '응용 프로그램이 설치되지 않았습니다.']}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summary = {}
    for lang, (pkg, voice, texts) in CASES.items():
        if only and lang != only:
            continue
        texts = list(texts) + list(EXTRA.get(lang, []))
        ok = 0
        n = 0
        for text in texts:
            eng = speak_capture(text, voice)
            if eng is None:
                print("[%s] MISS %s" % (lang, text))
                continue
            try:
                ours = synth(lang, text, pkg)
            except Exception as e:
                print("[%s] ERR %s: %s" % (lang, text, e))
                continue
            n += 1
            same = ours == eng
            ok += same
            print("[%s] %-32s %s (len %d/%d)" % (
                lang, text, "✓" if same else "✗", len(ours), len(eng)))
            if not same:
                # 打印简短差异
                for i, (a, b) in enumerate(zip(ours, eng)):
                    if a != b:
                        print("      diff@%d  ours=%d engine=%d" % (i, a, b))
                        break
        summary[lang] = (ok, n)
        print("[%s] 一致 %d/%d" % (lang, ok, n))
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
