#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""harvest_zh_latin_engine.py —— 抓取引擎对中文语境拉丁字母/英文词的读音（探针数据来源）

通过 SAPI + NaturalVoiceSAPIAdapter 播放文本，用 Frida hook 引擎 encoder 入口
（RVA 0x169318），记录引擎实际使用的音素 id 序列。

输出 data/zh_latin_engine.json（build_zh_latin_model.py 的输入）：
  letters_single: 单字母（A..Z）隔离读音
  letters_pair:   字母串内每个字母（用 "A"+X 探针推导）
  words:          英文词/缩写读音

依赖：Windows + SAPI + NaturalVoiceSAPIAdapter（见 rea_work 安装记录）+ frida。
语音包目录通过 --voice-root 指定（默认 OS_VOICE_ROOT 或 D:/Desktop/deepseek/AllVoice）。

用法::
    python tools/harvest_zh_latin_engine.py --voice-root D:/Desktop/deepseek/AllVoice
    python tools/harvest_zh_latin_engine.py --resume   # 续抓（已有结果跳过）
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "zh_latin_engine.json")
ENGINE = "microsoft.cognitiveservices.speech.extension.embedded.tts.dll"
PS_EXE = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MATCH = "Microsoft Xiaoxiao"

JS = r"""
const ENGINE = __ENGINE__;
const HOOK_RVA = 0x169318;
function tryHook() {
  const mod = Process.findModuleByName(ENGINE);
  if (!mod) { setTimeout(tryHook, 40); return; }
  Interceptor.attach(mod.base.add(HOOK_RVA), {
    onEnter(a) {
      try {
        const n = a[2].toInt32();
        let ids = [];
        if (!a[1].isNull() && n > 0 && n < 200000) {
          ids = Array.from(new Float32Array(a[1].readByteArray(n * 4))).map((v) => Math.round(v));
        }
        send({ ids });
      } catch (e) { send({ ev: 'err', msg: String(e) }); }
    }
  });
}
tryHook();
"""

PS = r'''$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
$v = $voice.GetVoices() | Where-Object { $_.GetDescription() -like '*%(match)s*' } | Select-Object -First 1
if (-not $v) { exit 3 }
$voice.Voice = $v
$fs = New-Object -ComObject SAPI.SpFileStream
$fs.Format.Type = 22
$fs.Open('%(wav)s', 3, $false)
$voice.AudioOutputStream = $fs
$xml = @'
%(xml)s
'@
$voice.Speak($xml, 8) | Out-Null
$fs.Close()
'''

SINGLE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WORDS = ['hello', 'windows', 'office', 'excel', 'word', 'iphone', 'app', 'wifi',
         'ok', 'pdf', 'ai', 'id', 'ip', 'url', 'usb', 'cpu', 'api', 'http', 'qq',
         'test', 'music', 'video', 'email', 'online', 'update', 'system', 'computer',
         'phone', 'power', 'point', 'file', 'data', 'service', 'error', 'setup',
         'install', 'download', 'upload', 'click', 'start', 'stop', 'open', 'close',
         'name', 'user', 'admin', 'login', 'mail', 'web', 'site', 'note', 'code',
         'game', 'news', 'chat', 'shop', 'pay', 'map', 'info', 'help', 'about',
         'contact', 'blog', 'photo', 'book', 'movie', 'time', 'date', 'size',
         'type', 'mode', 'read', 'write', 'print', 'copy', 'delete', 'change',
         'search', 'view', 'show', 'hide', 'edit', 'format', 'window', 'button',
         'menu', 'title', 'page', 'list', 'item', 'group', 'team', 'work', 'home',
         'school', 'city', 'world', 'china', 'chinese', 'english', 'yes', 'no',
         'hi', 'bye', 'good', 'bad', 'big', 'small', 'fast', 'slow', 'new', 'old',
         'hot', 'cool', 'best']


def run(match, xml, tag, timeout=90):
    wav = os.path.join(os.environ["TEMP"], "zl_%s.wav" % tag)
    ps_path = os.path.join(os.environ["TEMP"], "zl_%s.ps1" % tag)
    with open(ps_path, "w", encoding="utf-8-sig") as f:
        f.write(PS % {"match": match, "wav": wav, "xml": xml})
    try:
        proc = subprocess.Popen([PS_EXE, "-NoProfile", "-ExecutionPolicy", "Bypass",
                                 "-File", ps_path])
    except Exception:
        return None
    import frida
    time.sleep(0.6)
    try:
        session = frida.attach(proc.pid)
    except Exception:
        return None
    hits = []

    def on_msg(m, d):
        if m.get("type") == "send":
            p = m.get("payload") or {}
            if "ids" in p:
                hits.append(p["ids"])

    script = session.create_script(JS.replace("__ENGINE__", json.dumps(ENGINE)))
    script.on("message", on_msg)
    script.load()
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < timeout:
        time.sleep(0.04)
    time.sleep(0.2)
    try:
        script.unload()
    except Exception:
        pass
    try:
        session.detach()
    except Exception:
        pass
    return hits[0] if hits else None


def rev_names(ids, sym):
    rev = {v: k for k, v in sym.items()}
    out = []
    for x in ids:
        out.append(rev.get(x, "?%d" % x))
        if x == 1:
            break
    return out


def core(ns):
    try:
        i = ns.index("-bos-")
        i2 = i + 1
    except ValueError:
        i2 = 0
    if i2 < len(ns) and ns[i2] == "br0":
        i2 += 1
    out = []
    for t in ns[i2:]:
        if t in ("br4", "~"):
            break
        out.append(t)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-root", default=os.environ.get("OS_VOICE_ROOT",
                                                           "D:/Desktop/deepseek/AllVoice"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "tools"))
    from _paths import sdk_root  # noqa: E402
    sdk = sdk_root()
    if os.path.isdir(os.path.join(sdk, "openshenzhou")):
        sys.path.insert(0, sdk)
    from openshenzhou import g2p_zh  # noqa: F401
    pkg = os.path.join(args.voice_root, "Xiaoxiao_Female_zh-CN")
    sym = {}
    for line in open(os.path.join(pkg, "phones.txt"), encoding="utf-8"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        n, i = line.rsplit(":", 1)
        sym[n] = int(i)

    raw = {"letters_single": {}, "letters_pair": {}, "words": {}}
    if args.resume and os.path.exists(OUT):
        raw = json.load(open(OUT, encoding="utf-8"))
    tasks = []
    for L in SINGLE:
        if L not in raw["letters_single"]:
            tasks.append(("letters_single", L, "L" + L))
    for L in SINGLE:
        if L not in raw["letters_pair"]:
            tasks.append(("letters_pair", "A" + L, "P" + L))
    for w in WORDS:
        if w not in raw["words"]:
            tasks.append(("words", w, "W" + w))
    print("待抓取", len(tasks), flush=True)
    for kind, text, tag in tasks:
        ids = run(MATCH, text, tag)
        if not ids:
            print("MISS", text, flush=True)
            continue
        c = core(rev_names(ids, sym))
        if kind == "letters_single":
            raw["letters_single"][text] = c
        elif kind == "letters_pair":
            X = text[1]
            if len(c) >= 2 and c[0] == "ey1" and c[1] == "-":
                raw["letters_pair"][X] = c[2:]
            else:
                print("PAIR-PARSE-FAIL", text, c, flush=True)
        else:
            raw["words"][text] = c
        json.dump(raw, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(text, "->", c, flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    main()
