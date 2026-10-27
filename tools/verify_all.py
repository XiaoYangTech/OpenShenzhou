#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_all.py —— 一键复现全部验证证据（全部离线，无需引擎）

依次执行：
  1. tests/test_all.py          SDK 端到端 + 106 项自检（含前端 golden）
  2. tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl   中文 599 句逐 id
  3. tools/check_node_parity.py                                 Node≡Python 逐 id
  4. tools/eval_ml_offline.py                                   八语种逐 id（golden 来自引擎实测）

用法::
    python tools/verify_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(title, cmd, cwd=HERE, timeout=3600):
    print("\n==== %s ====" % title)
    print("$", " ".join(cmd))
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as e:
        print("运行失败:", e)
        return False
    out = (p.stdout or "") + (p.stderr or "")
    tail = [l for l in out.strip().splitlines() if l.strip()][-10:]
    print("\n".join(tail))
    return p.returncode == 0


def main():
    voice_root = os.path.join(os.environ.get("TEMP", ""), "AllVoice")
    ok = True
    ok &= run("1. 端到端测试（106 项，含前端 golden）",
              [sys.executable, os.path.join(HERE, "tests", "test_all.py"),
               "--voice-root", voice_root])
    ok &= run("2. 中文 599 句逐 id",
              [sys.executable, os.path.join(HERE, "tools", "eval_first_sentence.py"),
               os.path.join(HERE, "data", "eval", "zh_eval_599.jsonl")])
    ok &= run("3. Node≡Python 逐 id",
              [sys.executable, os.path.join(HERE, "tools", "check_node_parity.py")])
    ok &= run("4. 八语种逐 id（离线 golden）",
              [sys.executable, os.path.join(HERE, "tools", "eval_ml_offline.py")])
    print("\n==== 结论: %s ====" % ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
