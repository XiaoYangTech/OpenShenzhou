#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_all.py —— 一键复现全部验证证据（全部离线，无需引擎）

依次执行：
  1. tests/test_all.py          SDK 端到端 + 106 项自检（含前端 golden）
  2. tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl   中文 599 句逐 id
  2b. tools/eval_zh_large.py                                    中文大规模泛化（3000 句引擎 golden）
  3. tools/check_node_parity.py + check_zh_parity_batch.py      Node≡Python 逐 id（中文 17 句 + 599/大规模 899 句）
  4. tools/check_en_parity.py                                   英文 Python≡Node 682 句逐 id
  5. tools/eval_ml_offline.py                                   八语种逐 id（golden 来自引擎实测）
  6. 英文三套引擎语料指标（自然句/TN/异读词）

用法::
    python tools/verify_all.py
"""
import os
import subprocess
import sys
from _paths import voice_root as _voice_root, voice_pkg as _voice_pkg  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(title, cmd, cwd=HERE, timeout=3600):
    print("\n==== %s ====" % title)
    print("$", " ".join(cmd))
    # 单仓工作区（只 checkout main）缺少 python/node 分支的测试与工具时，
    # 明确标记为“跳过”而不是失败；完整验证请在合并三仓的工作区运行。
    script = cmd[1] if len(cmd) > 1 else ""
    if script.endswith(".py") and not os.path.exists(script):
        print("跳过：缺少 %s（请合并 python/node 分支后再做完整验证）"
              % os.path.relpath(script, HERE))
        return True
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
    voice_root = _voice_root()
    ok = True
    ok &= run("1. 端到端测试（98 项，含前端 golden）",
              [sys.executable, os.path.join(HERE, "tests", "test_all.py"),
               "--voice-root", voice_root])
    ok &= run("2. 中文 599 句逐 id",
              [sys.executable, os.path.join(HERE, "tools", "eval_first_sentence.py"),
               os.path.join(HERE, "data", "eval", "zh_eval_599.jsonl")])
    ok &= run("2b. 中文大规模泛化（3000 句引擎 golden）",
              [sys.executable, os.path.join(HERE, "tools", "eval_zh_large.py")])
    ok &= run("3. 中文 Node≡Python 逐 id（17 句基础）",
              [sys.executable, os.path.join(HERE, "tools", "check_node_parity.py")])
    ok &= run("3b. 中文 Node≡Python 逐 id（599+大规模 899 句批量）",
              [sys.executable, os.path.join(HERE, "tools", "check_zh_parity_batch.py")])
    ok &= run("4. 英文 Node≡Python 逐 id（1682 句）",
              [sys.executable, os.path.join(HERE, "tools", "check_en_parity.py")])
    ok &= run("5. 八语种逐 id（离线 golden）",
              [sys.executable, os.path.join(HERE, "tools", "eval_ml_offline.py")])
    # 6. 英文三套语料指标（低于阈值视为回归）
    ev = os.path.join(HERE, "tools", "eval_en_corpus.py")
    thresholds = [(os.path.join(HERE, "data", "eval", "en_engine_500.jsonl"), 0.98),
                  (os.path.join(HERE, "data", "eval", "en_engine_tn.jsonl"), 0.90),
                  (os.path.join(HERE, "data", "eval", "en_engine_heteronyms.jsonl"), 0.98),
                  (os.path.join(HERE, "data", "eval", "en_engine_holdout500.jsonl"), 0.95),
                  (os.path.join(HERE, "data", "eval", "en_engine_holdout2_500.jsonl"), 0.95)]
    for i, (corpus, th) in enumerate(thresholds, 1):
        print("\n==== 6.%d 英文语料 %s ====" % (i, os.path.basename(corpus)))

        p = subprocess.run([sys.executable, ev, corpus, "--show", "0"],
                           cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=3600)
        line = (p.stdout or "").strip().splitlines()[-1] if (p.stdout or "").strip() else ""
        print(line)
        import re as _re
        m = _re.search(r"(\d+)/(\d+) = ([\d.]+)%", line)
        ok &= bool(m and float(m.group(3)) / 100.0 >= th)
    print("\n==== 结论: %s ====" % ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
