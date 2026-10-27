import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.getcwd()
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_ids.js")
node_dir = os.path.join(ROOT, "node").replace(os.sep, "/")

tests = {
    "zh": ("Xiaoxiao_Female_zh-CN", ["你好，世界。", "人工智能语音合成。", "系统加载中，请稍候。",
                                     "本应阻止进程。", "本机变量字段格式无效",
                                     "本卷没有任何问题。", "杜普洛彦标点符号奇努克句号。",
                                     "来自服务器端的会话密钥具有空负载。",
                                     "来做为事件源",
                                     "来配置你的系统以满足你视觉、听觉以及行动的需要。",
                                     "来指示发生了错误，但为时已晚，无法采取补救措施。"]),
    "ja": ("Keita_Male_ja-JP", ["こんにちは。", "ありがとう。"]),
    "ko": ("InJoon_Male_ko-KR", ["안녕하세요.", "감사합니다."]),
    "en": ("Aria_Female_en-US", ["hello world."]),
    "fr": ("Denise_Female_fr-FR", ["bonjour le monde."]),
}

ok = bad = 0
for lang, (pkg, texts) in tests.items():
    pkg_path = os.path.join(os.environ["TEMP"], "AllVoice", pkg)
    for text in texts:
        py = subprocess.run(
            [sys.executable, "-c",
             "import sys,json;sys.path.insert(0,'.');from openshenzhou import TTS;"
             "t=TTS(%r);print(json.dumps(t.text_to_ids(%r),ensure_ascii=False))" % (pkg_path, text)],
            capture_output=True, encoding="utf-8", errors="replace")
        node = subprocess.run(["node", script, pkg_path, text, node_dir],
                              capture_output=True, encoding="utf-8", errors="replace")
        try:
            a = json.loads(py.stdout.strip().splitlines()[-1])
            b = json.loads(node.stdout.strip().splitlines()[-1])
        except Exception:
            print(lang, text, "解析失败")
            print("  py:", (py.stderr or "")[-200:])
            print("  node:", (node.stderr or "")[-200:])
            bad += 1
            continue
        same = a == b
        ok += same
        bad += (not same)
        print("%-3s %-18s py=%-4d node=%-4d %s" % (lang, text, len(a), len(b), "一致" if same else "不同"))
print("合计 一致 %d / 不同 %d" % (ok, bad))
