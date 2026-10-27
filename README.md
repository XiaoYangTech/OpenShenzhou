# OpenShenzhou（神州）—— 端侧神经语音引擎的可验证黑盒复刻

对 Windows 本地神经语音引擎（内部代号 **shenzhou / 神州**）的**运行时 + 文本前端**做黑盒逆向与等价重建：
不调用引擎、不联网，只用 ONNX Runtime 跑用户自备的语音包模型，并给出**逐采样 / 逐音素 id 可验证**的复刻证据。

> **两份正式文档（仅保留这两份）**
> - 论文：[docs/paper.md](docs/paper.md)
> - 帖子：[docs/blog-52pojie.md](docs/blog-52pojie.md)
>
> 本仓库**不包含**微软的模型/语音包数据；`models/` 与 `node/node_modules/` 均不入库，运行时使用本机解包出的语音包目录。

## 当前状态（可一键复现）

| 指标 | 结果 | 复现命令 |
|---|---|---|
| 模型层逐采样一致（8 语种 / 23 音色包） | 最大绝对误差 ~1e-6，相关 1.0000000 | 见论文 §4 |
| 中文 599 句自然句首句逐音素 id | **575/599 = 96.0%** | `python tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl` |
| 八语种 138 句整句逐 id | **138/138 = 100%** | `python tools/eval_ml_offline.py` |
| Python / Node 双端一致性 | 9/9 逐 id 一致 | `python tools/check_node_parity.py` |
| SDK 端到端自检 | **106/106 通过** | `python tests/test_all.py --voice-root "$TEMP/AllVoice"` |
| 一键全量验证 | 全部通过 | `python tools/verify_all.py` |

中文残余 24 句：约 12 句为长尾词界（引擎词表的合并/拆分差异）、4 句为语料乱码行、其余为极少数读音/标点。

## 仓库结构

```
openshenzhou/   Python SDK：容器解码、私有流式算子重写、8 语种文本前端、分块推理
node/           Node SDK（纯 JS + onnxruntime-node，与 Python 逐 id 一致）
data/           8 语种词表/符号/断词模型/CRF 映射等标定数据 + eval/ 离线评测 golden
tools/          数据构建、评测、验证工具（仅保留生产性脚本）
tests/          端到端自检（106 项）
docs/           仅保留：paper.md（论文）、blog-52pojie.md（帖子）
```

## 快速开始

```python
from openshenzhou import TTS
tts = TTS(r"D:\Voice\Xiaoxiao_Female_zh-CN")   # 语音包目录（.bin 模型 + phones.txt）
tts.speak("人工智能语音合成。", "out.wav")
```

```bash
# CLI
python -m openshenzhou --voice "D:\Voice\Xiaoxiao_Female_zh-CN" --text "你好，世界。" --out out.wav

# Node
cd node && npm install
node cli.js --voice "D:\Voice\Xiaoxiao_Female_zh-CN" --text "你好，世界。" --out out.wav
```

语音包从 Windows 系统自带的讲述人自然语音（MSIX/appx）中解包得到：把 `*.bin` 模型与 `phones.txt` 放进同一目录即可。
Linux / macOS 上可运行同一套 SDK（ONNX Runtime 支持三平台），前提是已有解包好的语音包目录。

## 分支

- `main`：完整单仓（研究数据 + 工具 + Python/Node SDK）
- `research`：论文/帖子 + 复刻数据 + 工具
- `python`：Python SDK + 数据 + 工具 + 测试
- `node`：Node SDK + 数据
