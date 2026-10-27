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
| 中文 599 句自然句首句逐音素 id | **585/599 = 97.7%** | `python tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl` |
| 英文自然句 500 句逐 id（引擎探针标注） | **495/500 = 99.0%** | `python tools/eval_en_corpus.py data/eval/en_engine_500.jsonl` |
| 英文 TN 压力集 102 句逐 id | **99/102 = 97.1%** |
| 英文异读词句 80 句逐 id | **80/80 = 100%** |
| 英文 holdout 500 句（另外三本书，泛化评测） | **490/500 = 98.0%** | `python tools/eval_en_corpus.py data/eval/en_engine_tn.jsonl` |
| 八语种 138 句整句逐 id | **138/138 = 100%** | `python tools/eval_ml_offline.py` |
| Python / Node 双端一致性 | 中文 17 句 + 英文 1182 句逐 id 一致 | `python tools/check_node_parity.py` |
| SDK 端到端自检 | 全通过 | `python tests/test_all.py --voice-root <语音包根>` |
| 一键全量验证 | 全部通过 | `python tools/verify_all.py` |

中文残余 14 句：约 11 句为长尾词界拆分差异、3 句为语料乱码行。
英文下一步：继续扩词表（top-10k 词引擎标定）与标点/缩略语长尾；Node SDK 英文前端 v2 已与 Python 完全对齐。

## 仓库结构

```
openshenzhou/   Python SDK：容器解码、私有流式算子重写、8 语种文本前端、分块推理
node/           Node SDK（纯 JS + onnxruntime-node，与 Python 逐 id 一致）
data/           8 语种词表/符号/断词模型/CRF 映射/英文实测词表等标定数据 + eval/ 离线评测数据
tools/          数据构建、引擎标定、评测、验证工具（仅保留生产性脚本）
tests/          端到端自检
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

### 环境变量（研究工具用）

| 变量 | 作用 |
|---|---|
| `OS_VOICE_ROOT` | 语音包根目录（内含各 `*_Female_*` 语音包目录）；默认 `../AllVoice`，其次 `%TEMP%/AllVoice` |
| `OS_SDK_ROOT` | SDK 源码根（`openshenzhou/` 所在目录）。main 分支只放研究成果时，运行 `tools/` 需指向 python 分支的 checkout |
| `OS_ZH_DAT` | 可选：直接指定 `MSTTSLocZhCN.dat` 路径（默认从语音包目录查找） |

自然语音枚举需要一次性注册 SAPI 适配器（仅一次，之后零注册表写入亦可）：
`regsvr32 /s NaturalVoiceSAPIAdapter.dll`，然后启动引擎时用 `--voice-path=<语音包根>`。

## 分支（按交付物分门别类）

- `main`：**研究成果**——论文/帖子、复刻数据（`data/`）、工具（`tools/`）
- `python`：**Python SDK**——`openshenzhou/` + `tests/` + 数据
- `node`：**Node SDK**——`node/` + 数据

三个分支共享同一份 `data/`（git blob 去重，不额外占空间）；集成验证时把三者临时合并到工作区即可
（`re_integrate.sh` 或 `git worktree` + `git read-tree`）。
