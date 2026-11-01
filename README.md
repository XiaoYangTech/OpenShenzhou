# OpenShenzhou（开放神州）· Python SDK

> **📌 本项目是「亿方听力大师」（ListextEditor）的附属项目。**
> 如果想快速体验本项目效果，或直接制作听力材料，推荐使用 **亿方听力大师** —— 全网最好用的听力材料制作工具，**完全免费**。


**微软自然语音推理引擎的本地、离线、跨平台复现（Python 版）。**
不调用微软引擎、不联网、不需要 SAPI 注册，只用 ONNX Runtime 加载你自己解包的语音包模型；给定相同音素序列，合成音频与官方引擎逐采样一致。

> 完整项目介绍、研究成果与数据说明见 `main` 分支的 README；本分支只包含 Python SDK、自检与运行所需数据。

## 特性

- **纯本地 / 跨平台**：Python + onnxruntime，Windows / Linux / macOS 都能跑；
- **8 种语言、23 个音色包**：中、英、日、韩、西、法、德、葡；
- **逐采样一致**：与官方引擎音频最大绝对误差 ~1e-6、皮尔逊相关 1.0000000；
- **文本前端高保真**：中文 599 句逐音素 id 一致率 98.0%，英文自然/压力/泛化集 99.6%~100%，八语种 golden 138/138；
- **流式与分块推理**：复刻引擎的分块调度、卷积/加法缓存、逐块量化，低内存实时合成（4 核 CPU RTF≈0.113）。

## 环境要求

- Python 3.9+
- 依赖：`pip install numpy onnx onnxruntime pypinyin`

## 准备语音包

语音包来自 Windows 系统自带的「讲述人自然语音」（MSIX/appx），由你自行解包。目录内需包含：

```
am_v5_encoder.bin
am_v5_decoder.bin
device_vocoder_v6_streaming.bin
phones.txt
punc.txt
MSTTSLocZhCN.dat        # 中文必需（词表/读音/多音字模型）
```

（其他语言包结构类似，模型文件名以实际解包结果为准。）

## 快速开始

```python
from openshenzhou import TTS

tts = TTS(r"<语音包目录>")          # 例如解包出的 Xiaoxiao_Female_zh-CN 目录
tts.speak("人工智能语音合成。", "out.wav")
```

命令行：

```bash
python -m openshenzhou --voice "<语音包目录>" --text "你好，世界。" --out out.wav
# 可选：--speed 1.2
```

### 多语音包 / 指定发音人 / 语速

```python
from openshenzhou import TTS

zh = TTS(r"<语音包根>\Xiaoxiao_Female_zh-CN")   # 发音人 = 语音包目录
en = TTS(r"<语音包根>\Aria_Female_en-US")       # 可同时加载多个语音包

zh.speak("中文合成测试。", "a.wav", speed=1.0)   # 语速：模型内 speed_ratios
en.speak("Hello world.", "b.wav", speed=1.3)
```

- **同时加载多个语音包**：new 多个 `TTS(目录)` 实例即可，各自独立；
- **指定发音人**：选择对应语音包目录（本机共 23 个音色包）；
- **语速**：`speed` 参数（模型内调节；实测 0.7/1.0/1.3 时长约 5.08s/3.48s/2.71s）；
- **词边界事件**：`tts.word_events(text)` 返回与官方引擎一致的 WordBoundary 事件
  （`audio_ticks` / `duration_ticks` / `text_offset` / `text_length` / `word` / `type`），
  英文实测 19/19 案例逐字段一致；
- **无 SAPI、无服务器**：纯本地库 + CLI，不需要注册 COM/SAPI，也不含 HTTP 服务。

### 词边界事件（WordBoundary）

```python
from openshenzhou import TTS

tts = TTS(r"<语音包目录>")
for e in tts.word_events("Hello world. How are you?"):
    print(e["audio_ticks"], e["text_offset"], e["text_length"], e["word"])

# 500000 0 5 Hello
# 4500000 6 5 world
# 11625000 13 3 How
# ...
```

事件字段与官方嵌入式 SDK 一致：`audio_ticks`（100ns）、`duration_ticks`（引擎实测恒为 0）、
`text_offset`、`text_length`、`word`、`type`。

## 自检

```bash
python tests/test_all.py --voice-root "<语音包根目录>"   # 内含各语音包子目录
```

自检覆盖：数据完整性、8 语种 G2P 冒烟、8 语种合成冒烟、Node≡Python 一致性（需 Node SDK 分支）等。

## 与其它分支的关系

| 分支 | 内容 |
|---|---|
| `main` | 研究成果：标定/评测数据（`data/`）、构建与评测脚本（`tools/`）、一键验证 |
| `python`（本分支） | **Python SDK**：`openshenzhou/` + `tests/` + 运行所需数据 |
| `node` | **Node SDK**：`node/` 包（与 Python 逐 id 一致） |

做完整验证时，把三个分支内容合并到同一工作区后运行 `python tools/verify_all.py`。

## 声明

- 本分支不包含任何微软的模型、语音包或引擎二进制文件；使用前需自行准备**正版语音包**；
- 本项目用于互操作、离线部署与学习研究；请遵守你所在地法律与微软相关许可。
