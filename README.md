# OpenShenzhou（开放神州）· Node SDK

> **📌 本项目是「亿方听力大师」（ListextEditor）的附属项目。**
> 如果想快速体验本项目效果，或直接制作听力材料，推荐使用 **亿方听力大师** —— 全网最好用的听力材料制作工具，**完全免费**。


**微软自然语音推理引擎的本地、离线、跨平台复现（Node.js 版）。**
纯 JS + `onnxruntime-node`，不调用微软引擎、不联网、不需要 SAPI 注册，只用你自己解包的语音包模型；输出与 Python SDK **逐 id 一致**。

> 完整项目介绍、研究成果与数据说明见 `main` 分支的 README；本分支只包含 Node SDK 与运行所需数据。

## 特性

- **纯 JS + onnxruntime-node**：无需 Python，Windows / Linux / macOS 可用；
- **8 种语言、23 个音色包**：中、英、日、韩、西、法、德、葡；
- **与官方引擎逐采样一致**（误差 ~1e-6）；与 Python SDK 逐 id 一致（中文 899 句 + 英文 1682 句全对）；
- **文本前端高保真**：中文 599 句 98.0%、英文自然/压力/泛化集 99.6%~100%、八语种 golden 138/138。

## 环境要求

- Node.js 18+
- 依赖由 npm 安装：`onnxruntime-node`、`onnx-proto`、`protobufjs`

## 准备语音包

语音包来自 Windows 系统自带的「讲述人自然语音」（MSIX/appx），由你自行解包。目录内需包含：

```
am_v5_encoder.bin
am_v5_decoder.bin
device_vocoder_v6_streaming.bin
phones.txt
punc.txt
MSTTSLocZhCN.dat        # 中文必需
```

## 快速开始

```bash
cd node
npm install
```

```js
const { TTS } = require('./index');

(async () => {
  const tts = new TTS('<语音包目录>');
  await tts.speak('人工智能语音合成。', 'out.wav');
})();
```

命令行：

```bash
node cli.js --voice "<语音包目录>" --text "你好，世界。" --out out.wav
# 可选：--speed 1.2
```

### 多语音包 / 指定发音人 / 语速

```js
const { TTS } = require('./index');

(async () => {
  const zh = new TTS('<语音包根>/Xiaoxiao_Female_zh-CN'); // 发音人 = 语音包目录
  const en = new TTS('<语音包根>/Aria_Female_en-US');     // 可同时加载多个语音包

  await zh.speak('中文合成测试。', 'a.wav', 1.0);          // 语速：模型内 speed_ratios
  await en.speak('Hello world.', 'b.wav', 1.3);
})();
```

- **同时加载多个语音包**：new 多个 `TTS(目录)` 实例即可，各自独立；
- **指定发音人**：选择对应语音包目录（本机共 23 个音色包）；
- **语速**：`speed` 参数（模型内调节；实测 0.7/1.0/1.3 时长约 5.08s/3.48s/2.71s）；
- **无 SAPI、无服务器**：纯本地库 + CLI，不需要注册 COM/SAPI，也不含 HTTP 服务。

## 与其它分支的关系

| 分支 | 内容 |
|---|---|
| `main` | 研究成果：标定/评测数据（`data/`）、构建与评测脚本（`tools/`）、一键验证 |
| `python` | **Python SDK**：`openshenzhou/` + `tests/` |
| `node`（本分支） | **Node SDK**：`node/` 包（纯 JS + onnxruntime-node） |

做完整验证时，把三个分支内容合并到同一工作区后运行 `python tools/verify_all.py`（会同时检查 Python/Node 双端一致性）。

## 声明

- 本分支不包含任何微软的模型、语音包或引擎二进制文件；使用前需自行准备**正版语音包**；
- 本项目用于互操作、离线部署与学习研究；请遵守你所在地法律与微软相关许可。
