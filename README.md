# OpenShenzhou（神州）—— 把 Windows 本地神经语音引擎「搬」进 Python/Node

一句话：**Windows 上有一个音质很好的本地语音引擎，但它只给 Windows 用、代码闭源、模型加壳。这个项目靠"看它跑、记答案、再自己写一套"的方式，把这套语音合成能力重新实现成纯 Python / Node 的 SDK。**

复刻是否成功不是靠"听起来像"，而是靠可验证的硬指标：同一个音素序列喂进去，合成音频和官方引擎**逐采样一致**（误差 ~1e-6）；同一个句子读出来，音素序列和官方引擎**逐 id 一致**。

> **两份正式文档（仅保留这两份）**
> - 论文：[docs/paper.md](docs/paper.md)（为什么、怎么做、证据、局限）
> - 帖子：[docs/blog-52pojie.md](docs/blog-52pojie.md)（面向爱好者的完整过程）
>
> 本仓库**不包含**任何微软的模型、语音包或引擎二进制；运行时使用你自己机器上解包出来的语音包目录。

## 为什么做这件事（大白话）

- **痛点**：这个引擎音质接近真人，但只能在 Windows 里通过系统接口用，Python/Node 程序、Linux 服务器、容器里都用不了；直接调用还要注册表、SAPI 适配器，部署麻烦。
- **为什么难**：推理代码是闭源 DLL，模型用加密容器分发，没有文档。社区常见做法是"用 SAPI 转接器间接调用"——还是离不开 Windows。
- **我们的思路**：不破解、不盗版、不上传任何东西。只在本机做三件事：
  1. **借模型**：引擎最终要把模型交给 ONNX Runtime 跑，我们半路把明文模型"接"下来（它本来就在自己进程里，只是没公开）。
  2. **重写推理**：把引擎私有的流式算子、分块解码、量化规则等价重写成标准 ONNX，自己跑。
  3. **复刻"读字"规则**：多音字怎么读、词怎么切、数字/标点怎么念——这些规则引擎没公开，就用**探针**把引擎当"答案机"，一句句问出标准答案，再用统计/模型去学。
- **目标**：让任何人只要机器上有正版语音包，就能在任意平台得到**和官方引擎一模一样**的语音；同时把每一步都做成可复现、可验证的证据。

## 数据从哪来（重要）

本仓库只放**小规模评测数据与标定数据**，所有数据的来源都是公开或本机自带的东西：

| 数据 | 来源 | 说明 |
|---|---|---|
| 被逆向的引擎 DLL | 微软公开的 Azure「嵌入式语音」SDK（NuGet 包 `Microsoft.CognitiveServices.Speech.Extension.Embedded.TTS`，0.2.10 x64） | 是微软公开分发的 SDK，不是破解产物；只用于"对答案"，SDK 运行时不依赖它 |
| 语音包（模型/词典/音素表） | Windows 系统自带的「讲述人自然语音」（MSIX/appx） | 需用户自备并解包；仓库不分发 |
| 中文界面语料（约 15 万句） | Windows 系统本地化资源里的界面文案 | 不是网上爬的聊天语料；599 句评测集是从中留出、训练全程没看过的"考试题" |
| 中文小说语料 | Project Gutenberg（古登堡计划）中文公版书（《红楼梦》等） | 用于大规模泛化评测与词表分析 |
| 英文语料 | Project Gutenberg 英文公版书 9 本：《爱丽丝梦游仙境》《白鲸》《双城记》《傲慢与偏见》《福尔摩斯》《德古拉》《科学怪人》《世界大战》《绿野仙踪》 | 自然 500 与 holdout 分别用不同的书，做"同分布 / 跨书 / 跨年代"三档泛化测试 |
| 英文 TN / 异读压力集 | 自建（人工构造句子 + 引擎标注） | 专测数字、日期、单位、缩略语、异读词 |
| 全部"标准答案"（音素 id） | 用探针让引擎逐句合成并记录 | 相当于请原作者逐题写答案；不在运行时使用引擎 |

> 以上第三方数据都只保留**评测所需的小样本与统计结果**，不随仓库分发整本书/整包语音。

## 当前状态（可一键复现）

| 指标 | 结果 | 复现命令 |
|---|---|---|
| 模型层逐采样一致（8 语种 / 23 音色包） | 最大绝对误差 ~1e-6，相关 1.0000000 | 见论文 §4 |
| 中文 599 句自然句首句逐音素 id | **587/599 = 98.0%** | `python tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl` |
| 英文自然句 500 句逐 id（引擎探针标注） | **498/500 = 99.6%** | `python tools/eval_en_corpus.py data/eval/en_engine_500.jsonl` |
| 英文 TN 压力集 102 句逐 id | **102/102 = 100%** |
| 英文异读词句 80 句逐 id | **80/80 = 100%** |
| 英文 holdout 500 句（另三本书，泛化） | **499/500 = 99.8%** |
| 英文 holdout2 500 句（另三本书/不同年代） | **492/500 = 98.4%** |
| 中文大规模泛化 3000 句（界面语料留出） | **2875/3000 = 95.8%** |
| 八语种 138 句整句逐 id | **138/138 = 100%** | `python tools/eval_ml_offline.py` |
| Python / Node 双端一致性 | 中文 899 句 + 英文 1682 句逐 id 一致 | `python tools/check_node_parity.py` / `check_zh_parity_batch.py` |
| SDK 端到端自检 | 全通过 | `python tests/test_all.py --voice-root <语音包根>` |
| 一键全量验证 | 全部通过 | `python tools/verify_all.py` |

残余差异说明（不影响"发音标准"）：
- 中文 12 句是「同音不同词界」——音素完全一样，只是词边界 token 不同，听感上只有轻微韵律差别；
- 英文 11 句是引擎内部韵律停顿与 LaTeX 上标标记，属引擎自身行为；
- 具体量化实验见论文 §6。

## 仓库结构

```
openshenzhou/   Python SDK：容器解码、私有流式算子重写、8 语种文本前端、分块推理
node/           Node SDK（纯 JS + onnxruntime-node，与 Python 逐 id 一致）
data/           8 语种词表/符号/断词模型/CRF 映射/英文实测词表等标定数据 + eval/ 离线评测数据
tools/          数据构建、评测、验证工具
tests/          端到端自检
docs/           仅保留：paper.md（论文）、blog-52pojie.md（帖子）
```

## 快速开始

```python
from openshenzhou import TTS
tts = TTS(r"<你的语音包目录>")   # 目录内需有 *.bin 模型与 phones.txt
tts.speak("人工智能语音合成。", "out.wav")
```

```bash
# CLI
python -m openshenzhou --voice "<语音包目录>" --text "你好，世界。" --out out.wav

# Node
cd node && npm install
node cli.js --voice "<语音包目录>" --text "你好，世界。" --out out.wav
```

语音包从 Windows「讲述人自然语音」（MSIX/appx）解包得到：把 `*.bin` 模型与 `phones.txt` 放进同一目录即可。
SDK 本身是纯 ONNX Runtime，Windows / Linux / macOS 都能跑，前提是已有解包好的语音包目录。

### 环境变量（研究工具用）

| 变量 | 作用 |
|---|---|
| `OS_VOICE_ROOT` | 语音包根目录（内含各 `*_Female_*` 语音包目录） |
| `OS_SDK_ROOT` | SDK 源码根（`openshenzhou/` 所在目录）；在只含研究成果的 main 分支运行工具时需要指向 python 分支 |
| `OS_ZH_DAT` | 可选：直接指定 `MSTTSLocZhCN.dat` 的路径（默认从语音包目录查找） |

## 分支（按交付物分门别类）

- `main`：**研究成果**——论文/帖子、复刻数据（`data/`）、工具（`tools/`）
- `python`：**Python SDK**——`openshenzhou/` + `tests/` + 数据
- `node`：**Node SDK**——`node/` + 数据

三个分支共享同一份 `data/`（git blob 去重，不额外占空间）；做完整验证时，把 python/node 分支内容合并进一个临时工作区
（`git worktree` + `git read-tree`）再运行 `tools/verify_all.py`。
