# OpenShenzhou：闭源端侧神经语音引擎的黑盒重建与逐采样一致性验证

**摘要**　云端语音合成（如 Edge TTS 一类在线接口）依赖网络与远端服务，存在限流、接口漂移与离线不可用等问题；Windows 上的本地神经语音引擎虽然可以离线工作，但其推理代码闭源、模型以加密容器分发，社区常见方案只能通过 SAPI 转接器在 Windows 上间接调用，无法移植到 Linux 或容器环境。本文以 Windows 本地神经语音引擎（内部代号 shenzhou）为对象，研究**在不加载原引擎代码、不访问网络的前提下复现其推理**的可行性，并以"与官方引擎逐采样一致"作为可验证目标。我们完成了以下工作：（1）通过 Hook ONNX Runtime 的 `OrtGetApiBase` 截获引擎交给推理器的明文模型，反推出模型容器的 40 字节头 + 90 字节循环异或格式；（2）复原 8 个流式私有算子（`StreamingConv`、`StreamingConvTranspose`、`StreamingAdd`、`StreamingPad` 等）的语义，并等价重写为标准 ONNX 算子；（3）复原解码器的分块调度、卷积/加法缓存与逐块量化规则，用"每块静态 shape 图"实现稳定复现；（4）设计"仅前端"标定探针（在 encoder 执行后主动让 decoder 的 `Run` 失败），以约 15 句/秒的速率把引擎当作标注器，重建中文文本前端（词表、读音、词边界模型、数字与符号规范化）。在 8 种语言、23 个音色包上的实验表明：当输入音素序列与引擎一致时，合成音频与引擎**逐采样一致**（最大绝对误差约 $10^{-6}$，皮尔逊相关 1.0000000）。进一步地，我们定位并**逐字节复刻**了引擎文本前端内部的 LSTM 断词模型（`CBlstmWordBreaker`）：通过进程内代码注入 Hook 截获其输出的逐字类别，自研前向与引擎在 325 个字符上类别 100% 一致；再用引擎语料学习其 Trie 后处理的"合并率"，中文自然句在 599 句全新留出集上的首句整句音素一致率由 82.5% 提升至 **97.7%（585/599）**；进一步定位并复刻了引擎的多音字读音决策组件（87 个 CRF 模型）：解出 A 段 Trie 的 varint 编码、24 位大端子指针与提交掩码，离线重实现特征查找与权重打分，并动态探测出 80/87 个模型对应的汉字，纯汉字词界整句一致率达 98.27%；其中「一」变调等读音规则由 5.4 万句语料统计得到。进一步对「断词类别 → 词序列」的后处理做定量复刻：在语料抽样上我们的类别向量与引擎逐字一致（2999 句、0 差异），而合并/拆分后处理在 599 句留出集上有 585 句（97.7%）与引擎词序列完全相同：合并率缺失时以「独立词证据」（jieba 词典、引擎 .dat 词表、引擎语料实际输出词）兜底，消除了「来|做」「的|需要」这类非词误合并；残余 14 句为长尾拆分差异（如「终结|点」「计|数器」「加密|算法」）与 3 句语料乱码行。同一方法还复刻了日语的 LSTM 断词模型（54/54 字符类别一致）。系统在 4 核 CPU 上实时率 RTF≈0.113（约 8.8 倍实时），常驻内存约 375 MB，且不含任何 Windows 专有依赖。本文还给出了对目标引擎内部模型（87 个多音字读音 CRF）的逆向进展，并讨论了与微软亚洲研究院 FastSpeech 系列、LightSpeech 等工作的对应关系。

**关键词**　神经语音合成；ONNX；INT8 量化；流式推理；黑盒逆向；差分测试；逐采样一致性；韵律边界

---

## 1 引言

### 1.1 动机

神经语音合成（TTS）的工程落地通常有三条路线。

**（1）云端服务。** 调用在线 TTS 接口（如社区广泛使用的 `edge-tts` 一类工具）。优点是音质好、无需本地算力；缺点是必须联网、受服务端限流与接口策略约束，在离线设备、内网环境、隐私敏感场景中不可用。社区中"Edge TTS 不稳定"的抱怨大多源于此：它不是模型问题，而是**服务依赖**问题。

**（2）系统语音栈 + 转接器。** Windows 自带本地神经语音引擎，社区通过 SAPI 转接器（如 NaturalVoiceSAPIAdapter 一类项目）把它包装成系统音色，使传统应用可以调用。该路线可离线，但存在两个结构性限制：**只能 Windows**，且核心推理引擎是闭源 DLL，模型以加密容器分发，无法移植、无法调试、无法在 Linux/容器中部署。

**（3）自研模型。** 使用 VITS、FastSpeech 等开源模型自行训练，可控性最好，但需要数据与训练资源，且音色、发音与目标系统不一致。

本文研究的是第四条路：**不调用原引擎、不联网，直接加载语音包中的模型并复现其推理**，使结果与官方引擎**逐采样一致**。这样得到的系统既能离线部署、跨平台移植，又能作为官方引擎的"开源替身"用于回归测试与二次开发。

### 1.2 为什么必须"逐采样一致"

如果只要求"听起来像"，可以用任何近似实现。但工程上有两个理由要求严格一致：

1. **可替代性**：作为 drop-in 替换时，任何时长、能量、发音细节的漂移都会在下游（字幕对齐、口型、音频拼接）中引入回归；"听起来差不多"无法用于验收。
2. **可验证性**：逆向过程中如果没有一个客观的一致性指标，就无法判断某个算子语义、某个缓存策略、某个量化位置是否正确；而 `max|Δ|`、相关系数、音素 ID 一致率都是可量化、可自动化的。

因此本文把目标拆成两层：

- **模型层**：给定与引擎相同的音素序列（token 序列），输出音频与引擎逐采样一致；
- **前端层**：中文文本→音素序列的转换与引擎逐 ID 一致。

前者决定"声音像不像"，后者决定"读得对不对"。

### 1.3 挑战

- **模型被加密**：语音包中的 `.bin` 熵接近 8，不可直接解析；引擎调用的加密 API 链条复杂且随版本变化。
- **私有算子**：模型使用 8 个 `test.customop` 流式算子，标准 ONNX Runtime 无法执行；必须恢复其精确语义。
- **流式状态**：引擎分块解码并跨块传递卷积/加法缓存，任何缓存或量化位置偏差都会沿时间轴累积。
- **前端复杂**：中文存在分词、多音字、数字/符号规范化、韵律断句等规则，且不同音色包的符号 ID 完全不同。
- **黑盒**：没有源码、没有文档、没有训练数据，只有可执行的引擎和语音包。

### 1.4 贡献

1. 给出目标引擎的模型容器格式（40 字节头 + 90 字节循环异或），并说明"Hook 推理器终点"比"逆向加密算法"更稳健的工程理由；
2. 复原 8 个流式私有算子的语义，等价重写为标准 ONNX 算子，并提出**每块静态 shape 图**的方案解决 ONNX Runtime 动态 shape 冲突；
3. 复原分块调度、缓存语义与逐块量化，实现模型层与引擎**逐位/逐采样一致**（$10^{-6}$ 量级）；
4. 设计"仅前端"标定探针，把引擎变成可批量查询的标注器；据此重建中文前端的数据与读音词表；
5. 定位并精确复刻引擎内部的 LSTM 断词模型（`CBlstmWordBreaker`）：解出 `.dat` 中的模型块布局、字符→嵌入行映射、LSTM/DNN 权重与窗口调度；用**进程内代码注入 Hook** 把引擎内部逐字类别读出来做逐字节验证（中文 325/325、日语 54/54 字符一致）；
6. 复刻断词模型的 Trie 后处理：由类别统计推出"类别∈{0,3} 为词起始"的切分规则，并用引擎语料学习每个候选词的合并率，使中文 599 句首句整句音素一致率达到 **97.2%（582/599）**（词界单独评测 98.27%）；
7. 在 8 种语言、23 个音色包上给出量化评估，说明 8 种语言中仅中文/日语使用 LSTM 断词，并公开讨论残余差异的来源与改进路径；
8. 给出目标引擎内部读音模型（87 个 CRF）与韵律断句 CRF（BR0/BR2）的逆向进展：解出容器格式、标签/模板/权重分区，并在运行内存中逐字节验证。

---

## 2 背景与相关工作

### 2.1 在线 TTS 与本地 TTS

在线 TTS 以服务形式提供神经语音，客户端只负责传输与播放。其工程问题包括：网络抖动导致合成失败、服务端限流导致批量任务不可行、接口变更导致客户端失效。本地 TTS 将模型部署在终端，天然离线，但受本地算力与平台绑定限制。

### 2.2 Windows 本地神经语音与 SAPI 转接器

Windows 的本地神经语音由 `Microsoft.CognitiveServices.Speech.extension.embedded.tts.dll` 提供，其内部链接 ONNX Runtime 1.15.1，并注册了 8 个自定义算子。语音包以 appx 分发，内含 `am_v5_encoder.bin`、`am_v5_decoder.bin`、`device_vocoder_v6_streaming.bin` 三个模型与语言资源 `.dat`。社区转接器通过 SAPI 把该引擎包装成系统音色，但受限于 Windows 与闭源引擎。

### 2.3 端侧神经 TTS：微软亚洲研究院的工作线

目标引擎的架构与微软亚洲研究院（MSRA）在非自回归 TTS、端侧轻量 TTS 上的公开工作高度相关，这些论文对逆向过程中的"结构猜测"非常有价值：

- **FastSpeech**（Ren et al., 2019）：基于 Transformer 的非自回归 TTS，用时长预测与长度调节器并行生成 mel 谱，显著快于自回归模型；
- **FastSpeech 2 / 2s**（Ren, Hu, Tan, Qin, Zhao, Liu, 2020）：直接以真实语音为训练目标，引入 variance adaptor（duration / pitch / energy 预测器），并进一步提出 **FastSpeech 2s**——直接由文本/音素生成波形（text-to-waveform），以降低推理延迟；
- **LightSpeech**（Luo, Tan, Wang, Qin, Li, Zhao, Chen, Liu, 2021）：用神经网络结构搜索设计面向端侧（手机/嵌入式）的轻量 TTS，相比 FastSpeech 2 达到 15× 压缩、6.5× CPU 加速且音质相当；
- **NaturalSpeech**（Tan et al., 2022）：面向"人类水平音质"的端到端 TTS；
- MSRA 的 ISCSLP 2021 教程与 Efficient ML 系列报告：系统总结了 TTS 的轻量化、量化、端侧部署实践。

我们在第 5.4 节会说明：逆向得到的引擎结构与这些工作一致（时长预测 + 长度调节 + 轻量 INT8 模型 + 流式因果推理），但**生产引擎为流式做出了不同取舍**（帧级循环解码器、带 GRU 状态的流式声码器）。本文不预设"引擎=某篇论文"，而是把这些论文作为理解与验证的参照。

### 2.4 ONNX Runtime 与 INT8 量化

ONNX Runtime（ORT）是微软开源的跨平台推理引擎；目标引擎正是通过它执行模型。端侧部署常用动态 INT8 量化（`DynamicQuantizeLinear`、`MatMulInteger`、`DynamicQuantizeLSTM`）在几乎不损失音质的前提下压缩模型、加速 CPU 推理。本文的复现同样基于 ORT CPU ExecutionProvider，因此模型层的一致性可以做到"同算子、同精度、同结果"。

### 2.5 差分测试与黑盒逆向

黑盒逆向的一个核心方法是**差分测试**：构造同一输入，分别喂给目标系统与复现系统，比较输出/中间结果，逐步定位差异来源。本文大量使用该方法：逐层比较中间张量定位算子语义、逐块比较定位缓存策略、逐句比较定位前端差异。

---

## 3 目标引擎的黑盒观察

### 3.1 语音包结构

解包一个中文音色 appx 后：

```
am_v5_encoder.bin                 2.1 MB     # 音素 → 帧级隐状态 + 时长
am_v5_decoder.bin                 0.7 MB     # 帧级隐状态 → 80 维 mel（流式）
device_vocoder_v6_streaming.bin   2.3 MB     # mel → 24 kHz 波形（流式）
phones.txt / punc.txt                        # 符号表
MSTTSLocZhCN.dat                  48.9 MB    # 中文前端资源（词表、读音模型等）
ZhCN.address.dat / ZhCN.message.dat / ...     # 领域资源
```

### 3.2 引擎组件

`embedded.tts.dll`（x64，约 5.9 MB）内部注册了 8 个 domain 为 `test.customop` 的算子（见 §4.2）。RTTI 中可见完整的文本前端类族：

```
CCRFWordBreaker / CCRFWordBreakerImp / CNeuralCRFWordBreaker / CBlstmWordBreaker
CCRFBreakTagger / CCRFBreakTaggerImpl / CCRFPitchAccentLocTagger / CCRFEmphasisTagger
CCRFPolyphonyTagger / CCRFSentSepTagger / CCRFSentTypeDetector / CCRFModelManager
CProsodyBreakCARTPredictor / CProsodyRNNBreakTagger / CWordPronouncer / CTTSNgramTrieBreaker
```

这些类名直接告诉了我们引擎的前端组成：词切分、韵律断句、音高重音、强调、多音字、句子切分等。

### 3.3 逆向得到的模型架构

（下表来自对三套 ONNX 计算图的直接统计，而非猜测。）

| 模型 | 输入 | 输出 | 主要结构 | 量化 |
|---|---|---|---|---|
| `am_v5_encoder` | 音素 id `[1,N]`、`speed_ratios` `[1,N]` | 帧级隐状态 `[1,T,256]`、每个音素的 `prediction_durations` | 13×Conv(k=3)、4×双向 LSTM(hidden 128)、`CumSum/Range/If/Sigmoid`（时长预测 + 长度调节） | 动态 INT8（DQ-LSTM / MatMulInteger） |
| `am_v5_decoder` | 帧级隐状态 `[1,T,256]`、`h1/c1/h2/c2`（2 层 LSTM，hidden 128） | 80 维 mel `[1,T,80]`、新状态 | 6×Conv(k=3)、2×LSTM（帧级循环、跨块缓存） | 动态 INT8 |
| `device_vocoder_v6_streaming` | mel `[1,80,T]`、两个 GRU 状态（128/64） | 波形 `[1,1,T·256]`、新状态 | WaveNet 式空洞卷积（k=3，dilation 1/3/9；k=7 前置、k=63 后置）、4×`ConvTranspose`（256 倍上采样）、`LeakyReLU`、2×GRU 流式状态 | FP32 |

由此可以判断：

- 该引擎属于 **FastSpeech 系（非自回归 + 时长预测 + 长度调节）的端侧流式变体**；
- 生产实现为了**低延迟流式**，把声学解码器做成了帧级循环（带 LSTM 状态缓存），而不是 FastSpeech 2 的并行 Transformer 解码器；
- 声码器是**流式因果卷积 + 转置卷积上采样 + GRU 状态**的结构，与 WaveNet/Parallel WaveGAN 一脉相承，但为流式做了因果化与状态缓存；
- INT8 动态量化集中在 encoder/decoder（LSTM/MatMul），声码器保持 FP32——这与端侧部署中"重排密集层量化、卷积层保精度"的常见取舍一致。

这些观察与 §2.3 的公开工作互相印证：FastSpeech 2 的 variance adaptor、LightSpeech 的端侧轻量化、INT8 动态量化的部署实践，在引擎里都能找到对应物。**论文给了我们"可能的结构空间"，逆向给了我们"这个引擎实际选了哪种"。**

### 3.4 运行时行为

- 一次合成按"句子"切分，每句做一次 encoder，然后分块 decode + 流式 vocode；
- 解码分块调度（实验测得）：首块 64 帧（短句取 `min(32,T)`）；之后剩余 >64 切 64、剩余 >32 切 32，最后一块执行 flush；
- 每块解码后立即量化，再送入声码器；
- 声码器的输入为解码 mel 的转置乘 0.25，布局 `[1,80,T]`。

---

## 4 复现方法

本节按"实际动手顺序"叙述，重点说明**每一步为什么这样做、试过哪些替代方案、如何验证**。

### 4.1 模型获取：Hook 推理器而不是破解加密

**问题**：`.bin` 看起来是加密的。

**替代方案 A：逆向加密算法。** 我们确实钩到了引擎调用的 `BCryptGenerateSymmetricKey` / `BCryptDecrypt`，但评估后放弃：密钥派生链条复杂、与版本绑定、跨音色/跨版本迁移成本高。

**采用方案 B：截获推理器输入。** 无论容器怎么加密，引擎最终必须把**明文 ONNX protobuf** 交给 ONNX Runtime。于是我们在进程内 Hook `OrtGetApiBase`：

1. 拦截 `OrtGetApiBase`，返回一张自定义的 `OrtApiBase`；
2. `GetApi` 返回一张复制的 `OrtApi` 函数表，其中只改写需要观察的槽位：
   - `CreateSessionFromArray(env, model_data, len, ...)`：把 `model_data` 落盘；
   - `Run(session, run_opts, input_names, inputs, n, output_names, n, outputs)`：记录输入输出，用于差分对拍；
3. 其余槽位原样转发，保证引擎行为不变。

**验证**：第一次运行就抓到了完整的 ONNX 明文（首字节 `0x08`，protobuf field 1）。

**回到容器**：拿到明文后做已知明文攻击：`bin ⊕ onnx` 呈 90 字节周期 → 确认是循环异或（而非 AES）。密钥直接读出，内容为

```
ZCjZ7nHDSLvf4gpELteM4AnzaWUjTpn7UkV7D@vvksl0w1SNgon6d1905WANbktDc9S39oaA4r29HJNayXvTq8fJsq
```

23 个语音包交叉验证得到统一容器格式：

$$
\text{file} = \underbrace{[16B\ \text{hash}][16B\ 0][u32\ \text{payloadSize}][u32\ 0]}_{40\ \text{字节}}
,\qquad
\text{payload} = \text{ONNX} \oplus \text{key}^{\infty}.
$$

**方法论小结**：逆向加密算法是"正面强攻"，Hook 推理器是"截其必经之路"。后者的鲁棒性来自系统约束（模型必须交给 ORT），与加密实现无关。

### 4.2 私有算子：语义等价与差分验证

明文 ONNX 中有 8 个私有算子（domain `test.customop`）：

| 算子 | 语义 |
|---|---|
| `StreamingConv` | 维护 2 帧卷积历史的因果卷积 |
| `StreamingConvTranspose` | 因果转置卷积 |
| `StreamingAdd` | 两路流式张量按"低延迟优先"对齐相加 |
| `StreamingPad` | 带控制输入的补零（首块/中间块/收尾块不同） |
| `StreamingSplit` / `StreamingSlice` | 流式切分/切片 |
| `StreamingGather` | 流式索引 |
| `StreamingRound` | 跨块一致的量化取整 |

**为什么不能"近似实现"**：流式推理的误差会沿时间轴累积。缓存少 1 帧、补零方向错、块边界量化不一致，都会让后续所有帧偏离；即使听感差异很小，也无法满足"逐采样一致"。

**等价重写**。以卷积为例，设输入为 $x$，缓存为最近 2 帧：

$$
\text{StreamingConv}(x) = \text{Conv}\big([\text{cache}_2(x); x], W\big)[2:],\qquad
\text{cache}_2(x)=x_{-2:}.
$$

首块与中间块左侧补零为 `[0,0]`，flush 块补零为 `[0,1]`；首块输出需切片。`StreamingAdd` 的低延迟分支对齐规则同样分三种模式：首块直接截断、中间块"拼接缓存后截断"、flush 块"拼接后不截断"。

**验证**：用同一音素序列分别跑引擎与我们的图，逐层比较中间张量；直到 vocoder 输入最大绝对误差达到 $4\times10^{-6}$。

**工程坑：动态 shape**。ORT 对带控制输入、缓存读写、切片位置随块变化的图支持有限，符号 shape 一冲突就报 `'onnx::Add_164_g2' is not a graph input` 之类错误。我们试过统一动态图、试过 padding 到固定长度，都不稳定。最终采用**每块静态 shape**：按块类型/长度预生成若干 ONNX 图（64 帧、32 帧、flush、短句首块等），运行时按调度选图。代价是图变多，收益是彻底稳定、且可逐位复现。

### 4.3 流式对齐：调度、缓存、量化

**调度**：在 `Run` 钩子里记录每次 decoder 的输入长度，对不同长度句子观察，得到 §3.4 的分块规则。

**缓存**：`StreamingConv` 的 2 帧历史、`StreamingAdd` 的低延迟对齐、首块/中间块/flush 三种补零模式，都需要逐一试出。我们把每种模式实现成独立分支，用逐层张量对拍确认。

**量化**：引擎在每块解码完成后立即做 `Round` + 缩放，再送声码器；这一"逐块量化"必须复刻，否则跨块误差累积。

**结果**：vocoder 输入最大误差 $4\times10^{-6}$，最终音频最大误差约 $10^{-6}$（见 §5.2）。

### 4.4 文本前端

#### 4.4.1 符号名抽象：为什么不能用 ID

不同音色包的符号 ID 完全不同（中文两音色仅 62/193 个 ID 相同，日文仅 3/130，葡语 0/224）。若标定数据以 ID 存储，换音色即失效。因此所有标定数据以**符号名**存储（如 `yi`、`i_l`、`i_h`、`br1`），运行时按目标音色包的 `phones.txt` 解析为 ID。

#### 4.4.2 "仅前端"标定探针：把引擎变成标注器

正常合成一句需要约 1.4 s，采集 10 万句不可行。我们利用 §4.1 的钩子设计了一个"只跑前端"的探针：

1. 令 encoder 的 `Run` 正常执行（它的输入就是目标音素序列）；
2. 对之后 decoder/vocoder 的 `Run`，直接返回一个 ORT 错误状态（用 `OrtApi::CreateStatus` 构造）；
3. 引擎捕获错误并中止后续计算，SAPI 返回失败，但 encoder 输入已落盘。

效果：约 **15 句/秒**，比完整合成快约 20 倍；双进程并行可达 20+ 句/秒。由此构建了「中文文本 → 引擎音素序列」的数据集（词表标定、读音标定、韵律标定、验证集）。

#### 4.4.3 词表与读音

- **核心词表** 78,962 条：每条由引擎实测得到词内分词结构与读音；
- **扩展词表** 41,455 条：来自成语库、短语词典与歌词，逐条送入引擎实测；
- **单字读音表** 20,600 字（CJK 基本区全覆盖）；
- **拼音→音素映射** 1,375 条：用单字读音反推，修正规则表在 `iong`、`ei` 等韵母上的错误；
- **多音字上下文表** 215 字 / 7.2 万条 n-gram（(左字, 本字, 右字)→读音）。

#### 4.4.4 词边界（br1）：官方 LSTM 断词模型的识别与精确复刻

**先纠正一个早期结论**：br1 不是"上下文相关的韵律短语边界"，而是**词边界**；
引擎在文本前端内部用 `CBlstmWordBreaker`（双向 LSTM + DNN，4 类输出）预测词边界，
再经 **Trie 词典后处理** 完成最终分词。证据链如下。

**(1) 模型定位**。每个语音包 `MSTTSLoc<Lang>.dat` 开头即是一块 LSTM 断词模型：
`[+0xD8] offLayers/offX/offEmbedding/rows/cols` 头，`[+0x110] cols=64, classes=4, layers=2`，
BLSTM 配置 `+0x130`、前向/反向 LSTM 配置 `+0x150 / 0x208C0`、DNN `+0x41030`，
权重分别位于 `0x1C0(bf) / 0x5C0(Wf) / 0x20930(bb) / 0x20D30(Wb) / 0x41050(bo) / 0x41060(Wo) / 0x41868(emb)`。
运行时读取 `CBlstmWordBreaker` 对象，其权重指针恰为 `dat+0x41868`，rows=20,947、cols=64，与静态解析一致。

**(2) 算子语义**。用 Ghidra 逆出矩阵乘加内核与 LSTM 单元：`pre = [x;h]·W + b`（W 行=输入 128、列=输出 256，
按列优先存储）、门序 `[i,f,g,o]`、peephole 全零、窗口长度 20、双向输出拼接后接 DNN，
最后逐帧 argmax 得到 4 类输出。

**(3) 逐字节验证（进程内 hook）**。把引擎内部的向量 erase 函数入口改成绝对跳转，
注入 shellcode 记录 `(返回地址, vector 指针, start, count)` 并把 vector 内容复制到固定内存；
合成一句后读回，即得到**引擎内部逐字类别**。与自研前向对比：
40 句纯汉字、325 字符，**类别 100% 一致**（因此网络部分无任何误差）。

**(4) 后处理规则**。统计类别对与边界的关系发现：类别 1/2 = 词内延续，0/3 = 词起始，
故词界 = 在 `class∈{0,3}` 的字符之前切；随后按官方词表做相邻词合并，
每个候选词是否合并由**引擎语料统计出的合并率**决定（例：用于→常拆开、备份→常合并）。
该两步在纯汉字留出集把词界整句一致率从 95.1% 提升到 **98.27%**。

**(5)** 早期训练的哈希逻辑回归代理模型（前 3 字/后 3 字、n-gram、词典基底特征，48k 句）
已被官方模型取代，仅作为对照保留在 `tools/` 中。

**(6) 跨语言**：8 种语言中仅中文与日语使用 `LSTMWordBreaker`。
日语 `MSTTSLocJaJP.dat` 结构与中文一致（rows=21,214），字符→嵌入行号映射由
反编译 `FUN_1802548ec`/`FUN_180254c78` 的 ja-JP 分段公式得到，
同样用 hook 验证 **54/54 字符类别 100% 一致**。日语剩余差距是音高重音记号模型（§6）。

#### 4.4.5 多语种词表重标定与"哨兵屏障"协议

非中文语种的词表最初来自单音色小规模合成，存在两类问题：（i）词内音节切分缺失（英语、德语）；（ii）解码表用错（葡语词表被解成 `en-us_*` 符号堆叠）。

重标定时需要"逐词合成 + 探针抓取 ids"，但引擎推理是异步的：批量请求下探针文件出现顺序与请求顺序不一致，会让 A 词的 ids 被当成 B 词的读音，造成**静默数据污染**。为此我们设计"哨兵屏障"协议：
每处理一个词后立即请求固定哨兵文本 `zzzqqq`，按文件名计数器排序后，
哨兵文件的前一个文件即该词 ids；找不到哨兵则重试。该协议把 703 词德语词表重标定做到 0 失败：
`hallo → h a - l ow1`、`system → z uy s - t ey1 m`、`gerade → g ax - r aa1 - d ax`。

同时修正了非中文语种的音节切分规则：引擎对 OOV 词也按词典规则插入词内 `-`
（英文按最大 onset 原则，如 `settings → s eh1 - t ih ng z`），SDK 已实现同样规则（Python/Node 双端）。

#### 4.4.6 数字与符号规范化

按引擎实测规则实现 `tn_symbols()`：`℃`→摄氏度、`￥`→元、`～`→至、`×`→乘以、`÷`→除以、`%`→百分之…、日期→年月日、时间→点X分。9 项实测中 7 项与引擎逐 ID 一致。Python 与 Node 两端保持一致（新增数字读法表保证声调与拼音短语词典一致）。

### 4.5 内部读音模型（CRF）逆向进展

`MSTTSLocZhCN.dat` 中解析出 87 个 `TD\0\0 + GUID` 标签的模型容器，格式为：

```
TD\0\0 | GUID(16B) | 长度 | 常量 | "CRF\0" | ver=100 | size | nFeatures | nLabels |
nTemplates(=24) | lenA | lenB | hsz(=nLabels²×4) | 1.0f | 0.0f |
nLabels×u32(label 串偏移) | nTemplates×u32(模板串偏移) | A段 | B段 | C段
```

每个模型带有自己的 UTF-16 字符串表（`U00:` 的位置减去模板[0]偏移即基址）。解得：

- 标签是候选读音（如「卷」：`JV%A_H%NN_L`=juǎn、`JV%A_L%NN_L`=juàn、`QV%A_L%NN_H`=quán）；
- 模板固定 24 条，全部围绕 Token 的上下文（`Token.PreviousCharacters.GetCharacter(k)`、`Token.NextToken...` 等）；
- **A = 特征索引结构，B = 权重（float32，每项 L 个、成 ± 对），C = nLabels×nLabels 转移矩阵**。

在此基础上，我们用 ctypes 直接读取运行中的引擎进程内存，找到 **113 个已解析模型对象**，其数据块与 `.dat` 对应段**逐字节相同**（例如模型 8：A/B/C 在堆中连续排列，长度 460/296/16，恰为 `.dat` 的 `lenA/lenB/hsz`）。这为下一步"导出官方读音 CRF 并在 SDK 内实现 Viterbi"提供了基础。

**关于 br1 模型**：文中 §4.4.4 已确认通用中文断词由 `CBlstmWordBreaker` 完成并已精确复刻；
`MSTTSLocZhCN.dat` 中的 87 个 CRF 均为**读音/韵律**模型（多音字、轻声、变调），
`ZhCN.address.dat` / `ZhCN.message.dat` 为领域词表。系统 `MSTTSLoc_OneCore.dat` 中的
BR0/BR2 两个 CRF 的模板是 `Word.PrevWord.Text / Word.POS / Word.SyllableNumber` 等**词级韵律断句特征**，
负责 br3/br4 等韵律层级，与分词无关。

### 4.6 跨语言与工程实现

- **语言差异**：中文/英/法/德/西/葡使用同一套 frame 格式；日文 Keita、韩文 InJoon 使用另外两种格式；葡语 Francisca 为平铺音素流。每种语言的音素化器独立实现。
- **Python SDK**：仅依赖 `onnxruntime`、`numpy`、`pypinyin`（可选 `jieba`）；
- **Node SDK**：仅依赖 `onnxruntime-node`，实现同一套前端与调度；
- **appx 加载**：支持直接传 `.appx/.msix/.zip`（自动解包到缓存、自动识别语言），也支持已解包目录；
- **跨平台**：代码中无 Windows 专有调用（无 DLL、无 SAPI、无 COM、无注册表），模型为平台无关 ONNX。

---

## 5 实验与评估

### 5.1 实验设置

- **硬件**：Windows 10 x64，4 核 CPU，8 GB 内存（开发与对照环境；SDK 本身无 Windows 依赖）；
- **音色包**：8 种语言、23 个音色包（中、英、日、韩、西、法、德、葡）；
- **对照**：目标引擎本体（通过进程内钩子获取中间张量与最终音频）；
- **指标**：
  - 最大绝对误差 MaxAE、皮尔逊相关系数（音频对齐后）；
  - 整句音素 ID 一致率（前端）；
  - 实时率 RTF = 合成耗时 / 音频时长。

### 5.2 模型层一致性

当输入音素序列与引擎一致时：

| 语言 | MaxAE | 相关系数 |
|---|---|---|
| 中文 | $1\times10^{-6}$ | 1.0000000 |
| 英文 | $1\times10^{-6}$ | 1.0000000 |
| 法/德/西/葡 | $1\times10^{-6}$ | 1.0000000 |
| 日文 | $1\times10^{-6}$ | 1.0000000 |
| 韩文 | $1.3\times10^{-4}$ | 1.0000000 |

误差来自浮点累加顺序差异，听感无差别。该结果同时验证了 §4.2 的算子语义、§4.3 的缓存/量化策略是正确的。

### 5.3 前端一致性

中文自然句（Windows 系统文本资源抽取，599 句全新留出，比较首句完整音素 ID）：

| 方案 | 整句音素 ID 一致率 |
|---|---|
| 早期 SDK（规则 + 标定词表 + 哈希分词） | 82.5% |
| 复刻官方 `CBlstmWordBreaker`（未调后处理） | 89.6% |
| + 「词起始类别」切分 + 官方词表相邻词合并 | 91.0% |
| + 引擎语料学习的「合并率」表 | 93.8% |
| + 未见词默认合并率 0.5 | 94.7% |
| + 句尾非句末标点规则 | 95.3% |
| + 「一」变调规则（词首/例外前字） | 95.7% |
| + 多音字 CRF 离线复刻（应/为/将/中/只/地等 80 个模型） | **97.2%（582/599）** |

补充验证：

- **断词网络**与引擎逐字节一致（325/325 字符类别，见 §4.4.4）；
- **词界单独评测**（纯汉字留出集 3,000 句）：98.27% 整句词界一致；
- 剩余 26 句差异：读音 4（OOV 多音字）、词界（罕见词/乱码行）、标签类——
  其中读音差异来自引擎的 27 个多音字 CRF（§4.5），尚未移植；
- **《生僻字》歌词 10 行**：6 行与引擎整句音素完全一致（对应音频逐采样一致）；
- **符号句**（`-5℃～25℃`、`￥3,280.50`、`99.9%`、`2026/10/26`、`9:30`）：9 项符号规则中 7 项与引擎逐 ID 一致。

### 5.3.1 多语种前端实测（每语种 17~19 句自然句，整句 id 一致）

| 语言 | 一致 | 说明 |
|---|---|---|
| 中文 | **19/19** | 另有 599 句大评测 97.2% |
| 英文 | **17/17** | 冠词弱读、OOV 词最大 onset 音节切分、词表标定 |
| 法语 | **17/17** | 连诵（含 dix→z 替换型）、省音、词表标定 |
| 德语 | **17/17** | 词内 `-` 切分、词表标定 |
| 西语 | **17/17** | 词表标定 |
| 葡语 | **17/17** | 词表标定 + 连字符复合词 |
| 日语 | **17/17** | 断词模型精确复刻 + 读音/重音词表 + 上下文读音 + 数词切分 |
| 韩语 | **17/17** | 整句词读音采集（词表）|

合计 **138 句全部逐 id 一致（138/138 = 100%）**；
另外用引擎实测语料留出集（seg_B，3000 句 UI/系统文本）做大规规模中文泛化验证：**2855/3000 = 95.2%**；
逐项分析显示神经断词逐字类别已精确复刻，残差集中在「词表后处理合并」与「多音字上下文读音」两类长尾。
Python 与 Node 双端逐 id 一致（`tools/check_node_parity.py`，17/17，含 5 句 CRF 多音字决策与 3 句词界合并回归）。Node SDK 也完整复刻了 A 段 Trie 遍历与 B 段权重打分（`node/lib/crf_zh.js`），不依赖 Python 运行时。
评测工具：`tools/eval_ml_offline.py`（golden 为引擎实测 encoder 输入 id，离线可复现）；中文评测：
`tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl`。

### 5.3.2 英文大规模标定（自然语料 500 句 + TN 压力集 102 句）

138 句小样本之外，我们为英文单独建立了更大规模的引擎探针标注：
从公版英文小说抽取 500 句自然句（`data/eval/en_engine_500.jsonl`），另构造 102 句 TN（数字/货币/日期/
时间/单位/分数/缩写）压力集（`data/eval/en_engine_tn.jsonl`），全部用引擎 fast 探针抓取 encoder 输入 id。
利用这批数据，直接从引擎句内输出标定出 1402 个高频词的**句内读音（含弱读形态）**，再用隔离探针补全
2038 个词的单字读音，配合标点/括号/连字符复合词/罗马数字/部分 TN 规则重写英文前端。
结果：英文自然句逐 id 一致率从 **17.8% 提升到 99.6%（498/500）**，8 语种 golden 仍保持 138/138；
TN 压力集从 5.9% 提升到 **100%（102/102）**（覆盖货币、日期、时间、单位、分数、罗马数字、电话分组、
缩写点号、文件名/URL、引号 br2、groupnum 等规则）。另外用三本不同作者的书构造 500 句 holdout
（未见词表标定前 34%，补 1700 个引擎标定词、并实现连字符复合词/省略号/括号句末等标点规则后达
**499/500 = 99.8%**）。再用第三个独立 holdout2（Tale of Two Cities / Wizard of Oz / War of the Worlds，
500 句）验证：零词表标定时仅 **266/500 = 53.2%**，按本文流程补 490 个引擎标定词后达到
**492/500 = 98.4%**，说明词表覆盖 + 标点/异读规则是英文泛化的主要瓶颈（词表已扩到 10,618 词 + 缩写/大小写复合词专项词表）。异读词（present/record/address 等 40 词）按上下文触发词选择
动词/名词读音，专设 80 句异读评测 **80/80**；Node SDK 英文前端已同步移植，在 1682 句英文语料上与 Python 逐 id 一致。
评测工具：`tools/eval_en_corpus.py`，标定工具：`tools/capture_ml_corpus.py`、`tools/harvest_en_lexicon_engine.py`。

### 5.4 架构与公开工作的对应

| 逆向观察 | 对应公开工作 |
|---|---|
| 时长预测 + 长度调节（encoder 内 `CumSum/Range/If`） | FastSpeech / FastSpeech 2 的 length regulator 与 variance adaptor |
| 端侧小模型 + 动态 INT8（DQ-LSTM/MatMulInteger） | LightSpeech 的端侧轻量路线；ONNX Runtime 动态量化 |
| 流式低延迟推理（因果卷积 + 缓存 + 分块） | FastSpeech 2s 的 text-to-waveform / 流式思路 |
| 流式声码器（空洞卷积 + 转置卷积 + GRU 状态） | WaveNet / Parallel WaveGAN 一脉的流式变体 |

需要强调：生产引擎为流式做了不同取舍（帧级循环解码器而非并行 Transformer 解码器），因此**不能简单等同于某篇论文**；论文提供的是结构空间与设计动机，逆向提供的是实际实现。

### 5.5 性能

| 指标 | 数值 |
|---|---|
| 实时率 RTF | 0.113（约 8.8 倍实时） |
| 常驻内存 | ≈375 MB |
| 模型总体积 | ≈5 MB（encoder 2.1 MB、decoder 0.7 MB、vocoder 2.3 MB） |

### 5.6 复现要点核查表

为便于他人复现，列出必须精确复刻的细节：

1. 容器 XOR 密钥与 40 字节头；
2. 8 个流式算子的三种块模式（首块/中间/flush）与缓存更新顺序；
3. decoder 分块调度（64/32/flush，短句首块 32）；
4. 逐块量化（`Round` + 缩放）；
5. vocoder 输入布局（mel 转置 ×0.25）；
6. 符号名到 ID 的按包映射；
7. 前端音素格式（`-bos- br0 … br4 punc。 ~`）与 br1/br3 的插入规则。

---

## 5.7 可复现验证矩阵

仓库提供一键脚本 `tools/verify_all.py`，依次复现全部证据：

| 检查 | 命令 | 结果 |
|---|---|---|
| SDK 端到端（106 项，含 8 语种引擎实测前端 golden） | `tests/test_all.py` | **106/106** |
| 中文 599 句首句逐 id | `tools/eval_first_sentence.py data/eval/zh_eval_599.jsonl` | **582/599 = 97.2%** |
| Node≡Python 逐 id（含 CRF/词界回归） | `check_node_parity.py` | **17/17** |
| 八语种 138 句逐 id（离线 golden） | `tools/eval_ml_offline.py` | **138/138 = 100%** |

## 6 讨论与局限

1. **中文多音字读音模型尚未移植**。引擎的多音字读音由 CRF 模型决定
   （语音包 `.dat` 内 87 个、系统 `MSTTSLocZhCN.dat` 内 27 个，标签/模板已解析，
   如 应 = yīng/yìng、卷 = juǎn/juàn/quán）。我们通过进程内注入 hook 定位到
   `CVocabTrie::GetID` 并成功捕获上千次查找，但在 SAPI 合成路径下
   embedded.tts 的 CRF 运行时代码零调用，读音判定发生在其他组件（正在追查）。
   当前用「官方词表读音 + 语料统计上下文修正」逼近，中文首句一致率 95.7%，
   残余 14/599 句主要来自这里（其中约 11 句为长尾词界拆分差异，3 句为语料乱码行）。
2. **传统/边界情形**。极少数长尾词界（引擎合并/切分策略）与乱码行仍不同；
   词界单独评测为 98.27%。
3. **多语言混排**。引擎具备英文音素与 `mixlingual.dat` 支持；当前实现覆盖
   单语言整句，中英混排句式的边界情形未完全对齐。
4. **语速语义**。引擎通过 SAPI `rate` 在模型外做时域伸缩，而模型内 `speed_ratios`
   恒为 1；SDK 提供的是模型级语速缩放，二者语义尚不严格等价。
5. **数据依赖**。复现需要用户自备语音包（appx）；本文不分发任何微软语音数据。
6. **评估范围**。一致性结论基于 23 个音色包与 8 种语言、每语种 17~19 句的
   扩展评测集；更广泛语料下长尾词表仍需扩充。

---

## 7 结论

本文完成了对闭源端侧神经语音引擎的黑盒重建：从加密容器还原标准 INT8 ONNX 计算图，等价实现 8 个流式算子与分块解码/量化规则，使模型层输出与官方引擎逐采样一致；并通过"引擎作为标注器"的方法重建文本前端的数据与读音词表。在文本前端部分，我们进一步定位并逐字节复刻了引擎内部的 LSTM 断词模型（中文 325/325、日语 54/54 字符类别一致），并复刻其 Trie 后处理的合并规则，使中文自然句首句整句音素一致率达到 **97.2%（582/599）**（词界单独评测 98.27%）。

最终系统为纯本地、无 Windows 依赖的 Python/Node SDK，在 8 种语言、23 个音色包上验证了保真度与性能。
文本前端扩展评测（每语种 17~19 句，共 138 句）**全部与引擎逐 id 一致**，
Python 与 Node 双端输出一致；端到端测试 106 项（含 24 条引擎实测前端 golden）全通过；
一键验证脚本见 `tools/verify_all.py`。残余差异主要是中文多音字读音
（80/87 个模型已离线复刻并启用，599 句 97.2%）与极少数长尾词界。

---

## 参考文献

[1] Ren Y, Ruan Y, Tan X, Qin T, Zhao S, Zhao Z, Liu T-Y. FastSpeech: Fast, Robust and Controllable Text to Speech. NeurIPS, 2019.

[2] Ren Y, Hu C, Tan X, Qin T, Zhao S, Zhao Z, Liu T-Y. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. ICLR, 2021. arXiv:2006.04558. <https://arxiv.org/abs/2006.04558>（含 FastSpeech 2s：text-to-waveform）

[3] Luo R, Tan X, Wang R, Qin T, Li J, Zhao S, Chen E, Liu T-Y. LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search. ICASSP, 2021. arXiv:2102.04040. <https://arxiv.org/abs/2102.04040>

[4] Tan X, et al. NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality. arXiv:2205.04421. <https://arxiv.org/abs/2205.04421>

[5] Tan X. Pushing the Frontier of Neural Text to Speech. ISCSLP 2021 Tutorial, Microsoft Research Asia. <https://www.microsoft.com/en-us/research/uploads/prod/2021/07/ISCSLP2021-TTS-Tutorial-Xu-Tan.pdf>

[6] Tan X. Efficient ML for Speech and Music. Microsoft Research Asia. <https://www.microsoft.com/en-us/research/wp-content/uploads/2021/07/Efficient-ML-for-Speech-and-Music-Xu-Tan.pdf>

[7] ONNX Runtime: Cross-platform Machine Learning Model Accelerator. <https://onnxruntime.ai>

[8] Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR, 2018.

[9] Vaswani A, et al. Attention Is All You Need. NeurIPS, 2017.

[10] Yamamoto R, Song E, Kim J-M. Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-Resolution Spectrogram. ICASSP, 2020.

[11] van den Oord A, et al. WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499, 2016.

[12] 吾爱破解论坛. 逆向工程与 ONNX 模型分析相关技术讨论. <https://www.52pojie.cn>
