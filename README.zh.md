[English](README.md) | 中文

# dual-radeon-vllm

**两张消费级 Radeon(RX 7900 XT,gfx1100,ROCm 7.14)跑通 vLLM 张量并行的完整工程记录——拦住大多数人的那个 RCCL 崩溃,这里有根因、修复,和一个 30 行的复现程序。**

`gemma-4-31B`(w4a16)在 2× RX 7900 XT 上解码 **43 tok/s**,两张卡同时压满 265 W;26B 的 MoE 短上下文能到 **108 tok/s**。而且测试机还是台 VFIO 虚拟机:无 P2P、跨 die PCIe 3.0——拓扑故意挑了最差的。这里都能跑通;带 P2P 的裸机没有在这里测过,这些数字是最差拓扑下的底线,不是硬件的上限。

此后同一条阶梯又在另外十一种机器配置上跑过,租的和借的都有,全部拿来对照这一对卡;本页每一个数字发布前都由闸门从已入库的原始行重算。

### 测过的机器

| 机器 | 卡数 | 谁的 | 上下文阶梯 | 起 |
|---|---|---|---|---|
| **RX 7900 XT**(gfx1100,ROCm 7.14)—— 这里一切的主角 | 2 张,以及 1 张 | 自己的 | 500 – 32 000,2026-09-03 起到 **128 000** | 2026-07-25 |
| A100 SXM4 80G | 1 | Colab | 500 – 32 000 | 2026-08-29 |
| L4 24G · T4 16G | 1 | Colab | 500 – 32 000,L4 到 128 000 | 2026-08-30 |
| A100 SXM4 40G | 1 | Colab | 只测一件事:推得的带宽数字值多少 | 2026-09-02 |
| H100 80G | 1、2、4 | Modal 租的 | 500 – 128 000 | 2026-09-03 |
| H200 143G · B300 275G | 1 | Modal 租的 | 500 – 128 000 | 2026-09-03 |
| RTX PRO 6000 96G | 1、2 | Modal 租的 | 500 – 128 000 | 2026-09-03 |

十三种机器配置、八个 checkpoint、56 个结果文件里 5 624 条请求级测量、两份跨机器投影里 2 247 个 chart-grade 格子、八组双卡/四卡上 880 个 all-reduce 格、12 篇中英对照的长文——这些计数和下面每个数字一样,都由 [`verify_doc_figures.py`](benchmarks/analyze/verify_doc_figures.py) 从文件重算。

### 主要发现

一行一条，数字在前，链接后面是正文；每个数字发布前都由 `verify_doc_figures.py` 从已入库的行重算。

- **第二张卡值多少，由内存控制器决定，不由互联决定**——`mem_busy` 在五种互不共享硬件的设定里都排对了顺序，从第二张 Radeon 到第二张 H100（[`cuda-modal/`](benchmarks/cuda-modal/README.md)）。
- **集合通信在七组双卡/四卡上跨 62 倍，而推理一点没用到**：batch 1 解码落在延迟端，那一端只跨 3.2 倍（[`allreduce-2026-09-03/`](benchmarks/allreduce-2026-09-03/)）。
- **四张租来的卡自己选了三种注意力后端**，没人传过参数，所以每个跨机器比值都带一项后端差——从每份 serve 日志里读出来的，不是假定的（[`cuda-modal/`](benchmarks/cuda-modal/README.md)）。
- **128 000 token 处让曲线变平的是有界注意力窗口，不是循环状态**：H100 上 Muse-Glimmer 掉 4.8 %，混合 SSM 的 27B 掉 21.8 %，跟稠密 31B 一样深（[`cuda-modal/`](benchmarks/cuda-modal/README.md)）。
- **这对卡自己也到了 128 000**——六个模型里四个在两张 20 GB 卡上跑完十六档到 128 000，gemma 各臂到头只剩 500 token 时的一半左右（12B −52.5 %），有界窗口的 Muse-Glimmer 只掉 17.3 %，遥测能说清哪一个是算力、哪一个是内存（[`campaign-2026-09-03/`](benchmarks/campaign-2026-09-03/README.md)）。
- **分页解码内核里改十一行，两个滑窗模型在 32 K 各值 2.75 倍和 3.15 倍**，因为原版循环把整段序列读一遍再把窗口外的掩掉（[为什么](docs/sliding-window-block-skip.md)）。
- **投机解码在 32 K 慢 3.4 倍，是路径选择的问题**：每步两个 query token 让 Triton 注意力从分段的 3D 路径掉到串行的 2D 路径，放回 3D（vllm#45450）后这对卡在 32 K 从 8.81 回到 32.57 tok/s，两家厂商的卡上都验证过（[为什么](docs/speculative-decoding-on-rdna.md)）。
- **贪心解码的不确定性出在 W4A16 内核的 split-K 收尾**：把 vllm#54706 的固定顺序归约编进容器自己那个 vLLM 提交后，32 次贪心生成 32 次一致，而同一构建不打补丁在四个格子里有两个会变，后端固定不动（[A/B](benchmarks/gfx1100-w4a16-54706/README.md)）。
- **第二张 Radeon 在 BF16 上值 1.70 倍，w4a16 上 1.19 倍**，而让它能跑起来的那个 RCCL 修复在下一节。

> 这是一页浓缩的中文导览,只讲三件事:怎么确诊、怎么修、能跑多快。**所有数字与细节以英文文档为准**;报错、命令、配置一律保留英文原样,因为你要搜的、要跑的就是它们。

## 我是不是中招了?

两张以上 AMD 卡,vLLM、PyTorch DDP,或任何走 RCCL 的程序一启动就崩,报错长这样:

```
RuntimeError: NCCL error: unhandled cuda error
HIP failure 'the operation cannot be performed in the present state'
amdgpu 0000:0b:00.0: amdgpu: PCIE atomic ops is not supported
```

一条命令就能确诊,不需要装 RCCL、PyTorch 或 vLLM:

```bash
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3
```

plain 内核能跑、hostcall 内核显示 `REFUSED`,就是这个问题。原因一句话说完:PCIe AtomicOps 到不了 GPU,ROCr 就建不起 hostcall 缓冲区,凡是声明了 hostcall 的内核都会被拒绝派发,而 RCCL 从 2.27.7-b43 起的设备内核恰好全都声明了它。完整证据链、以及被逐一排除的 12 个假设,见 [docs/root-cause.md](docs/root-cause.md)。

## 怎么修:两条路

| 你的环境 | 修法 | 代价 |
|---|---|---|
| **裸机**,卡插在走芯片组的槽位 | 重建一个不带 hostcall 的 RCCL:[build/build-rccl-nohostcall.sh](build/build-rccl-nohostcall.sh);不想编译就用 [Releases](../../releases) 里带校验和的成品 | 约 85 分钟,或直接下载 |
| **虚拟机**(Proxmox/QEMU 直通) | 多数情况改一行就够:`hostpci0: 0000:0b:00` 改成 `hostpci0: 0000:0b:00.0`,也就是只直通 GPU 这一个功能 | 重启一次 |

消费级主板的第二条显卡槽往往走芯片组,裸机装双卡很容易正好踩进第一行。虚拟机那一行修复的原理与 A/B 验证在 [docs/vfio-atomics.md](docs/vfio-atomics.md),裸机完整流程在 [docs/deploy-vllm.md](docs/deploy-vllm.md),想逐步排查看 [docs/diagnosis.md](docs/diagnosis.md)。

版本要盯紧:重建请用 RCCL **2.27.7**(`release/rocm-rel-7.1.1.1` 分支)。2.30.4 的问题出在设备链接那一步,`NDEBUG` 治不了——已经在硬件上验证过会失败,细节见英文 README 的警告框。

## 能跑多快

五种架构 × 11 档上下文,解码 tok/s(TP=2,两轮均值,2026-07-25,原生 vLLM)。这是 2026-07-25 那一场的基线,保留作历史记录;之后的数据在十三种机器配置上、上下文到 128 000,入口是 [`benchmarks/cuda-modal/`](benchmarks/cuda-modal/README.md) 和两份跨机器投影 `prefill.jsonl` / `decode.jsonl`:

| 模型 | 500 | 32K | 一句话 |
|---|---:|---:|---|
| gemma-4-26B-A4B(int4 MoE) | **107.8** | 72.8 | 全场最快;代价是首次 engine 启动约 26 分钟,热启动成本本轮未测 |
| Qwen3-8B(BF16) | 79.6 | 61.4 | 双卡是单卡的 **1.70 倍**,BF16 的第二张卡买到的是速度 |
| gemma-4-12B(w4a16) | 59.9 | 41.9 | 双卡只快 1.19 倍,第二张卡实际买到的是并发容量 |
| gemma-4-31B(w4a16) | 43.2 | 29.5 | 干活主力,两张卡同步 265 W |
| Qwen3.6-27B(hybrid SSM) | 12.1 | 4.2 | 解码随上下文线性下滑,原生 vLLM 跑长上下文要避开 |

虚线是单卡,实线是双卡:蓝色 BF16 一路拉开,绿色 4-bit 几乎贴在一起——第二张卡到底值多少,取决于模型吃不吃带宽:

![单卡对双卡,2026-08-24](docs/assets/tp1-vs-tp2-2026-08-24.svg)

### 这台机器目前能到多少

每个模型一条线,取的是它被测到的最好配置;线型说明代价:实线装上发行版 vLLM 就有,虚线需要一个还没合并的补丁,具体是哪个写在图下面。数据来自 [`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl),每个点都带着自己的日期、vLLM、ROCm 和补丁清单。

![这台机器的最佳解码吞吐](docs/assets/decode-vs-context-best.svg)

滑窗模型 Muse-Glimmer-30B 从自己的窗口位置起一路跑平,32K 仍有 **37.4 tok/s**。图上的虚线要的补丁上游都还没合并,复现脚本在 [patches/](patches/)。

hybrid SSM 的崩塌单独一张图。同一个模型、同一台机器,只差一个补丁:发行版 vLLM 在 32K 上每 token 要 **261.9 ms** 且随上下文直线上升,加上 vllm#45916 是 **27.7 ms** 且是平的 —— 9.5 倍,斜率从 7.41 降到 0.26 ms/千 token。两臂各跑两轮且顺序反转,路由从 TP worker 内部记录;8K 那格四次运行分成两个模态,图注写明了画的是高的那个。

![hybrid SSM 的崩塌与修复](docs/assets/hybrid-ssm-collapse.svg)

按 campaign 分的旧图(每张图里所有模型都钉在同一个软件栈上)仍在 [docs/benchmarks.md](docs/benchmarks.md)。

滑窗跳块是本仓库自己的 11 行改动,收益曲线的形状本身就是机制证明:窗口以内 1.00×(没有块可跳),出了窗口单调上升,到 32K 是 2.75× 到 3.15×:

![滑窗跳块的收益曲线](docs/assets/sliding-window-block-skip.svg)

完整分析——prefill 峰值拟合、KV 容量与并发、控制组、以及为什么架构比参数量重要——见 [docs/benchmarks.md](docs/benchmarks.md) 与 [docs/architecture-notes.md](docs/architecture-notes.md)。

## 还有三件事值得知道

- **权重加载慢得离谱(最差 2 MiB/s)?** 那是 Ubuntu HWE 内核 `7.0.0-28` 的回归,升到 `7.0.0-30` 就去掉了大头;剩下的可写映射惩罚,用 [vllm#49991](https://github.com/vllm-project/vllm/pull/49991) 的 clone flag 绕开。来龙去脉在 [docs/open-questions.md](docs/open-questions.md) §8。
- **FP8、AITER、调优过的 MoE 配置,在 gfx1100 上都没有。** 完整的"什么不行"清单在英文 README,买硬件之前先看它。
- **双卡贴着装,上面那张会吸下面那张的排风**:持续负载结温能到 99 °C。对着卡缝加一把 120 mm 风扇能压回 90 °C,是整台机器最便宜的一处改进。

## 目录怎么走

```
diagnose/    从这里开始:零依赖探针,hipgate3.cpp 一条命令确诊
build/       重建 RCCL,并独立验证产物
deploy/      往 ROCm/vLLM 容器里注入的三件套
benchmarks/  全部原始数据和分析脚本,没有 GPU 也能复算每个数字
patches/     第二次测量用到的 vLLM 下游补丁
docs/        根因、基准、修复、开放问题(英文)
```

---

两边不一致时,以英文版为准。MIT 许可;与 AMD 无任何关联。仓库不含 RCCL 源码;若分发编译产物,BSD-3 义务见 [NOTICE.md](NOTICE.md)。
