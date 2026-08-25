[English](README.md) | 中文

# dual-radeon-vllm

**在两张消费级 Radeon(RX 7900 XT,gfx1100,ROCm 7.14)上把 vLLM 张量并行真正跑通的端到端记录——包括拦住大多数人的那个 RCCL 崩溃:根因、修复和 30 行的复现器。**

`gemma-4-31B`(w4a16)在 2× RX 7900 XT 上解码 **43 tok/s**,两张卡同时 265 W;26B MoE 短上下文 **108 tok/s**。测试机是 VFIO 虚拟机、无 P2P、跨 die PCIe 3.0——故意选的最不利拓扑,在这里能跑通的,裸机只会更好。

> 本页是浓缩的中文导览:判定、修复、性能速览。**全部数字与细节以英文文档为准**;命令、报错、配置一律保留英文原样——你搜到的和要跑的就是它们。

## 我是不是中招了?

两张以上 AMD 卡,vLLM / PyTorch DDP / 任何走 RCCL 的东西一启动就死,报错长这样:

```
RuntimeError: NCCL error: unhandled cuda error
HIP failure 'the operation cannot be performed in the present state'
amdgpu 0000:0b:00.0: amdgpu: PCIE atomic ops is not supported
```

六十秒判定,不需要 RCCL、PyTorch 或 vLLM:

```bash
hipcc --offload-arch=gfx1100 -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3
```

plain 内核通过而 hostcall 内核被拒(`REFUSED`),就是这个问题。机制一句话:PCIe AtomicOps 到不了 GPU,ROCr 就建不起 hostcall 缓冲,于是任何声明了 hostcall 的内核都被拒绝派发,而 RCCL ≥ 2.27.7-b43 的设备内核恰好全都声明了它。完整证据链与排除过的 12 个假设:[docs/root-cause.md](docs/root-cause.md)。

## 修复:两条路

| 你的环境 | 修复 | 代价 |
|---|---|---|
| **裸机**,卡在芯片组转接的槽位 | 重建一个不含 hostcall 的 RCCL:[build/build-rccl-nohostcall.sh](build/build-rccl-nohostcall.sh),或直接用 [Releases](../../releases) 里带校验和的成品 | 约 85 分钟,或下载 |
| **虚拟机**(Proxmox/QEMU 直通) | 多数情况只改一行:`hostpci0: 0000:0b:00` 改成 `hostpci0: 0000:0b:00.0`,即单函数直通 | 一次重启 |

主流板子的第二条全长槽常挂在芯片组下,所以裸机双卡很容易落进第一行;虚拟机的一行修复原理与 A/B 验证见 [docs/vfio-atomics.md](docs/vfio-atomics.md),裸机全流程见 [docs/deploy-vllm.md](docs/deploy-vllm.md),逐步排查见 [docs/diagnosis.md](docs/diagnosis.md)。

注意版本:重建请用 RCCL **2.27.7**(`release/rocm-rel-7.1.1.1` 分支)。2.30.4 的问题出在设备链接步骤,`NDEBUG` 治不了,已在硬件上验证失败——见英文 README 的警告框。

## 跑起来之后:性能速览

五种架构 × 11 档上下文长度,解码 tok/s(TP=2,两轮均值,2026-07-25 campaign):

| 模型 | 500 | 32K | |
|---|---:|---:|---|
| gemma-4-26B-A4B(int4 MoE) | **107.8** | 72.8 | 最快;需一次约 26 分钟的编译,之后有缓存 |
| Qwen3-8B(BF16) | 79.6 | 61.4 | 双卡对单卡 **1.70×**,BF16 买到的是速度 |
| gemma-4-12B(w4a16) | 59.9 | 41.9 | 双卡只有 1.19×,第二张卡买到的是并发容量 |
| gemma-4-31B(w4a16) | 43.2 | 29.5 | 主力模型,两卡同步 265 W |
| Qwen3.6-27B(hybrid SSM) | 12.1 | 4.2 | 随上下文线性劣化,原生 vLLM 下长上下文避开 |

![解码吞吐与上下文长度](docs/assets/decode-vs-context.svg)

完整分析——含打补丁后的第二次 campaign、滑窗模型 37 tok/s 跑平 32K 的曲线、prefill 峰值拟合——见 [docs/benchmarks.md](docs/benchmarks.md);架构差异为何这么大见 [docs/architecture-notes.md](docs/architecture-notes.md)。

## 还要知道的三件事

- **权重加载可能慢到 2 MiB/s。** Ubuntu HWE 内核 `7.0.0-28` 的回归,升级 `7.0.0-30` 即除去主害;残余的可写映射惩罚用 [vllm#49991](https://github.com/vllm-project/vllm/pull/49991) 的 clone flag 绕开。细节在 [docs/open-questions.md](docs/open-questions.md) §8。
- **FP8、AITER、调优过的 MoE 配置在 gfx1100 上都不可用**;hybrid-SSM 与滑动窗口解码的下游补丁在 [patches/](patches/)。完整的"什么不行"清单见英文 README。
- **双卡贴槽安装时上卡吸下卡的排风**,持续负载结温可到 99 °C;卡间对着缝隙加一把 120 mm 风扇能压回 90 °C,是整台机器最便宜的一处改进。

## 目录

```
diagnose/    从这里开始:零依赖探针(hipgate3.cpp 最关键)
build/       重建 RCCL,并独立验证产物
deploy/      注入 ROCm/vLLM 容器的三件套
benchmarks/  全部原始数据与分析脚本,无 GPU 也能复算
patches/     campaign 用到的 vLLM 下游补丁
docs/        根因、基准、修复、开放问题(英文)
```

---

本页对应英文版 2026-08-25 的状态(commit `4ba455f`);两者不一致时,以英文版为准。MIT 许可;与 AMD 无任何关联。仓库不含 RCCL 源码,分发编译产物的 BSD-3 义务见 [NOTICE.md](NOTICE.md)。
