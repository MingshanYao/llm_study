// 报告索引元数据：由 paper-research 工作流维护，docs/index.html 通过 <script> 加载
//（不用 fetch + reports.json，是为了让 index.html 在 file:// 下双击也能打开）
var REPORTS = [
  {
    "slug": "deepseek-v4/deepseek-v4-architecture",
    "title": "DeepSeek-V4 总体架构解读：CSA/HCA 混合注意力 + mHC + Muon 的百万上下文 MoE 架构",
    "date": "2026-09-15",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/deepseek-v4-attention",
    "title": "DeepSeek-V4 Attention 设计解读：压缩稀疏注意力（CSA）与重压缩注意力（HCA）",
    "date": "2026-09-14",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/csa-compression",
    "title": "CSA 压缩过程详解：DeepSeek-V4 压缩稀疏注意力的序列维压缩",
    "date": "2026-09-14",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/deepseek-v4-kvcache",
    "title": "DeepSeek-V4 KV Cache 组织解读：异构条目、双缓存布局与盘上存储",
    "date": "2026-09-14",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/deepseek-v4-lightning-indexer",
    "title": "DeepSeek-V4 Lightning Indexer 深度解读：稀疏注意力的可学习索引器",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/deepseek-mhc",
    "title": "DeepSeek mHC 解读：流形约束超连接（Manifold-Constrained Hyper-Connections）",
    "date": "2026-09-14",
    "paper": "mHC: Manifold-Constrained Hyper-Connections (paper/mhc.pdf)",
    "arxiv": "https://arxiv.org/abs/2512.24880",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/deepseek-moe",
    "title": "DeepSeek MoE 进化史：细粒度专家、无损均衡与 MegaMoE",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/dspark-speculative-decoding",
    "title": "DSpark：置信度调度的半自回归投机解码（DeepSeek 生产级 spec decoding）",
    "date": "2026-10-08",
    "paper": "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation（本地 paper/dspark.pdf）",
    "arxiv": "https://arxiv.org/abs/2607.05147",
    "repo": "https://github.com/deepseek-ai/DeepSpec",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v4/v4pro-ridge-point",
    "title": "逐算子看 DeepSeek-V4-Pro 的算术强度：prefill 与 decode 的 roofline 账本",
    "date": "2026-10-08",
    "paper": "DeepSeek-V4 (arXiv:2606.19348) + 官方 config.json + DSpark (arXiv:2607.05147) 生产口径",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "deepseek-v4"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t0-overview",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 0 讲：总览——为长程 Agent 而生的 KV cache 压缩之作",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t1-ced",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 1 讲：CED 因果编码器-解码器架构",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t2-csa2",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 2 讲：CSA2 注意力——跨层 KV 复用与层级稀疏索引",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t3-extensions",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 3 讲：架构扩展篇——Single-Pass mHC、Engram、DSpark 与 FP4 Main KV Cache",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t4-infra",
    "title": "DeepSeek-V4.1 零基础七讲·第 4 讲：基础设施——KV cache 的存储层级、持久缓存管理与 SWA Bounded Replay",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t5-pretraining",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 5 讲：预训练——45T token 多模态语料是怎么炼成的",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "deepseek-v41/deepseek-v41-t6-posttraining",
    "title": "DeepSeek-V4.1 零基础七讲 · 第 6 讲：后训练——数据、环境与异步基础设施",
    "date": "2026-09-20",
    "paper": "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 paper/deepseek_v41.pdf）",
    "group": "deepseek-v41"
  },
  {
    "slug": "moe/moe-t0-foundation",
    "title": "MoE 零基础五讲 · 第 0 讲：前置知识（FFN/SwiGLU、参数vs激活、softmax/sigmoid/softplus、top-k）",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4 (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "group": "moe"
  },
  {
    "slug": "moe/moe-t1-basics",
    "title": "MoE 零基础五讲 · 第 1 讲：MoE 基本原理（专家/门控/top-k、负载不均、辅助损失两难）",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4 (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "group": "moe"
  },
  {
    "slug": "moe/moe-t2-v1v3",
    "title": "MoE 零基础五讲 · 第 2 讲：DeepSeek V1→V3 演进细讲（细粒度专家、共享隔离、bias 门控）",
    "date": "2026-09-16",
    "paper": "DeepSeekMoE (arXiv:2401.06066) · DeepSeek-V3 (arXiv:2412.19437)",
    "arxiv": "https://arxiv.org/abs/2412.19437",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "moe"
  },
  {
    "slug": "moe/moe-t3-v4",
    "title": "MoE 零基础五讲 · 第 3 讲：DeepSeek V4/V4.1 细讲（√Softplus、MegaMoE、FP4、Hash 路由）",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4 (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "moe"
  },
  {
    "slug": "moe/moe-t4-code",
    "title": "MoE 零基础五讲 · 第 4 讲：DeepSeek MoE 代码逐行走读（Gate/Expert/MoE + 手动模拟 forward）",
    "date": "2026-09-16",
    "paper": "DeepSeek-V4 (paper/deepseek_v4.pdf)",
    "arxiv": "https://arxiv.org/abs/2606.19348",
    "repo": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference",
    "group": "moe"
  },
  {
    "slug": "quantization/quant-t0-foundation",
    "title": "大模型量化零基础七讲 · 第 0 讲：全景图（显存账本、浮点数速成、有损压缩直觉与术语表）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t1-basics",
    "title": "大模型量化零基础七讲 · 第 1 讲：量化数学基础（scale/zero-point、对称非对称、粒度谱系、误差与校准）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t2-ptq",
    "title": "大模型量化零基础七讲 · 第 2 讲：训练后量化——GPTQ 与 AWQ（输出误差目标、逐列误差补偿、激活感知缩放）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t3-activation",
    "title": "大模型量化零基础七讲 · 第 3 讲：激活量化与 FP8（outlier 四对策：LLM.int8、SmoothQuant、QuaRot/SpinQuant、FP8）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t4-ultra-low-bit",
    "title": "大模型量化零基础七讲 · 第 4 讲：极低比特——2-bit 与 1.58-bit（矢量量化、AQLM、QuIP#、BitNet、MXFP4/NVFP4）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t5-kv-systems",
    "title": "大模型量化零基础七讲 · 第 5 讲：KV Cache 量化与推理系统落地（KIVI、attention sink、Marlin kernel、QServe、GGUF 选型）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "quantization/quant-t6-qat-roadmap",
    "title": "大模型量化零基础七讲 · 第 6 讲：QAT、量化 scaling law 与成长路线图（STE、LSQ、4-bit 定理、四阶段专家路线）",
    "date": "2026-09-20",
    "group": "quantization"
  },
  {
    "slug": "rl/rl-t0-foundation",
    "title": "RL 训练工程师零基础七讲 · 第 0 讲：全景图（后训练流程、RL 循环、术语表与系列地图）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t1-policy-gradient",
    "title": "RL 训练工程师零基础七讲 · 第 1 讲：策略梯度从零讲（softmax 策略、REINFORCE、baseline 减方差）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t2-llm-as-rl",
    "title": "RL 训练工程师零基础七讲 · 第 2 讲：把 LLM 接上 RL（token 即动作、KL 约束、温度采样）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t3-ppo",
    "title": "RL 训练工程师零基础七讲 · 第 3 讲：PPO 精讲（重要性采样、clip、GAE、四模型架构）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t4-rlhf-dpo",
    "title": "RL 训练工程师零基础七讲 · 第 4 讲：RLHF 与 DPO（奖励模型、Bradley-Terry、隐式奖励推导）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t5-rlvr-grpo",
    "title": "RL 训练工程师零基础七讲 · 第 5 讲：RLVR 与 GRPO（可验证奖励、组内基线、训练病症与药方）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t6-systems",
    "title": "RL 训练工程师零基础七讲 · 第 6 讲：系统与实战（rollout 引擎、显存账本、异步训练、调试手册）",
    "date": "2026-09-16",
    "group": "rl"
  },
  {
    "slug": "rl/rl-t7-score-centering",
    "title": "RL 训练工程师零基础七讲 · 番外篇：Score Centering 精读（TIM 漂移、一行修复、与重要性采样对比）",
    "date": "2026-09-20",
    "paper": "Score Centering Stabilizes Off-policy Reinforcement Learning (arXiv:2609.20807)",
    "arxiv": "https://arxiv.org/abs/2609.20807",
    "repo": "https://github.com/martin-marek/score-centering",
    "group": "rl"
  },
  {
    "slug": "vllm/vllm-t0-overview",
    "title": "vLLM 源码解析九讲 · 第 0 讲：整体架构总览——分层地图与一次请求的生命周期",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t1-engine",
    "title": "vLLM 源码解析九讲 · 第 1 讲：Engine 与进程模型——AsyncLLM、EngineCore 与 ZMQ 消息总线",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t2-scheduler",
    "title": "vLLM 源码解析九讲 · 第 2 讲：Scheduler——token budget、混批、抢占与 async scheduling",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t3-kvcache",
    "title": "vLLM 源码解析九讲 · 第 3 讲：KV Cache 与 PagedAttention——BlockPool、prefix caching 与 offload",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t4-modelrunner",
    "title": "vLLM 源码解析九讲 · 第 4 讲：GPUModelRunner——调度输出如何变成一次 GPU 执行",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t5-distributed",
    "title": "vLLM 源码解析九讲 · 第 5 讲：分布式与并行——TP/PP/EP/DP 的 rank 世界",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t6-spec-structured",
    "title": "vLLM 源码解析九讲 · 第 6 讲：投机解码与结构化输出——draft-verify 与 grammar bitmask",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t7-models-quant",
    "title": "vLLM 源码解析九讲 · 第 7 讲：模型加载与量化——weight_loader 回调与 quant_method 多态",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t8-multimodal-lora",
    "title": "vLLM 源码解析九讲 · 第 8 讲：多模态与 LoRA——mm_hash 指纹、encoder cache 与 punica kernel",
    "date": "2026-10-08",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "vllm/vllm-t9-ops",
    "title": "vLLM 源码解析十讲 · 第 9 讲：算子是怎么被支持的——先懂 PyTorch，再懂 vLLM（以 DeepSeek-V4 为实例）",
    "date": "2026-10-16",
    "paper": "vLLM 官方仓库源码（main@fa75d34，2026-10-16 抓取）",
    "repo": "https://github.com/vllm-project/vllm",
    "group": "vllm"
  },
  {
    "slug": "misc/llm-arch-evolution",
    "title": "LLM 架构演进技术报告：FFN 与 Attention 的降算/降存主线（短板迁移视角的 2022–2026 全景）",
    "date": "2026-09-20",
    "group": "misc"
  },
  {
    "slug": "misc/verilog-crash-course",
    "title": "Verilog 速成：半天写出能仿真的 RTL",
    "date": "2026-09-16",
    "group": "misc"
  },
  {
    "slug": "misc/mha-prefill-decode",
    "title": "MHA 的 Prefill 与 Decode 账本：并行维度、计算量、访存量——以及 MHA→GQA→MLA 的改善主线",
    "date": "2026-10-16",
    "group": "misc"
  },
  {
    "slug": "misc/attention-ai-calculator",
    "title": "Attention AI 计算器：一条统一公式算尽 MHA→GQA→MQA→MLA→稀疏→量化→投机解码（可调参数）",
    "date": "2026-10-19",
    "group": "misc"
  },
  {
    "slug": "misc/mla-training-inference",
    "title": "MLA 训练与推理全解：低秩压缩、解耦 RoPE、weight absorption 与 FlashMLA kernel",
    "date": "2026-10-19",
    "paper": "DeepSeek-V2 (arXiv:2405.04434, paper/deepseek_v2.pdf) · DeepSeek-V3 (arXiv:2412.19437, paper/deepseek_v3.pdf)",
    "arxiv": "https://arxiv.org/abs/2405.04434",
    "repo": "https://github.com/deepseek-ai/FlashMLA（main@0f11e17 / sm90 快照 ba89a34）· vLLM（main@fa75d34）· SGLang（main@5b683d6b72f8）",
    "group": "misc"
  },
  {
    "slug": "misc/flash-attention-v1-to-v4",
    "title": "FlashAttention v1→v4 演进全解：从 online softmax 到 Blackwell 非对称流水线（算法过程详解）",
    "date": "2026-10-19",
    "paper": "FlashAttention v1-v4（arXiv:2205.14135 / 2307.08691 / 2407.08608 / 2603.05451，本地 paper/flash-attention-v1~v4.pdf）",
    "arxiv": "https://arxiv.org/abs/2603.05451",
    "repo": "https://github.com/Dao-AILab/flash-attention（main@e7a6402，2026-10-19 抓取）",
    "group": "misc"
  },
  {
    "slug": "misc/triton-cutlass-cute-dsl",
    "title": "GPU Kernel 三条开发路线对比：Triton JIT、CUTLASS C++ 模板与 CuTe DSL",
    "date": "2026-10-20",
    "paper": "Triton v3.6–3.8 release notes · CUTLASS 4.8.1 · NVIDIA CuTe DSL 官方博客（2025-11）",
    "repo": "https://github.com/NVIDIA/cutlass · https://github.com/triton-lang/triton（2026-10-20 抓取）",
    "group": "misc"
  },
  {
    "slug": "misc/deepinfer",
    "title": "DeepInfer 框架解读：DeepSeek 生产级推理引擎的 C++ 调度器、ZOS 异步流水线与确定性推理",
    "date": "2026-10-21",
    "paper": "DeepInfer 官方仓库与部署 Q&A（doc/inference_deployment_qa.md）· 模型：DeepSeek-V4.1-Flash（paper/deepseek_v41.pdf）",
    "repo": "https://github.com/deepseek-ai/deepinfer（main@774ce24，2026-10-21 抓取）",
    "group": "misc"
  },
  {
    "slug": "misc/flash-dwc-tensor-core",
    "title": "Flash-DWC 解读：Depthwise 卷积如何被 Tensor Core 加速（GEMV→Toeplitz GEMM 的 56.25% 密度映射，附 benchmark 独立复核）",
    "date": "2026-10-26",
    "paper": "Flash-DWC: Making Depthwise Convolution Compute-Efficient on GPUs（ACM TACO 2026, 23(4), DOI 10.1145/3837088；正文被 Cloudflare 拦截未获取，取材为 Crossref 完整摘要 + 官方源码全文）",
    "repo": "https://github.com/GilgameshXYZ123/Flash-DWC（main，2026-10-26 抓取）",
    "group": "misc"
  },
  {
    "slug": "vla/vla-t0-overview",
    "title": "具身智能与 VLA 零基础七讲 · 第 0 讲：全景图——具身智能是什么，VLA 为什么突然爆发（术语表与系列地图）",
    "date": "2026-10-21",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t1-robot-learning",
    "title": "具身智能与 VLA 零基础七讲 · 第 1 讲：机器人学习基础——动作空间、模仿与强化学习、ACT 动作分块与数据金字塔",
    "date": "2026-10-21",
    "paper": "ACT/ALOHA (arXiv:2304.13705) · Open X-Embodiment (arXiv:2310.08864) · UMI (arXiv:2402.10329)",
    "arxiv": "https://arxiv.org/abs/2304.13705",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t2-rt1-rt2",
    "title": "具身智能与 VLA 零基础七讲 · 第 2 讲：VLA 的诞生——RT-1 与 RT-2：动作变成 token、VLM 直接输出动作与语义泛化",
    "date": "2026-10-21",
    "paper": "RT-1 (arXiv:2212.06817) · RT-2 (arXiv:2307.15818) · SayCan (arXiv:2204.01691) · PaLM-E (arXiv:2303.03378)",
    "arxiv": "https://arxiv.org/abs/2307.15818",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t3-openvla",
    "title": "具身智能与 VLA 零基础七讲 · 第 3 讲：OpenVLA 与开源生态——7B 开放 VLA 架构解剖、动作离散化词表、LoRA 微调与 OFT 变体",
    "date": "2026-10-21",
    "paper": "OpenVLA (arXiv:2406.09246) · OpenVLA-OFT (arXiv:2502.19645) · Octo (arXiv:2405.12213)",
    "arxiv": "https://arxiv.org/abs/2406.09246",
    "repo": "https://github.com/openvla/openvla",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t4-diffusion-pi0",
    "title": "具身智能与 VLA 零基础七讲 · 第 4 讲：连续动作头——Diffusion Policy 与 π0：扩散/流匹配取代自回归 token，50Hz 灵巧操作",
    "date": "2026-10-21",
    "paper": "Diffusion Policy (arXiv:2303.04137) · π0 (arXiv:2410.24164) · π0.5 (arXiv:2504.16054) · FAST (arXiv:2501.09747) · Flow Matching (arXiv:2210.02747)",
    "arxiv": "https://arxiv.org/abs/2410.24164",
    "repo": "https://github.com/Physical-Intelligence/openpi",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t5-frontier",
    "title": "具身智能与 VLA 零基础七讲 · 第 5 讲：2025–2026 前沿全景——Helix、GR00T N1、Gemini Robotics：双系统架构、世界模型与数据引擎",
    "date": "2026-10-21",
    "paper": "Helix (Figure 博客) · GR00T N1 (arXiv:2503.14734) · Gemini Robotics (arXiv:2503.20020) · GO-1 (arXiv:2503.06669) · RDT-1B (arXiv:2410.07864) · WALL-OSS (arXiv:2505.21864)",
    "arxiv": "https://arxiv.org/abs/2503.14734",
    "repo": "https://github.com/NVIDIA/Isaac-GR00T",
    "group": "vla"
  },
  {
    "slug": "vla/vla-t6-roadmap",
    "title": "具身智能与 VLA 零基础七讲 · 第 6 讲：实战与成长路线图——LeRobot、仿真基准复现、四个动手项目与四阶段专家路线",
    "date": "2026-10-21",
    "paper": "LeRobot · LIBERO (arXiv:2306.03310) · SimplerEnv (arXiv:2405.05941) · ManiSkill3 (arXiv:2410.00425) · RoboCasa (arXiv:2406.02523)",
    "repo": "https://github.com/huggingface/lerobot",
    "group": "vla"
  }
];

// 导航层次：主题域(TOPICS) → 系列/专题(collections) → 报告(REPORTS)，由 docs/index.html 渲染，
// 每个主题域的推荐阅读路线在 docs/hub/<id>.html。
// collection 两种写法：{ group } 引用 REPORTS 里整组（按数组顺序），{ slugs } 显式列出跨组/单篇集合。
// 新增报告时：先在 REPORTS 追加条目，再把 slug 挂进对应 collection；没挂的会落进 index 的「其他」。
var TOPICS = [
  {
    "id": "foundations",
    "name": "基础入门",
    "desc": "零基础系列，不要求读过论文，从直觉、公式到代码逐讲递进，建议作为全站入口",
    "hub": "hub/foundations.html",
    "collections": [
      { "name": "MoE 零基础五讲", "group": "moe" },
      { "name": "大模型量化零基础七讲", "group": "quantization" }
    ]
  },
  {
    "id": "architecture",
    "name": "架构专题",
    "desc": "Attention / MoE / 长上下文的模型设计深读，以 DeepSeek V4 与 V4.1 为主线",
    "hub": "hub/architecture.html",
    "collections": [
      { "name": "LLM 架构演进综述", "slugs": ["misc/llm-arch-evolution"] },
      { "name": "MLA 训练与推理深读", "slugs": ["misc/mla-training-inference"] },
      { "name": "DeepSeek-V4.1 零基础七讲", "group": "deepseek-v41" },
      {
        "name": "DeepSeek-V4 专题深读",
        "note": "本系列的 DSpark 投机解码与 V4-Pro roofline 账本两篇归入「推理与系统」主题",
        "slugs": [
          "deepseek-v4/deepseek-v4-architecture",
          "deepseek-v4/deepseek-v4-attention",
          "deepseek-v4/csa-compression",
          "deepseek-v4/deepseek-v4-lightning-indexer",
          "deepseek-v4/deepseek-v4-kvcache",
          "deepseek-v4/deepseek-mhc",
          "deepseek-v4/deepseek-moe"
        ]
      }
    ]
  },
  {
    "id": "inference",
    "name": "推理与系统",
    "desc": "推理引擎源码走读、prefill/decode 成本账本，以及可以动手调参的计算器",
    "hub": "hub/inference.html",
    "collections": [
      { "name": "vLLM 源码解析十讲", "group": "vllm" },
      {
        "name": "成本账本与计算器",
        "slugs": [
          "misc/mha-prefill-decode",
          "misc/attention-ai-calculator",
          "deepseek-v4/v4pro-ridge-point"
        ]
      },
      { "name": "投机解码", "slugs": ["deepseek-v4/dspark-speculative-decoding"] },
      { "name": "DeepInfer 框架解读", "slugs": ["misc/deepinfer"] }
    ]
  },
  {
    "id": "training",
    "name": "训练",
    "desc": "后训练与 RL 工程化的零基础主线",
    "hub": "hub/training.html",
    "collections": [
      { "name": "RL 训练工程师零基础七讲（含番外）", "group": "rl" }
    ]
  },
  {
    "id": "hardware",
    "name": "硬件与底层",
    "desc": "面向硬件的 kernel 算法与 RTL 入门",
    "hub": "hub/hardware.html",
    "collections": [
      { "name": "FlashAttention v1→v4 演进", "slugs": ["misc/flash-attention-v1-to-v4"] },
      { "name": "Flash-DWC：Depthwise 卷积上 Tensor Core", "slugs": ["misc/flash-dwc-tensor-core"] },
      { "name": "Kernel 开发路线对比", "slugs": ["misc/triton-cutlass-cute-dsl"] },
      { "name": "Verilog 速成", "slugs": ["misc/verilog-crash-course"] }
    ]
  },
  {
    "id": "embodied",
    "name": "具身智能与 VLA",
    "desc": "面向零基础的具身智能主线：从机器人学习基础到 RT-1/RT-2、OpenVLA、π0 与 2025–2026 前沿，落到实战与成长路线图",
    "hub": "hub/embodied.html",
    "collections": [
      { "name": "具身智能与 VLA 零基础七讲", "group": "vla" }
    ]
  }
];
