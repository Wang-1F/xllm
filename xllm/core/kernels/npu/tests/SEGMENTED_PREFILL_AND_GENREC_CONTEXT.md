# NPU Attention 单测上下文说明

本文档与 `segmented_prefill_attention_test.cpp` 配套，供后续开发/审阅快速恢复上下文。**路径**：`xllm/core/kernels/npu/tests/`。

---

## 1. 文件与构建

| 项 | 说明 |
|----|------|
| 源码 | `segmented_prefill_attention_test.cpp` |
| 开关 | `USE_NPU` 为 ON 时才会编译该测试 |
| 链接库 | `torch`、`ascendcl`（`aclrt*`）、`nnopbase`（`aclCreateTensor` 等）、`opapi`（`aclnnFusedInferAttentionScoreV3*`） |
| 运行器 | `GTest::gtest_main`，日志 `glog` |

算子 API 头文件：`aclnnop/aclnn_fused_infer_attention_score_v3.h`（当前环境为 **V3**，非 V4）。

---

## 2. 设备与 Stream

- 测试类中 `kDeviceId` 必须与 `torch_npu::init_npu("npu:<id>")` **一致**，否则可能出现设备错配崩溃。
- 计时使用 `std::chrono::steady_clock` + 首尾 `aclrtSynchronizeStream`（wall-clock，非 ACL Event）。

---

## 3. 融合 Attention 算子（aclnn 两阶段）

- **阶段 1**：`aclnnFusedInferAttentionScoreV3GetWorkspaceSize`（规划，多在 Host CPU）
- **阶段 2**：`aclnnFusedInferAttentionScoreV3(workspace, ...)`（异步提交到 `aclrtStream`）

布局字符串：`"BNSD"`，张量形状为 `[B, N, S, D]`（本测试中常取 `B=1`）。

### 3.1 `attenMask`（BOOL）

- **语义**：`True` = 该位置被屏蔽（不参与注意力）。
- **因果**：`triu(1)` 上三角为 `True`（只看过去）；建议在 **CPU** 上构造 BOOL 再 `.to(device)`（NPU 上 BOOL 的 `triu` 可能不稳定）。

### 3.2 全注意力 / 空 mask

- 部分 CANN 版本在 **交叉注意力**（`q` 的 S ≠ `k/v` 的 S）下 **`attenMask=nullptr` 会报错（如 561002）**。
- 做法：传入形状 `[1, 1, Q_S, KV_S]` 的 **全 `False`** mask（`CreateFullAttentionMask`），语义等价于「不屏蔽」。

### 3.3 `softmaxLse` 与 `softmaxLseFlag`

- 当 `softmaxLseFlag=true` 时，输出 LSE 的 **shape 必须为 `[B, N, Q_S, 1]`**（FP32），见官方文档；**不能**用 `[B, N, Q_S]` 三维，否则会 `GetWorkspaceSize` 失败。

### 3.4 `actualSeqLengths` / `actualSeqLengthsKv`

- 本测试中多数场景为 **张量 shape 即有效长度**，传 `nullptr` 即可。
- 若使用「大 tensor + 有效长度更短」的生产布局，需配合 `aclIntArray` 传入有效长度（与 demo 对齐）。

### 3.5 `aclTensor` 与 `aclTensorList`

- `aclCreateTensor` 本质是 **描述符**（指向已有设备指针 + shape/stride），开销很小；可对 **slice 后的连续视图** 建描述符（数据不拷贝）。
- **不要**对同一批 tensor 再调 `aclDestroyTensorList` 若官方 demo 不显式销毁 list（曾出现 **double-free**）；按当前代码：`DestroyAcl` 只 `Destroy` 各 `aclTensor`，不销毁 list。

---

## 4. 多段序列（Segmented Prefill）语义

用于：一条长序列被切成多段，每段独立 **形状正确的** `q/k/v/out`（BNSD），逐段调用融合算子。

- **流水线优化**：
  - 每段 **预创建** `AclSegmentDesc`（`aclTensor` 一次建好）；
  - **Workspace 按所有段 `GetWorkspaceSize` 的最大值只分配一次**，同一条 stream 上 kernel 顺序执行，可 **复用** 同一块 workspace；
  - 热路径：`GetWorkspaceSize`（CPU）与上一段 `Execute`（NPU 异步）可重叠。
- **正确性**：`CorrectnessBasic`、`CorrectnessVsReference`（与 CPU FP32 单段 self-attn 参考对比）。

---

## 5. 生成式推荐（GenRec）组合流程

业务：单条用户序列按时间/类型分为四段（沿序列轴拼接）：

`[ history | context | real_time | target ]`

长度由 `SeqPartLengths { h, c, r, t }` 描述，总长度 `S = h+c+r+t`。

### 5.1 各段注意力语义（self-attention 视角）

| 段 | 含义（直观） |
|----|----------------|
| **history** | 仅 history 内部 **因果**（看过去） |
| **context** | 可看 **全部 history + context**（段内 full） |
| **real_time** | 可看 **全部 history + context**；在 real_time 内部 **因果** |
| **target** | 可看 **全部 history + context + real_time**；在 target 内部 **只看自己**（对角） |

### 5.2 实现策略：5 次 launch + 2 次合并（Host 侧）

用 **同一条** `q_seq/k_seq/v_seq`（形状 `[S, N 或 Nkv, D]`）切片得到各步输入，**全部为 BNSD**：

| Step | 名称 | Q 范围 | KV 范围 | Mask | LSE |
|------|------|--------|---------|------|-----|
| 1 | `hist_causal` | history | history | 因果 | 否 |
| 2a | `ctx_rt_full` | context+real_time | history+context | 全 False（等价 full） | 是 `[1,N,Sq,1]` |
| 2b | `rt_causal` | real_time | real_time | 因果 | 是 |
| 3a | `tgt_full` | target | history+context+real_time | 全 False | 是 |
| 3b | `tgt_diag` | target | target | 对角（仅自位置） | 是 |

**合并**（两段 disjoint key 上的归一化输出用 LSE 做数值稳定加权）：

- **real_time 最终**：`CombineFlashOutputs( step2a 的 real_time 切片输出与 LSE, step2b 的输出与 LSE )`
- **target 最终**：`CombineFlashOutputs( step3a, step3b )`

公式（与代码一致）：  
`max_l = max(lse1, lse2)`，`ei = exp(lsei - max_l)`，`out = (o1*e1 + o2*e2) / (e1+e2)`。

### 5.3 Batch 维（`seq_lens`）

- 多请求时可将多条序列在 **S 维拼接** 为 `[total_S, N, D]`，用长度前缀数组 `seq_lens`（长度 `batch_size+1`，单调递增）对每条序列切 `[seq_lens[b], seq_lens[b+1])` 再跑上述 5+2 流程。基准 `BenchGenRecAttention` 即为此模式。

### 5.4 CPU 精度金标准（`ReferenceGenRecGold`）

- 在 **CPU FP32** 上按「最终每个 query 可见的 key 集合」**直接**算 attention，**不**依赖两段 LSE 合并公式，用于与 NPU 流水线对比。
- `GenRecCorrectnessVsReference`：小形状 + GQA 配置下对比 history、context 切片、合并后 real_time/target 的 `max_abs`。

### 5.5 GenRec V2（另一种拆法，`BuildGenRecV2Calls`）

在 **history 仍只做因果自注意力** 的前提下，将后面三段改为：

| Step | Q | KV | sparseMode | 说明 |
|------|---|---|------------|------|
| 1 | history | history | **2（leftUpCausal）** | 因果自注意力，压缩 2048×2048 mask |
| 2a | context + real_time + **target** | history + context | 0 | full（全 False mask） |
| 2b | real_time + **target** | **仅 real_time** | **2（leftUpCausal）** | Sq=r+t > Skv=r：前 r 行自然因果，后 t 行全可见（j≤i 在 i≥r 时恒成立）。共享压缩 2048×2048 mask，kernel 走优化路径 |
| 3 | target | target | 0 | 对角（仅自己） |

- **real_time 合并**：2a 中 real_time 对应行（Q 子序列中 index `c .. c+r-1`）与 2b 中前 `r` 行做 `CombineFlashOutputs`。
- **target 合并（三路）**：2a 中 target 行、2b 中 target 行、step3 对角自注意，用 `CombineFlashOutputsThree`（与两次 `CombineFlashOutputs` 链式等价）。

与 **5.2 五段拆法** 的最终可见 key 集合一致时，gold 仍用 **`ReferenceGenRecGold`**；单测：`GenRecV2CorrectnessVsReference`（小形状，CPU gold 可接受耗时）。

**性能单测**：`BenchGenRecV2Attention` 使用与 GenRec 相同的 `parts{1300,8,400,800}`、batch=3，路径与 `BenchmarkSegmentedPrefill` 一致：先对所有 `AttnCallDesc` 做 `CreateAcl`，`ComputeMaxWorkspaceGenRecV2` 取最大 workspace，**单次 `aclrtMalloc`**，热循环内 `RunGenRecV2OneBatch`（4× launch + combine），最后 `BenchmarkGenRecV2` 内同步并 `aclrtFree`。封装函数：`BenchmarkGenRecV2`、`RunGenRecV2OneBatch`。

**sparseMode 优化说明**：

- **sparseMode=2（leftUpCausal）**：以左上角为起点的下三角 mask。CANN 要求传入 **压缩 2048×2048** 下三角 mask（`CreateCompressedCausalMask2048`），kernel 内部按 block 级别跳过上三角区域，效率远高于 sparseMode=0/1 逐元素检查。
- **关键洞察**：step 2b 中 Sq = r+t > Skv = r，leftUpCausal 中 row `i` 可见 column `j ≤ i`；当 `i ≥ r`（target 行）时 `j ≤ i` 恒成立（因为 `j < r ≤ i`），因此 target 行自动获得 full attention，无需显式 full mask。
- Step 1（history）和 Step 2b **共享同一个** `causal2048` tensor（避免重复分配）。
- `AttnCallDesc` 内含 `sparse_mode`/`pre_tokens`/`next_tokens` 字段，`RunGenRecV2OneBatch` 和 `ComputeMaxWorkspaceGenRecV2` 自动透传。

---

## 6. 主要 GTest 用例（便于 filter）

| 用例 | 作用 |
|------|------|
| `CorrectnessBasic` | 多段有限值/非零 |
| `CorrectnessVsReference` | 单段与 CPU FP32 参考 `max_diff` |
| `Bench*` | 分段 benchmark（流水线 workspace） |
| `BenchGenRecAttention` | GenRec 多 batch 耗时 |
| `GenRecCorrectnessBasic` | GenRec 健康检查 |
| `GenRecCorrectnessVsReference` | GenRec 与 `ReferenceGenRecGold` 对齐 |
| `GenRecV2CorrectnessVsReference` | GenRec V2（4 launch + 合并）与 gold 对齐（小形状） |
| `BenchGenRecV2Attention` | GenRec V2 长序列 benchmark（workspace 复用，`1300/8/400/800`） |

示例（仅跑 GenRec 精度）：

```bash
./segmented_prefill_attention_test --gtest_filter='*GenRecCorrectnessVsReference'
```

GenRec V2：

```bash
./segmented_prefill_attention_test --gtest_filter='*GenRecV2CorrectnessVsReference'
```

GenRec V2 长序列性能：

```bash
./segmented_prefill_attention_test --gtest_filter='SegmentedPrefillAttentionTest.BenchGenRecV2Attention'
```

---

## 7. 已知坑位（维护时对照）

1. `kDeviceId` 与 `init_npu` 不一致 → 崩溃/错设备。
2. `softmaxLse` 维数错误 → `GetWorkspaceSize` 返回非 0（如 561002）。
3. 交叉注意力下 `attenMask=nullptr` → 可能 561002；改用全 `False` mask。
4. `aclDestroyTensorList` + 再 `Destroy` 内部 tensor → double-free。
5. BOOL mask 尽量 CPU 构造再拷到 NPU。

---

## 8. 变更本单测时建议

- 改 mask 语义或分段边界 → **同步** `ReferenceGenRecGold` 与 `BuildGenRecCalls`。
- 升级 CANN / 换 `FusedInferAttentionScore` 版本 → 重新核对头文件与文档中的 **LSE shape、Optional 参数**。

---

*文档与 `segmented_prefill_attention_test.cpp` 保持同步更新为佳。*
