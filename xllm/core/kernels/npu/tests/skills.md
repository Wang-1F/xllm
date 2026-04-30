# MTGR Single-Side Skill

这个文件约定了以后在 `xllm/core/kernels/npu/tests` 下继续尝试新的 MTGR 组图方式时，默认应该怎么写代码、怎么做精度验证、怎么做性能统计。

## 适用场景

当需要尝试新的 MTGR attention 组图方式时，后续只需要描述这些信息：

- 适用场景：`no_matched` 或 `partial_rt_matched`
- 每一段的 `Q / KV`
- 每一段的模式：`full`、`sparse_mode=2/3/4`、`target_diag_bmm`
- 最终哪些分支需要 `attention_update`

不需要再重复说明如何做 warmup、如何打印 `GetWorkspace`、kernel、gap、host/device 总耗时，这些默认按本文件执行。

## 默认落盘方式

### no_matched

- 复用 `mtgr_attention_one_vs_multi_stage_perf_test.cpp` 的写法和统计格式。
- 如果是替换当前最优方案，直接在该文件中新增或替换单个实现函数与测试用例。

### partial_rt_matched

- 当前仓内不再长期保留 `partial_rt_matched` 的单侧实验文件。
- 如果要尝试新的 partial_rt 组合方式，按本 skill 临时新增一个单侧实验文件即可：
  - 先写候选实现
  - 先做单侧精度与分段性能验证
  - 方案确认后再同步进 `mtgr_attention.cpp`
  - 实验结束后只保留最终主线，旧候选删除

### 端到端回归

- 如果新的最佳方案已经确认并同步进项目实现，需要同时更新
  `mtgr_attention_e2e_precision_test.cpp`
- 该文件负责真实调用 `MTGRAttentionImpl::forward` 做端到端精度回归。

## 默认精度规则

- 基线始终使用 one-stage irregular-mask FIA。
- 精度测试前默认 warmup。
- 输出至少包含：
  - `max_abs`
  - `mean_abs`
  - `allclose`
- 如果是 matched 场景：
  - matched prefix cache 预填充不计入运行阶段
  - 输出只对比 unmatched 部分

## 默认性能规则

- 默认 `warmup=5`
- 默认 `repeat=20`
- 若用户明确要求快速验证，可降到 `repeat=5`
- `malloc` 和大块 workspace 预申请默认放在计时外
- 若使用 shared workspace，默认不把 workspace 申请时间记入 kernel 统计

## 默认性能打印

每次新增组图方式，性能输出默认包含以下内容：

### 总耗时

- `avg_total_dev_ms`
- `avg_total_wall_ms`
- `host_overhead_ms = wall - dev`

### 执行顺序

- `kernel_launch` 顺序
- `getworkspace` 顺序

### 分段耗时

- 每一段 `avg_dev_ms`
- 每一段 `avg_getws_ms`

### 段间信息

- kernel-to-kernel `device_gap`
- kernel-to-kernel `host_interval`

如果存在 overlap 设计，必须从打印中能直接看出 gap 是否被消除。

## 默认实现要求

- 优先复用现有 helper，不重复造一套 ACL 封装
- `target_diag` 默认优先使用解析 `BMM` 写法
- matched 场景优先复用已有 cache 预填充逻辑
- 新方案先写单侧测试，再考虑同步进 `mtgr_attention.cpp`

## 当前保留的主线文件

- `mtgr_attention_e2e_precision_test.cpp`
  - 真实项目实现端到端回归
- `mtgr_attention_one_vs_multi_stage_perf_test.cpp`
  - `no_matched` 当前最优单侧
  - `one_stage` 对照扫参 CSV 输出

## 当前默认目标

- `no_matched` 主线：以 `mtgr_attention_one_vs_multi_stage_perf_test.cpp` 为准
- `partial_rt_matched` 主线：以 `mtgr_attention.cpp` 实现 + `mtgr_attention_e2e_precision_test.cpp` 端到端回归为准
- 任何新方案都必须至少给出：
  - 精度对比
  - 分段 `GetWorkspace`
  - 分段 kernel 耗时
  - gap / host interval
  - 总耗时
