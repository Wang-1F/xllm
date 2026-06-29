# MTGR 实验运行手册（当前重构快照）

本文档面向“没有任何上下文的新窗口/新同学”，目标是让对方能直接在当前重构后的 xLLM 快照上跑通：

1. MTGR 端到端服务实验
2. MTGR attention 算子/单测/harness 实验
3. 之前已经跑过的历史实验到底是怎么跑的

这份文档刻意把两类东西分开写：

- **当前快照可直接复现**：拿这份 repo 就能跑。
- **历史实验参考**：当时确实这么跑过，但依赖的是旧 worktree、旧二进制或旧策略代码；当前快照不一定直接复现得出来。

## 1. 当前工作快照

- 当前推荐使用的干净快照路径：
  - `/tmp/xllm_feat_seg_snapshot_20260626`
- 当前文档路径：
  - `/tmp/xllm_feat_seg_snapshot_20260626/MTGR_EXPERIMENT_RUNBOOK_20260629.md`

本次我已经把之前实验中常用、但原本不在这份快照里的工具脚本补进 repo 了：

- `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_qps_sweep.py`
- `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_prefix_cache_report.py`
- `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_qps_sweep_oracle.py`
- `/tmp/xllm_feat_seg_snapshot_20260626/tools/rewrite_mtgr_segment4_fixed.py`

其中：

- `mtgr_qps_sweep.py`：通用 MTGR QPS 客户端。
- `mtgr_prefix_cache_report.py`：从服务日志汇总 prefix-cache 命中/驱逐等指标。
- `mtgr_qps_sweep_oracle.py`：历史 oracle admission/附加 tensor 的 wrapper，现已改成 repo 内自引用，不再依赖旧绝对路径。
- `rewrite_mtgr_segment4_fixed.py`：把四段式数据集的第 4 段长度固定成给定值，现已改成 argparse 版本，可直接复用。

## 2. 当前快照到底支持什么，不支持什么

### 2.1 当前快照里**直接存在**的 MTGR KV-cache policy

代码入口：

- `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/common/global_flags.cpp`
- `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/common/rec_model_utils.h`

当前可直接用的 policy 名称：

- `full_cache` / `full_sequence`
- `prefix_only`
- `prefix_only_value_gated` / `value_gated`
- `dynamic_length_admission`
- `prefix_only_reuse_horizon` / `reuse_horizon` / `rha_writeback`

重要默认行为：

- `--mtgr_kv_cache_policy=auto`
  - 若 `--mtgr_attention_backend=flashinfer_token_mask`，会走 `full_cache`
  - 若 `--mtgr_attention_backend=hopper`，会走 `prefix_only`

也就是说，**为了实验可读性，建议显式写 policy，不要依赖 `auto`**。

### 2.2 当前快照里**不存在**的后续历史策略

以下这些名字，在这份干净快照里搜不到，对应实验属于“历史结果参考”，不是“当前快照可直接复现”：

- `prefix_only_watermark_cache`
- `len_value`
- `offline_oracle_admission`

所以，如果另一个窗口要复现实验，先分清楚：

- 想跑**当前重构后的干净项目**：请按本文档第 5、6 节走。
- 想复现**历史 len_value / oracle / pure-prefix-all-admit**：请看第 7 节，它们依赖的是旧 worktree/旧脚本/旧 server flag。

## 3. 环境与编译

### 3.1 容器边界

正常的代码检查、编译、起服务、跑 benchmark，都在容器内做。本文档默认你已经在目标容器中。

简单自检：

```bash
test -f /.dockerenv && echo IN_CONTAINER || echo NOT_IN_CONTAINER
```

### 3.2 进入快照目录

```bash
cd /tmp/xllm_feat_seg_snapshot_20260626
```

### 3.3 拉齐 submodule

```bash
git submodule update --init --recursive
```

### 3.4 编译

官方 quick start 在：

- `/tmp/xllm_feat_seg_snapshot_20260626/docs/zh/getting_started/quick_start.md`

最稳妥的编译方式：

```bash
python setup.py build
```

如果你明确只在 CUDA 环境里，也可以尝试：

```bash
python setup.py build --device cuda
```

### 3.5 定位二进制

服务端二进制：

```bash
export XLLM_BIN=$(find build -type f -path '*/xllm/xllm' | head -n 1)
echo "$XLLM_BIN"
```

MTGR attention harness 二进制：

```bash
export MTGR_HARNESS_BIN=$(find build -type f -name 'mtgr_attention_harness_e2e_test' | head -n 1)
echo "$MTGR_HARNESS_BIN"
```

### 3.6 `launch_xllm.py` 能不能用

可以，它在：

- `/tmp/xllm_feat_seg_snapshot_20260626/xllm/launch_xllm.py`

但它本质只是个很薄的 Python wrapper，最终还是调用编译出的 `xllm` 二进制。MTGR 这类实验我建议**直接调二进制**，更直白，也更接近之前的实验方式。

如果你坚持走 wrapper：

```bash
PYTHONPATH=$PWD python3 xllm/launch_xllm.py --help
```

## 4. 常用数据集与模型路径

### 4.1 主数据集

50k short-heavy access24：

- `/tmp/mtgr_50000_short_heavy_jsonl_20260602/requests_access24_growth_50000_short_heavy_seg4_time_ordered_seed20260602.jsonl`

5k access24：

- `/export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl`

5k amazon-history：

- `/export/home/zhangshen/datas/mtgr_5000_amazon_history_mapped_jsonl_20260608/requests_amazon_history_mapped_5000_user10x_short_heavy_seg4_time_ordered_seed20260608.jsonl`

5k 三段式 MTFM 数据集：

- `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/datasets/requests_access24_growth_5000_user10x_short_heavy_mtfm3seg_seed20260605.jsonl`

### 4.2 历史上常用的 xLLM 模型目录

1B / 60 层：

- `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_60layer_xllm`

0.6B / 36 层：

- `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_36layer_xllm`

1.5B / 90 层：

- `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_90layer_xllm`

旧 50k baseline 常用模型：

- `/tmp/xllm_full_cache_50000_hit_checkpoints_kv16_worker019e82b4_20260601/model_kv16`

## 5. 当前快照：如何跑端到端 MTGR 服务实验

这一节只写**当前快照直接支持**的实验。

### 5.1 推荐统一参数

历史上 MTGR xLLM 端到端实验比较常见的一组服务参数是：

```bash
--backend=rec
--enable_prefix_cache=true
--block_size=128
--max_memory_utilization=0.8
--max_tokens_per_batch=24384
--max_tokens_per_chunk_for_prefill=24384
--enable_chunked_prefill=false
```

`max_seqs_per_batch`：

- flashinfer 常见是 `8`
- 一些 hopper/60-layer 历史实验常用 `4`

### 5.2 先设置公共变量

```bash
cd /tmp/xllm_feat_seg_snapshot_20260626

export XLLM_BIN=$(find build -type f -path '*/xllm/xllm' | head -n 1)
export GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

export MODEL=/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_60layer_xllm
export DATASET=/export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl

export PORT=18080
export MASTER_PORT=19080
export TRANSFER_PORT=28080
export OUT_ROOT=/tmp/mtgr_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_ROOT"
```

### 5.3 通用服务启动模板

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=flashinfer_token_mask \
  --mtgr_kv_cache_policy=full_cache \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=8 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server.log" 2>&1 &

export SERVER_PID=$!
until rg -q "Brpc Server started" "$OUT_ROOT/server.log"; do sleep 1; done
```

停服务：

```bash
kill -INT "$SERVER_PID"
wait "$SERVER_PID"
```

### 5.4 通用客户端模板

```bash
python3 tools/mtgr_qps_sweep.py \
  --dataset "$DATASET" \
  --adapter xllm \
  --url "http://127.0.0.1:$PORT/v1/completions" \
  --model "$(basename "$MODEL")" \
  --out-dir "$OUT_ROOT/client_qps16" \
  --qps 16 \
  --warmup-s 0 \
  --measure-s 312.5 \
  --timeout-s 240 \
  --max-concurrency 0 \
  --max-tokens 1 \
  --dataset-limit 5000 \
  --order sequential \
  --write-details
```

说明：

- `--max-concurrency 0` 表示 open-loop，不在 client 侧限流。
- `--measure-s` 通常取 `dataset_limit / qps`。
  - 例如 5000 请求、QPS16，则 `5000 / 16 = 312.5`
  - 例如 50000 请求、QPS150，则 `50000 / 150 = 333.333...`

### 5.5 当前快照可直接跑的几个典型 policy

#### A. `flashinfer_token_mask + full_cache`

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=flashinfer_token_mask \
  --mtgr_kv_cache_policy=full_cache \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=8 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server_full_cache.log" 2>&1 &
```

#### B. `hopper + prefix_only`

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=hopper \
  --mtgr_kv_cache_policy=prefix_only \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=4 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server_hopper_prefix_only.log" 2>&1 &
```

#### C. `prefix_only_value_gated`

当前 value 函数是 `prefix_len`，阈值由 `--mtgr_kv_cache_prefix_len_threshold` 控制。

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=hopper \
  --mtgr_kv_cache_policy=prefix_only_value_gated \
  --mtgr_kv_cache_value_fn=prefix_len \
  --mtgr_kv_cache_prefix_len_threshold=2000 \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=4 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server_value_gated.log" 2>&1 &
```

#### D. `dynamic_length_admission`

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=hopper \
  --mtgr_kv_cache_policy=dynamic_length_admission \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=4 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server_dynamic_length.log" 2>&1 &
```

#### E. `prefix_only_reuse_horizon`

```bash
"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=hopper \
  --mtgr_kv_cache_policy=prefix_only_reuse_horizon \
  --mtgr_rha_pressure_free_block_ratio=0.10 \
  --mtgr_rha_min_reuse_count=2 \
  --mtgr_rha_cold_prefix_len_threshold=2000 \
  --mtgr_rha_entity_hot_skip_min_seen=0 \
  --mtgr_rha_state_max_entries=100000 \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=4 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  --mtgr_trace_log_level=1 \
  > "$OUT_ROOT/server_reuse_horizon.log" 2>&1 &
```

补充说明：

- 当前 repo 的服务端**支持**接收 `entity_id` tensor，相关代码在：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/distributed_runtime/rec_master.cpp`
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/framework/block/block_manager_pool.cpp`
- 但当前 bundled 的 `tools/mtgr_qps_sweep.py` **默认不会**把 `entity_id` 送进请求。
- 所以直接跑 `reuse_horizon` 时，如果客户端不改，它会退化为“prefix-family hash”而不是真正的用户 ID 版本。

### 5.6 如何把 prefix-cache 指标一起产出

最方便的方式是在 client 侧顺手加上：

```bash
python3 tools/mtgr_qps_sweep.py \
  --dataset "$DATASET" \
  --adapter xllm \
  --url "http://127.0.0.1:$PORT/v1/completions" \
  --model "$(basename "$MODEL")" \
  --out-dir "$OUT_ROOT/client_qps16" \
  --qps 16 \
  --warmup-s 0 \
  --measure-s 312.5 \
  --timeout-s 240 \
  --max-concurrency 0 \
  --max-tokens 1 \
  --dataset-limit 5000 \
  --order sequential \
  --write-details \
  --prefix-log "$OUT_ROOT/server.log" \
  --prefix-source xllm \
  --prefix-phase measure \
  --prefix-report-json "$OUT_ROOT/prefix_report.json" \
  --prefix-report-csv "$OUT_ROOT/prefix_report.csv"
```

或者跑完后单独分析：

```bash
python3 tools/mtgr_prefix_cache_report.py \
  --log "$OUT_ROOT/server.log" \
  --source xllm \
  --qps-summary "$OUT_ROOT/client_qps16/summary.json" \
  --phase measure \
  --block-size 128 \
  --out-json "$OUT_ROOT/prefix_report.json" \
  --out-csv "$OUT_ROOT/prefix_report.csv" \
  --print-table
```

## 6. 当前快照：如何跑 MTGR attention 算子实验

相关目录：

- harness 说明：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/mtgr_attention_harness/README.md`
- gtest 主体：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_attention_e2e_test.cpp`
- NVTX 汇总脚本：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/profiling/mtgr_attention_nvtx_report.py`
- FlexAttention Python baseline：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_flex_attention_baseline.py`

### 6.1 常用 gtest 入口

当前 harness 里比较关键的入口：

- `MTGRAttentionHarnessTest.Precision`
- `MTGRAttentionHarnessTest.PerfNvtxCsv`
- `MTGRAttentionHarnessTest.FixedShapeBatchPerfNvtxCsv`
- `MTGRAttentionHarnessTest.KVWritebackPerfNvtxCsv`
- `MTGRAttentionHarnessTest.DynamicSegmentRulesPrecision`
- `MTGRAttentionHarnessTest.DynamicWritebackThenAttentionPrecision`
- `MTGRAttentionHarnessTest.MixedBatchWritebackThenAttentionPrecision`

### 6.2 Precision 对比

```bash
export MTGR_HARNESS_BIN=$(find build -type f -name 'mtgr_attention_harness_e2e_test' | head -n 1)
export CUDA_VISIBLE_DEVICES=0

export XLLM_MTGR_HARNESS_PRECISION_PAIRS=50
export XLLM_MTGR_HARNESS_SEED=20260608
export XLLM_MTGR_HARNESS_MAX_ABS_MILLI=200

"$MTGR_HARNESS_BIN" \
  --gtest_filter=MTGRAttentionHarnessTest.Precision
```

如果只想测动态 segment 规则：

```bash
"$MTGR_HARNESS_BIN" \
  --gtest_filter=MTGRAttentionHarnessTest.DynamicSegmentRulesPrecision
```

### 6.3 固定 shape 的算子性能 sweep

```bash
export XLLM_MTGR_HARNESS_FIXED_LABELS=/tmp/mtgr_attention_fixed_shape_batch_labels.csv
export XLLM_MTGR_HARNESS_FIXED_TOTAL_LENS=2000,3000,4000,5000
export XLLM_MTGR_HARNESS_FIXED_BATCH_SIZES=1,4,8,16
export XLLM_MTGR_HARNESS_FIXED_TARGET=800
export XLLM_MTGR_HARNESS_FIXED_WARMUP=1
export XLLM_MTGR_HARNESS_FIXED_REPEAT=1
export XLLM_MTGR_HARNESS_FIXED_BACKENDS=full_flashinfer_base,block_sparse_flashinfer_base,torch_native_base,hopper_unified

"$MTGR_HARNESS_BIN" \
  --gtest_filter=MTGRAttentionHarnessTest.FixedShapeBatchPerfNvtxCsv
```

### 6.4 用 Nsight Systems 抓 NVTX，再转表

采集：

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -o /tmp/mtgr_fixed_shape_batch \
  "$MTGR_HARNESS_BIN" \
  --gtest_filter=MTGRAttentionHarnessTest.FixedShapeBatchPerfNvtxCsv
```

导出 sqlite：

```bash
nsys export \
  --type sqlite \
  --output /tmp/mtgr_fixed_shape_batch.sqlite \
  /tmp/mtgr_fixed_shape_batch.nsys-rep
```

用 helper 脚本聚合：

```bash
python3 xllm/core/kernels/cuda/tests/profiling/mtgr_attention_nvtx_report.py \
  --sqlite /tmp/mtgr_fixed_shape_batch.sqlite \
  --labels /tmp/mtgr_attention_fixed_shape_batch_labels.csv \
  --csv /tmp/mtgr_fixed_shape_batch_report.csv \
  --summary /tmp/mtgr_fixed_shape_batch_summary.json
```

### 6.5 FlexAttention baseline

最小 smoke：

```bash
python3 xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_flex_attention_baseline.py \
  --pairs 1 \
  --warmup 0 \
  --labels /tmp/mtgr_flex_attention_smoke_labels.csv
```

如果想复用 C++ harness 已经吐出来的 shape label：

```bash
python3 xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_flex_attention_baseline.py \
  --input-labels /tmp/mtgr_attention_fixed_shape_batch_labels.csv \
  --labels /tmp/mtgr_flex_attention_fixed_shape_batch_labels.csv
```

## 7. 历史实验：之前那些结果当时是怎么跑出来的

这一节的重点不是“当前快照直接复现”，而是“以后别人接盘时知道老结果从哪来的”。

### 7.1 50k pure-prefix all-admit @ QPS150

历史脚本：

- `/tmp/xllm_mtgr_new_short_heavy_qps150_pure_prefix_all_admit_worker019e864e_20260602/run_qps150_pure_prefix_all_admit.py`

对应语义：

- `prefix_only_watermark_cache`
- `W=0.50`
- `T_high=0`
- `T_low=0`
- 解释为：只写 `[history | context | realtime]`，不写 target，不跳请求

历史上常用的数据/模型：

- dataset:
  - `/tmp/mtgr_50000_short_heavy_jsonl_20260602/requests_access24_growth_50000_short_heavy_seg4_time_ordered_seed20260602.jsonl`
- model:
  - `/tmp/xllm_full_cache_50000_hit_checkpoints_kv16_worker019e82b4_20260601/model_kv16`

注意：`prefix_only_watermark_cache` 这条 server-side policy **不在当前干净快照里**。

### 7.2 oracle admission / len_value / capdiv10 历史实验

历史主脚本之一：

- `/tmp/mtgr_memutil_sweep_20260610/run_oracle_admission_mem02.py`

历史上常见的 server flag：

```text
--mtgr_offline_oracle_admission_enable=true/false
--mtgr_len_value_admission_enable=true/false
--mtgr_len_value_free_block_ratio=0.10
--mtgr_len_value_remaining_threshold=3.0
--mtgr_kv_block_capacity_divisor=10
```

这些 flag 对应的实现也**不在当前干净快照里**，所以：

- 当前快照可以保留这些历史脚本作参考
- 但不能指望直接在当前快照重新跑出完全一样的策略效果

历史结果根目录之一：

- `/tmp/mtgr_offline_oracle_admission_worker20260604`

里面有大量现成结果，可直接查：

- `bench/metrics_*.json`
- `logs/server_*.log`

### 7.3 tp99 < 1000ms 的 even-QPS 极限搜索

历史搜索脚本之一：

- `/tmp/mtgr_tp99_1000_even_limit_20260611/xllm_hlv_4seg_1b_access24/run_search.py`

相关结果根目录：

- `/tmp/mtgr_tp99_1000_even_limit_20260611`

里面分了几类：

- `xllm_hlv_4seg_1b_access24`
- `xllm_hlv_4seg_06b_access24`
- `xllm_hlv_3seg_1b_access24`
- `xllm_hlv_4seg_15b_access24`
- `xllm_hlv_4seg_1b_amazon`

历史上用到的典型模型/数据对应关系：

- 0.6B / 4seg / access24
  - model:
    - `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_36layer_xllm`
  - dataset:
    - `/export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl`
- 1B / 4seg / access24
  - model:
    - `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_60layer_xllm`
  - dataset:
    - 同上 5k access24
- 1B / 3seg / access24
  - model:
    - 同 1B / 60 层模型
  - dataset:
    - `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/datasets/requests_access24_growth_5000_user10x_short_heavy_mtfm3seg_seed20260605.jsonl`
- 1.5B / 4seg / access24
  - model:
    - `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_90layer_xllm`
- 1B / 4seg / amazon
  - model:
    - 同 1B / 60 层模型
  - dataset:
    - `/export/home/zhangshen/datas/mtgr_5000_amazon_history_mapped_jsonl_20260608/requests_amazon_history_mapped_5000_user10x_short_heavy_seg4_time_ordered_seed20260608.jsonl`

### 7.4 avg <= 400ms 的 even-QPS 极限搜索

历史汇总：

- `/tmp/mtgr_avg400_even_limit_20260611/report.md`

这个文件可以直接看最终表格，适合快速回忆“当时哪组配置能顶到多少 QPS”。

### 7.5 三个固定 seg4 长度数据集是怎么生成的

现在已经把生成脚本变成 repo 内可复用工具：

- `/tmp/xllm_feat_seg_snapshot_20260626/tools/rewrite_mtgr_segment4_fixed.py`

用法：

```bash
cd /tmp/xllm_feat_seg_snapshot_20260626

python3 tools/rewrite_mtgr_segment4_fixed.py \
  --source /export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl \
  --out-dir /tmp/mtgr_5000_seg4_fixed_len_variants_20260629 \
  --fixed-target-lengths 800,1600,2400 \
  --seed 20260602
```

输出会是：

- `.../requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605_seg4fixed800.jsonl`
- `.../requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605_seg4fixed1600.jsonl`
- `.../requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605_seg4fixed2400.jsonl`
- `.../summary.json`

## 8. 常见坑

### 8.1 不要把“当前快照可复现”和“历史策略结果”混在一起

最容易踩的坑就是：

- 代码仓里没有 `len_value`
- 但 `/tmp` 目录里还有它当年的脚本和结果

所以你能：

- 查历史结果
- 看当年怎么跑

但不能：

- 假装当前快照已经包含那套 server-side 实现

### 8.2 `auto` 不适合做论文/实验命令

因为它会按 backend 自动选 policy：

- flashinfer -> full_cache
- hopper -> prefix_only

为了让实验记录可读，建议永远显式写：

- `--mtgr_attention_backend=...`
- `--mtgr_kv_cache_policy=...`

### 8.3 `tools/mtgr_qps_sweep.py` 默认不传 `entity_id`

它会发：

- `token_ids`
- `segment_offsets`
- `segment_rules`

不会默认发：

- `entity_id`

所以任何真正在意 `entity_id` 的在线策略，如果不改 client，只能退化成“没有真实用户 ID”的版本。

### 8.4 `launch_xllm.py` 不是另一套服务框架

它只是个很薄的入口包装。做 MTGR benchmark 时，直接调二进制更稳。

### 8.5 5k / 50k 的结论边界要自己心里有数

历史上已经反复确认过：

- **5k 冷启动**不能拿来证明 eviction/admission 策略有效
- 因为很可能还没有真正填满 KV cache

所以：

- 5k 可以用来做 plumbing/smoke/correctness
- 想证明 cache policy 收益，优先看 50k 或明确产生 cache pressure 的设置

## 9. 最小复现清单

如果你只想快速确认“这个快照能不能跑通 MTGR”：

1. 编译

```bash
cd /tmp/xllm_feat_seg_snapshot_20260626
git submodule update --init --recursive
python setup.py build
```

2. 起一个 `flashinfer + full_cache`

```bash
export XLLM_BIN=$(find build -type f -path '*/xllm/xllm' | head -n 1)
export CUDA_VISIBLE_DEVICES=0
export MODEL=/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_60layer_xllm
export PORT=18080
export MASTER_PORT=19080
export TRANSFER_PORT=28080
mkdir -p /tmp/mtgr_smoke

"$XLLM_BIN" \
  --backend=rec \
  --devices=cuda:0 \
  --model="$MODEL" \
  --port="$PORT" \
  --transfer_listen_port="$TRANSFER_PORT" \
  --master_node_addr=127.0.0.1:$MASTER_PORT \
  --enable_prefix_cache=true \
  --mtgr_attention_backend=flashinfer_token_mask \
  --mtgr_kv_cache_policy=full_cache \
  --max_memory_utilization=0.8 \
  --block_size=128 \
  --max_tokens_per_batch=24384 \
  --max_seqs_per_batch=8 \
  --max_tokens_per_chunk_for_prefill=24384 \
  --enable_chunked_prefill=false \
  > /tmp/mtgr_smoke/server.log 2>&1 &

export SERVER_PID=$!
until rg -q "Brpc Server started" /tmp/mtgr_smoke/server.log; do sleep 1; done
```

3. 跑 5k smoke

```bash
python3 tools/mtgr_qps_sweep.py \
  --dataset /export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl \
  --adapter xllm \
  --url http://127.0.0.1:$PORT/v1/completions \
  --model model_kv16_60layer_xllm \
  --out-dir /tmp/mtgr_smoke/client \
  --qps 16 \
  --warmup-s 0 \
  --measure-s 312.5 \
  --timeout-s 240 \
  --max-concurrency 0 \
  --max-tokens 1 \
  --dataset-limit 5000 \
  --order sequential \
  --write-details
```

4. 收尾

```bash
kill -INT "$SERVER_PID"
wait "$SERVER_PID"
cat /tmp/mtgr_smoke/client/summary.json
```

## 10. 相关文件总览

当前快照内最关键的文件：

- 运行文档：
  - `/tmp/xllm_feat_seg_snapshot_20260626/MTGR_EXPERIMENT_RUNBOOK_20260629.md`
- 服务启动入口：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/launch_xllm.py`
- MTGR policy flag：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/common/global_flags.cpp`
- MTGR policy 选择逻辑：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/common/rec_model_utils.h`
- MTGR block manager / reuse horizon 逻辑：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/framework/block/block_manager_pool.cpp`
- QPS 客户端：
  - `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_qps_sweep.py`
- prefix cache report：
  - `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_prefix_cache_report.py`
- oracle wrapper：
  - `/tmp/xllm_feat_seg_snapshot_20260626/tools/mtgr_qps_sweep_oracle.py`
- 固定 seg4 数据集生成：
  - `/tmp/xllm_feat_seg_snapshot_20260626/tools/rewrite_mtgr_segment4_fixed.py`
- operator harness README：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/mtgr_attention_harness/README.md`
- operator harness gtest：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_attention_e2e_test.cpp`
- NVTX report helper：
  - `/tmp/xllm_feat_seg_snapshot_20260626/xllm/core/kernels/cuda/tests/profiling/mtgr_attention_nvtx_report.py`

