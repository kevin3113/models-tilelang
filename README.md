# DeepSeek V4.1 Flash L0 片段：SuperKernel 融合与收益

本文档对应脚本：

```text
/home/w00821441/model-tilelang/deepseek-v4.1-flash/dsv4.1-flash-attout-to-ffnpre-multilayer.py
```

## 1. 模型片段介绍

该片段来自 `ops-tilelang-L0-max-contiguous-fragment-report.md` 第 4.2 节，
是 DeepSeek V4.1 Flash L0 层内最长的连续算子片段：**Attention 输出 + FFN 前处理**。
每层依次执行 7 个 ops-tilelang 公共算子：

1. `mhc_post_forward`：Attention 输出与 4 路 mHC 残差做 post 混合；
2. `mhc_gemm_with_sqrsum_forward`：FFN hc_mixes GEMM（N=24, K=4*H），同时算平方和；
3. `mhc_reduce_partials_and_rmsnorm_forward`：归并 partial + RMSNorm；
4. `mhc_pre_split_mixes_forward`：把 24 路 mixes 拆成 pre(4) / post(4) / comb(16)；
5. `mhc_sinkhorn_forward`：4x4 comb 矩阵的 Sinkhorn 归一化；
6. `mhc_pre_apply_mix_forward`：用 pre mix 把 4 路 mHC 压成 1 路；
7. `norm_fn`（ffn_norm / RMSNorm）：输出 FFN 输入。

脚本把该片段改造成可循环的多层模型：

```python
NUM_LAYERS = 4      # 4 层 fragment 循环
BATCH = ...         # 本次 forward 的序列数，测试范围 1/8/16/32/64/128/256
SEQ_LEN = 1         # decode：每条序列每次 forward 1 个 token
HIDDEN = 5120
MHC = 4
MHC3 = 24
```

输入/输出 shape（B=序列数，L=SEQ_LEN，T=B*L）：

```text
x:              (B, L, 5120)          bf16
residual:       (B, L, 4, 5120)       bf16
post_layer_mix: (B, L, 4, 1)          fp32
comb_res_mix:   (B, L, 4, 4)          fp32
fn:             (24, 4*5120)          fp32
ffn_norm_weight:(5120,)               bf16

输出:
ffn_input:      (B, L, 5120)          bf16
post:           (B, L, 4, 5120)       bf16
post_mix_next:  (B, L, 4, 1)          fp32
comb_out:       (B, L, 4, 4)          fp32
```

所有算子都是逐 token 的；B 和 L 只通过 `T = B * L` 影响算子 shape。
每层 7 个算子，4 层共 28 个 task（不开 SK 时）。

每个 ops-tilelang 算子都通过 `torch.library.custom_op` 包装，并注册 fake impl，
以便 `torch.compile(backend="npugraph_ex")` 可以整图捕获。

## 2. SuperKernel 开启方法

### 2.1 开关与 JIT pass config

脚本中：

```python
ENABLE_SUPER_KERNEL = True   # 默认 False；打开后启用 SK
RUN_NPUGRAPH_EX = True       # 使用 npugraph_ex 图模式
```

`_enable_superkernel()` 会给以下 7 个 JIT 对象加上：

```python
{"tl.ascend_enable_superkernel": True}
```

对象列表：

```python
_post_kernel.get_mhc_post_forward_kernel
_norm_fn_kernel.mhc_gemm_with_sqrsum_fwd_asc
_norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd_asc
_pre_split_mixes_kernel.mhc_pre_split_mixes_forward_kernel
_pre_apply_mix_kernel.mhc_pre_apply_mix_fwd_asc
_norm_kernel._norm_fwd_tl_ascend
_sinkhorn_kernel.mhc_sinkhorn_forward_kernel
```

### 2.2 npugraph_ex 编译选项

```python
if ENABLE_SUPER_KERNEL:
    options = {"super_kernel_optimize": True}
else:
    options = {}

compiled_model = torch.compile(
    model,
    backend="npugraph_ex",
    options=options,
    dynamic=False,
    fullgraph=True,
)
```

### 2.3 SK scope 的写法

GEMM 外面保留一个 `None` scope：

```python
if ENABLE_SUPER_KERNEL:
    torch.npu.super_kernel_scope_begin(None)

out_mul_splitted, sqrsum_splitted = mhc_gemm_sqrsum_op(...)

if ENABLE_SUPER_KERNEL:
    torch.npu.super_kernel_scope_end(None)
```

第 0 层的首个算子（`mhc_post_forward`）单独用同样的 `None` scope 排除出 SK：

```python
if ENABLE_SUPER_KERNEL and self.layer_idx == 0:
    torch.npu.super_kernel_scope_begin(None)

post = mhc_post_op(x, residual, post_layer_mix, comb_res_mix)

if ENABLE_SUPER_KERNEL and self.layer_idx == 0:
    torch.npu.super_kernel_scope_end(None)
```

要点：

- `MhcFfnPreBlock` 需要 `layer_idx`，model 初始化时传入 `0..3`；
- 每个 `super_kernel_scope_begin(name)` 必须有配对的 `super_kernel_scope_end(name)`，
  否则 `npugraph_ex` 会直接报 unpaired scope；
- `sinkhorn` 虽然只有 1 个核，但**保留**它的 SK pass config。它的存在是
  npugraph_ex 把“上一层尾部 + 下一层 post”融合成一个大 `SuperKernel` 的条件之一；
  不要给它单独加 scope，否则整图会从 9 个 task 拆成 17 个 task，性能反而下降。

### 2.4 最终 SK 融合结构

开启 SK 后，每张 ACL graph 是 **9 个 task**：

```text
mhc_post（普通 kernel，第 0 层）
GEMM
SuperKernel(第0层尾部 + 第1层 post)
GEMM
SuperKernel(第1层尾部 + 第2层 post)
GEMM
SuperKernel(第2层尾部 + 第3层 post)
GEMM
SuperKernel(第3层尾部)
```

其中：

- `SuperKernel`：MIX_AIV，Block Num=56（56 个 AIV）；
- GEMM：MIX_AIC，Block Num=4、Mix Block Num=8（4 AIC + 8 AIV）；
  B=256（T=256）时变为 8/16。

## 3. 收益（ACL graph span）

### 3.1 测试条件

- 设备：Ascend 950DT（共享环境）；
- `SEQ_LEN=1`，`NUM_LAYERS=4`，`HIDDEN=5120`；
- `RUN_NPUGRAPH_EX=True`，`dynamic=False`，`fullgraph=True`；
- msprof 采集，指标为一次 ACL graph 执行的 span：

```text
span = max(task_start + task_duration) - min(task_start)
```

- 每次运行 1 次首次编译执行 + 5 次稳态 replay，取最后 5 次；
- 每个 B 下 SK 开/关各跑 1 次，背靠背执行以减小设备漂移。

### 3.2 数据表

| B | T | 不开启 SK | 开启 SK | 差值 | 相对 | 每图 task 数 | 首 task | GEMM block/mix |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 1 | 109.25 us | 106.25 us | -3.00 us | -2.75% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 8 | 8 | 125.80 us | 123.50 us | -2.30 us | -1.83% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 16 | 16 | 133.20 us | 131.20 us | -2.00 us | -1.50% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 32 | 32 | 138.40 us | 136.15 us | -2.25 us | -1.63% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 64 | 64 | 146.70 us | 144.35 us | -2.35 us | -1.60% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 128 | 128 | 166.35 us | 164.40 us | -1.95 us | -1.17% | 28 -> 9 | `mhc_post` 普通 | 4/8 |
| 256 | 256 | 197.95 us | 193.70 us | -4.25 us | -2.15% | 28 -> 9 | `mhc_post` 普通 | 8/16 |

结论：

- SK 在 B=1..256 全部有收益，降幅 **1.17%–2.75%**；
- 收益来自 task 数 28 -> 9 的融合，而不是单个算子变快；
- 第 0 层首个算子用 `None` scope 排除后，首 task 是普通 `mhc_post`，
  不再承担 SK begin/end/placeholder 的额外开销；
- B=256 时 T=256，GEMM 从 4 AIC/8 AIV 扩展到 8 AIC/16 AIV。

### 3.3 注意事项

- 以上是 **device 侧 ACL graph 执行 span**，不包含 host 侧 graph launch、
  `ForeachCopy` 输入输出搬运和 msprof 开销；
- 950DT 为共享设备，不同轮次之间绝对耗时会有约 0.5–1 us 漂移；
- 只覆盖 4 层、7 算子的 L0 片段，不是完整 40 层模型；
- 全量数据在 `sk_scope_batch_sweep_layer0scope/`，本地分析报告在
  `/home/data/Claude/deepseek/layer0scope_sweep/REPORT.md`。

## 4. 运行方式

```bash
bash /home/w00821441/run_tlmain.sh \
  /home/w00821441/model-tilelang/deepseek-v4.1-flash/dsv4.1-flash-attout-to-ffnpre-multilayer.py
```

`run_tlmain.sh` 已包含 `msprof python3 ...`，会生成 PROF 目录和
`mindstudio_profiler_output/op_summary_*.csv`。

脚本备份：

```text
/home/w00821441/model-tilelang/deepseek-v4.1-flash/dsv4.1-flash-attout-to-ffnpre-multilayer.py.bak_layer0nosk_20261023_091608
```
