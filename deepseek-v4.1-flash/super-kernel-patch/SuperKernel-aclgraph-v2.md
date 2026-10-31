# SuperKernel 原理、TileLang 适配与网络片段收益（v2）

## 1. SuperKernel 原理（ACL Graph 模式）

### 1.1 Ascend 的执行模型

Ascend 950（910B/C也支持） 上，一个 kernel 由 FFTS 调度到 AIC（cube）和 AIV（vector）核上执行。
普通模式下每个 kernel 都是一次独立的 task：host 下发、FFTS 调度。
ACL graph 把静态算子序列捕获成一张图，replay 时省掉逐算子的 host 下发，
但图里执行的仍然是一个个独立 task。

### 1.2 SuperKernel 融合

SuperKernel（SK）把多个相邻的aicore task 合并成一个 **SuperKernel task**：

```text
普通:  task0 -> task1 -> task2 -> ... -> taskN
SK:    SuperKernel(task0..taskN)
```
ACL graph For SK 执行流程：
- 图捕获：基于ACL graph捕获一个可重复运行的modelRI实例。
- SuperKernel 融合：基于捕获的modelRI实例进行合并，并更新modelRI实例。
- 图执行：重复执行modelRI实例。

算子适配要点：
- SK 生成一个 `__sk__` 入口，参数通过 `sk::SkSystemArgs*` 和 `_sk_args` 结构体传入；
- 入口调用 `asc_init()`、`SkSystemArgs::SkInit()`，用 `SkGetNumBlocks()` 获取本次
  可用的 block 数，再用 `SkGetBlockIdx()` 做 task 级索引；
  
  | Kernel type | Block index | Runtime block count | Sub-block index |
  | --- | --- | --- | --- |
  | Vector | `SkGetBlockIdx()` | `SkGetNumBlocks()` | Original implementation |
  | Cube | Original implementation | `SkGetNumBlocks()` | Original implementation |
  | MIX 1:2 | Original implementation | `SkGetNumBlocks()` | Original implementation |
  
- 同一份计算绑定 1–4 个 SK 变体（`SK_BIND`），运行时按实际可用 block 数选择；
- 合并后的 kernel 内部完成必要同步，减少 task 边界、进行指令和数据预取优化；
- 算子计算逻辑本身不会，收益主要来自**调度开销的减少、预取相关优化**。

## 2. TileLang 适配 SuperKernel 的 patch 修改

patch：[tilelang-superkernel-v2.patch](https://github.com/kevin3113/models-tilelang/blob/main/deepseek-v4.1-flash/super-kernel-patch/tilelang-superkernel-v2.patch)，共修改 13 个文件，核心是新增：

```text
src/ascend/codegen/codegen_superkernel.h
src/ascend/codegen/codegen_superkernel.cc
docs/ascend_superkernel.md
testing/ascend/target/test_ascend_superkernel.py
testing/ascend/runtime/test_superkernel_ordinary_launch.py
```

并修改：

```text
src/ascend/codegen/codegen_ascend.cc/.h
src/ascend/codegen/rt_mod_ascend.cc
src/ascend/op/builtin.cc/.h
tilelang/transform/pass_config.py
tilelang/ascend/pipeline.py
tilelang/ascend/README.md
```

依赖特性：ascend支持schedmode [PR](https://github.com/tile-ai/tilelang/pull/3614) ，待合入。

### 2.1 新增 pass config

```python
tilelang.PassConfigKey.TL_ASCEND_ENABLE_SUPERKERNEL      # 默认 False
tilelang.PassConfigKey.TL_ASCEND_SUPERKERNEL_NUM_VARIANTS # 1..4，默认 4
tilelang.PassConfigKey.TL_ASCEND_SUPERKERNEL_DISABLE_DCCI # cap 0/4，默认 False
```

`enable_superkernel` 打开后，每个 device kernel 保留普通入口，同时生成：

- `<name>_sk_args` 参数结构体（字段按最终 device 签名，<4B 标量 `alignas(4)`）；
- `__sk__` 入口（以 MIX kernel 为例）：
  ```cpp
  __sk__ __mix__(1, 2) void <name>_sk(
      const <name>_sk_args* tl_sk_args,
      sk::SkSystemArgs* tl_sk_sys);
  ```
- `SK_BIND(<name>, cap, <name>_sk<0>, ..., <name>_sk<N-1>)`，绑定 1–4 个变体；
- `.ascend.meta` 绑定记录，供 device link 和 SK runtime 使用。

### 2.2 codegen 组织

普通 kernel 仍然走原来的 `CodeGenTileLangAscend::AddFunction`；SK 生成是之后
独立的一步：

```cpp
for (...) {
  cg.AddFunction(f);
}
EmitSuperKernel(mod, &cg);
```

- 普通 codegen 只把每函数初始化逻辑抽成 `PrepareFunction`，与 SK emitter 共享；
  没有引入 SK 模式分支或额外的语句/表达式 visitor；
- `codegen_superkernel.cc` 负责配置校验、符号检查、参数结构体、入口签名、
  task-context IR 重写、参数 unpack 和 `SK_BIND`；
- `SkEntryRewriter` 把纯 Vector kernel 的 block-index 绑定替换成显式 IR 变量，
  再把 task context 传给 VF helper；
- 源 IR module 和普通 launch metadata 不变。

### 2.3 生成代码样例  
```cpp
__global__ __vector__ void swiglu(...) {
}

template <int variant>
__sk__ __vector__ void swiglu_sk(...) {
}

SK_BIND(swiglu, 0, swiglu_sk<0>, swiglu_sk<1>, swiglu_sk<2>, swiglu_sk<3>);
```

- [完整swiglu样例](codegen_with_sk.cpp)  


## 3. dsv4.1-flash 模型片段测试

### 3.1 Attention 输出 + FFN 前处理片段介绍

**Attention 输出 + FFN 前处理**，是 L0 层内最长的连续算子片段。
每层 7 个 ops-tilelang 算子：

```text
mhc_post_forward
  -> mhc_gemm_with_sqrsum_forward
  -> mhc_reduce_partials_and_rmsnorm_forward
  -> mhc_pre_split_mixes_forward
  -> mhc_sinkhorn_forward
  -> mhc_pre_apply_mix_forward
  -> norm_fn (ffn_norm)
```

脚本把它组织成可循环的多层模型：

```python
NUM_LAYERS = 4
BATCH = ...      # 1/8/16/32/64/128/256
SEQ_LEN = 1      # decode：每序列每步 1 token
HIDDEN = 5120
MHC = 4
MHC3 = 24
```

输入/输出：

```text
x:              (B, L, 5120)     bf16
residual:       (B, L, 4, 5120)  bf16
post_layer_mix: (B, L, 4, 1)     fp32
comb_res_mix:   (B, L, 4, 4)     fp32
fn:             (24, 4*5120)     fp32
ffn_norm_weight:(5120,)          bf16

ffn_input:      (B, L, 5120)     bf16
post:           (B, L, 4, 5120)  bf16
post_mix_next:  (B, L, 4, 1)     fp32
comb_out:       (B, L, 4, 4)     fp32
```

所有算子逐 token，B/L 只通过 `T = B * L` 影响 shape。每个算子都用
`torch.library.custom_op` + fake impl 包装，保证 `fullgraph=True` 可整图捕获。
模型归档：[FFN前处理片段代码仓](https://github.com/kevin3113/models-tilelang)。

### 3.2. 开启 SuperKernel 前后的收益

测试条件：

- Ascend 950DT，`SEQ_LEN=1`，`NUM_LAYERS=4`，`HIDDEN=5120`；
- `RUN_NPUGRAPH_EX=True`，`dynamic=False`，`fullgraph=True`；
- 指标：一次 ACL graph 执行的
  `span = max(task_start + task_duration) - min(task_start)`；
- 每次运行 1 次首次编译执行 + 5 次稳态 replay，取最后 5 次；
- 每个 B 下 SK 开/关各跑 1 次，背靠背执行。

| B | no-Pre nosk / sk | Pre nosk / sk | SK 收益 no-Pre | SK 收益 Pre |
|---:|---:|---:|---:|---:|
| 1 | 109.25 / 106.25 | 117.75 / 114.75 | +3.00 (+2.75%) | +3.00 (+2.55%) |
| 8 | 125.80 / 123.50 | 140.50 / 129.75 | +2.30 (+1.83%) | **+10.75 (+7.65%)** |
| 16 | 133.20 / 131.20 | 143.25 / 137.50 | +2.00 (+1.50%) | **+5.75 (+4.01%)** |
| 32 | 138.40 / 136.15 | 154.75 / 143.75 | +2.25 (+1.63%) | **+11.00 (+7.11%)** |
| 64 | 146.70 / 144.35 | 161.75 / 152.75 | +2.35 (+1.60%) | **+9.00 (+5.56%)** |
| 128 | 166.35 / 164.40 | 185.00 / 180.00 | +1.95 (+1.17%) | **+5.00 (+2.70%)** |
| 256 | 197.95 / 193.70 | 220.75 / 212.50 | +4.25 (+2.15%) | **+8.25 (+3.74%)** |

结论：

- SK 在 B=1..256 全部有收益，降幅 **1.17%–2.75%**，在加了冷启动（模拟模型中执行）后收益扩大到**2.5%-7%**；
- 当前最多6个算子融合，`mhc_gemm_with_sqrsum_forward`这个算子有全核同步，
  核数与其他算子不一致，暂未加入融合。

注意：以上是Ascend950DT **device 侧 ACL graph 执行 span**，不包含 host graph launch、
`ForeachCopy` 输入输出搬运等前后处理，仅包括 4 层、7 算子的 L0 片段，不是完整 40 层模型。

## 5. FFN GMM1 → SwiGLU → GMM2 模型片段收益

### 5.1 样例与五组配置

FFN 片段取 DecodeFFN 的一个 local MoE：

```text
GMM1 (FP4 weight, per-token FP8 scale)
  -> (cann_ops_nn / ops-tilelang) SwiGLU + per-token MX FP8 quant
  -> GMM2 (FP4 weight)
```

E=48, H=5120, I=2304；测试 T=1/72/432。五组配置，其中GMM1/GMM2都是torch.npu上
基于AscendC开发的算子，针对B0/B/C三个场景将SwiGLU由AscendC改为TileLang实现：

| ID | SwiGLU | TileLang SK for SwiGLU | torch.compile SK |
|---|---|---|---|
| A0 | `cann_ops_nn.swiglu_group_quant` | n/a | off |
| A | `cann_ops_nn.swiglu_group_quant` | n/a | on |
| B0 | ops-tilelang `swiglu_forward_and_per_token_cast` | off | off |
| B | ops-tilelang `swiglu_forward_and_per_token_cast` | off | on |
| **C** | ops-tilelang `swiglu_forward_and_per_token_cast` | **on** | **on** |
```text
实际 B0/B/C 组算子序列为GMM1 -> Cumsum -> Cast -> SwiGLU -> GMM2
```

### 5.2 测试性能数据

GMM1 → SwiGLU → GMM2 的device执行时间（预热）：

| T | A0 | A | B0 | B | **C** |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.00 | 19.46 | 23.00 | 22.75 | **21.69** |
| 72 | 252.75 | 250.21 | 254.00 | 253.50 | **252.21** |
| 432 | 337.25 | 335.59 | 340.75 | 340.25 | **338.38** |

GMM1 → SwiGLU → GMM2 的device执行时间（冷启动，清L2和Icache）：

| T | A0 | A | B0 | B | **C** |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.25 | 24.45 | 30.75 | 29.25 | **26.50** |
| 72 | 299.50 | 295.67 | 315.25 | 302.75 | **296.54** |
| 432 | 397.25 | 392.87 | 413.25 | 401.25 | **400.27** |

- 其中对比 C 和 B 两个场景，在tilelang支持SuperKernel后的进一步收益：
  SwiGLU支持SuperKernel后，融合的算子更多可以被削减的调度开销更大，同时优化也更大。

### 5.3 TileLang 支持 SK 的价值

**TileLang 支持 SuperKernel 后，网络可以把 TileLang 算子纳入
SK 融合范围，而不是在每个TileLang算子处断开**——这也是上述样例中 C 相对 B/B0 的
结构性收益，它在更大网络中比这个 3-op 样例更具价值。
