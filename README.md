# model-tilelang

本目录存放 DeepSeek V4.1 Flash 的 TileLang 模型片段示例。

## deepseek-v4.1-flash

DeepSeek V4.1 Flash L0 的 **Attention 输出 + FFN 前处理** 片段。每层依次执行
7 个 ops-tilelang 算子：

```text
mhc_post_forward
  -> mhc_gemm_with_sqrsum_forward
  -> mhc_reduce_partials_and_rmsnorm_forward
  -> mhc_pre_split_mixes_forward
  -> mhc_sinkhorn_forward
  -> mhc_pre_apply_mix_forward
  -> norm_fn (ffn_norm)
```

- 可循环 4 层，默认 decode 形态 `SEQ_LEN=1`，`BATCH` 可调
  （1/8/16/32/64/128/256）；
- `HIDDEN=5120`，`MHC=4`，`MHC3=24`；
- 支持 SuperKernel + `npugraph_ex` ACL graph；
- B=1..256 实测 SK 收益 **1.17%–2.75%**（ACL graph span，最后 5 次 replay）。

详细的模型片段说明、SK 开启方法、融合结构和性能数据见：

**[deepseek-v4.1-flash/README.md](deepseek-v4.1-flash/README.md)**
