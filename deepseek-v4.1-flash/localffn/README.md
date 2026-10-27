# FFN GMM1 → SwiGLU → GMM2 profiling bundle (Ascend 950DT)

Date: 2026-10-27
Device: Ascend 950DT
Model fragment: DecodeFFN, E=48 experts, H=5120, I=2304

contains five configurations:

| ID | SwiGLU | TileLang SK for SwiGLU | torch.compile SK |
|---|---|---|---|
| A0 | `cann_ops_nn.swiglu_group_quant` | n/a | **off** |
| A | `cann_ops_nn.swiglu_group_quant` | n/a | on |
| B0 | ops-tilelang `swiglu_forward_and_per_token_cast` | off | **off** |
| B | ops-tilelang `swiglu_forward_and_per_token_cast` | off | on |
| C | ops-tilelang `swiglu_forward_and_per_token_cast` | on | on |

```text
bench_ffn_no_sk.py          # A0
bench_ffn_torch_sk.py       # A
bench_ffn_ops_no_sk.py      # B0
bench_ffn_ops_torch_sk_off.py  # B
bench_ffn_ops_torch_sk_on.py   # C
```
