"""Multi-layer loop of the DeepSeek V4.1 Flash L0 fragment, with graph mode.

Each layer runs report section 4.2 (7 public ops).  Every ops-tilelang operator
is wrapped as a ``torch.library.custom_op`` with a fake/meta implementation
(following the earlier concat graph deliverable) so the whole loop can be
captured by ``torch.compile(backend="npugraph_ex")``.

State carried between layers:

  x              : attention output, shape (B, L, H).  The attention and FFN
                   GEMMs are outside this fragment, so the fragment's own
                   ffn_norm output is fed back as the next layer's attention
                   output.
  residual       : the 4 mHC streams (B, L, 4, H), i.e. mhc_post_forward's
                   output.
  post_layer_mix : (B, L, 4, 1) produced by mhc_pre_split_mixes_forward.
  comb_res_mix   : (B, L, 4, 4) produced by mhc_sinkhorn_forward.

The operators are forward-only, so this model is for inference/benchmarking.
Configuration is via the module-level variables below; no CLI arguments.

SuperKernel is enabled on the six AscendC mHC/FFN kernels (sinkhorn remains on
PTO).  The verified SuperKernel path is eager: SK kernels are not capture-safe
under ``npugraph_ex`` in this environment (aicore error), so
``RUN_NPUGRAPH_EX`` defaults to ``False``.

Run (community TileLang build at commit fd2ddda):

  bash /home/w00821441/run_tlmain.sh \
      /home/w00821441/model-tilelang/dsv4.1-flash-attout-to-ffnpre-multilayer.py
"""

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import cann_ops_tilelang as ops
import cann_ops_tilelang.mhc.norm_fn_kernel as _norm_fn_kernel
import cann_ops_tilelang.mhc.post_kernel as _post_kernel
import cann_ops_tilelang.mhc.pre_apply_mix_kernel as _pre_apply_mix_kernel
import cann_ops_tilelang.mhc.pre_split_mixes_kernel as _pre_split_mixes_kernel
import cann_ops_tilelang.mhc.sinkhorn_kernel as _sinkhorn_kernel
import cann_ops_tilelang.normalization.norm_kernel as _norm_kernel

_SK_PASS_CONFIG = {"tl.ascend_enable_superkernel": True}


def _enable_superkernel():
    """Enable TileLang SuperKernel for the AscendC mHC/FFN kernels.

    The sinkhorn kernel keeps its explicit PTO target (the v2 SuperKernel
    adaptation requires the AscendC backend), so it is intentionally not
    patched here.
    """
    jit_objects = (
        _post_kernel.get_mhc_post_forward_kernel,
        _norm_fn_kernel.mhc_gemm_with_sqrsum_fwd_asc,
        _norm_fn_kernel.mhc_reduce_partials_and_rmsnorm_fwd_asc,
        _pre_split_mixes_kernel.mhc_pre_split_mixes_forward_kernel,
        _pre_apply_mix_kernel.mhc_pre_apply_mix_fwd_asc,
        _norm_kernel._norm_fwd_tl_ascend,
        _sinkhorn_kernel.mhc_sinkhorn_forward_kernel,
    )
    for jit_obj in jit_objects:
        jit_obj.pass_configs = {**(jit_obj.pass_configs or {}), **_SK_PASS_CONFIG}


MHC = 4
MHC3 = 24

# Default run configuration (edit these in place; no CLI arguments).
NUM_LAYERS = 4
BATCH = 8
SEQ_LEN = 1
HIDDEN = 5120
SEED = 0
# SuperKernel kernels are not capture-safe under npugraph_ex in this
# environment (aicore error during graph compilation), so the verified SK path
# is eager. Set to True to also try the compiled path.
RUN_NPUGRAPH_EX = True

ENABLE_SUPER_KERNEL = True

if ENABLE_SUPER_KERNEL:
    _enable_superkernel()


# ============================================================================
# Graph deliverables: ops-tilelang operators as torch custom ops with fake impls
# ============================================================================


@torch.library.custom_op("tileop::mhc_post", mutates_args=(), device_types="npu")
def mhc_post_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = residual.new_empty(residual.shape)
    ops.mhc_post_forward(x, residual, post_layer_mix, comb_res_mix, out=out)
    return out


@mhc_post_op.register_fake
def _mhc_post_fake(x, residual, post_layer_mix, comb_res_mix):
    return residual.new_empty(residual.shape)


@torch.library.custom_op("tileop::mhc_gemm_sqrsum", mutates_args=(), device_types="npu")
def mhc_gemm_sqrsum_op(
    x: torch.Tensor, fn: torch.Tensor, n_splits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    mhc3 = fn.shape[0]
    out_mul = x.new_empty((1, num_tokens, 1, mhc3), dtype=torch.float32)
    sqrsum = x.new_empty((1, num_tokens, 1), dtype=torch.float32)
    ops.mhc_gemm_with_sqrsum_forward(x, fn, out_mul, sqrsum, n_splits=n_splits)
    return out_mul, sqrsum


@mhc_gemm_sqrsum_op.register_fake
def _mhc_gemm_sqrsum_fake(x, fn, n_splits):
    num_tokens = x.shape[0]
    mhc3 = fn.shape[0]
    return (
        x.new_empty((1, num_tokens, 1, mhc3), dtype=torch.float32),
        x.new_empty((1, num_tokens, 1), dtype=torch.float32),
    )


@torch.library.custom_op("tileop::mhc_reduce_rmsnorm", mutates_args=(), device_types="npu")
def mhc_reduce_rmsnorm_op(
    out_mul_splitted: torch.Tensor,
    sqrsum_splitted: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = out_mul_splitted.shape[1]
    mhc3 = out_mul_splitted.shape[3]
    out_mul = out_mul_splitted.new_empty((num_tokens, 1, mhc3))
    sqrsum = out_mul_splitted.new_empty((num_tokens, 1))
    mixes = out_mul_splitted.new_empty((num_tokens, mhc3))
    ops.mhc_reduce_partials_and_rmsnorm_forward(
        out_mul_splitted,
        sqrsum_splitted,
        out_mul,
        sqrsum,
        mixes,
        rms_group_size=rms_group_size,
        rms_eps=rms_eps,
        n_splits=n_splits,
    )
    return out_mul, sqrsum, mixes


@mhc_reduce_rmsnorm_op.register_fake
def _mhc_reduce_rmsnorm_fake(
    out_mul_splitted, sqrsum_splitted, rms_group_size, rms_eps, n_splits
):
    num_tokens = out_mul_splitted.shape[1]
    mhc3 = out_mul_splitted.shape[3]
    return (
        out_mul_splitted.new_empty((num_tokens, 1, mhc3)),
        out_mul_splitted.new_empty((num_tokens, 1)),
        out_mul_splitted.new_empty((num_tokens, mhc3)),
    )


@torch.library.custom_op("tileop::mhc_pre_split", mutates_args=(), device_types="npu")
def mhc_pre_split_op(
    mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = mixes.shape[0]
    pre = mixes.new_empty((num_tokens, MHC))
    post = mixes.new_empty((num_tokens, MHC))
    comb = mixes.new_empty((num_tokens, MHC * MHC))
    ops.mhc_pre_split_mixes_forward(
        mixes,
        mhc_scale,
        mhc_base,
        pre,
        post,
        comb,
        mhc_post_mult_value=mhc_post_mult_value,
        mhc_pre_eps=mhc_pre_eps,
    )
    return pre, post, comb


@mhc_pre_split_op.register_fake
def _mhc_pre_split_fake(mixes, mhc_scale, mhc_base, mhc_post_mult_value, mhc_pre_eps):
    num_tokens = mixes.shape[0]
    return (
        mixes.new_empty((num_tokens, MHC)),
        mixes.new_empty((num_tokens, MHC)),
        mixes.new_empty((num_tokens, MHC * MHC)),
    )


@torch.library.custom_op("tileop::mhc_sinkhorn", mutates_args=(), device_types="npu")
def mhc_sinkhorn_op(comb: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    out = comb.new_empty(comb.shape)
    ops.mhc_sinkhorn_forward(comb, out, repeat=repeat, eps=eps)
    return out


@mhc_sinkhorn_op.register_fake
def _mhc_sinkhorn_fake(comb, repeat, eps):
    return comb.new_empty(comb.shape)


@torch.library.custom_op("tileop::mhc_pre_apply", mutates_args=(), device_types="npu")
def mhc_pre_apply_op(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    num_tokens, _, hidden = x.shape
    out = x.new_empty((num_tokens, hidden))
    ops.mhc_pre_apply_mix_forward(x, mix, out)
    return out


@mhc_pre_apply_op.register_fake
def _mhc_pre_apply_fake(x, mix):
    return x.new_empty((x.shape[0], x.shape[2]))


@torch.library.custom_op("tileop::ffn_norm", mutates_args=(), device_types="npu")
def ffn_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.new_empty(x.shape)
    result, _, _ = ops.norm_fn(x, weight, eps, is_rms_norm=True, out=out)
    return result


@ffn_norm_op.register_fake
def _ffn_norm_fake(x, weight, eps):
    return x.new_empty(x.shape)


# ============================================================================
# Model
# ============================================================================


class MhcFfnPreBlock(nn.Module):
    """One layer of the Attention-output + FFN-preprocessing fragment."""

    def __init__(self, hidden: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "fn",
            (torch.randn(MHC3, MHC, hidden, generator=generator) * 1e-4)
            .flatten(1, 2)
            .contiguous()
            .float(),
        )
        self.register_buffer("mhc_scale", torch.randn(3, generator=generator).float())
        self.register_buffer("mhc_base", torch.randn(MHC3, generator=generator).float())
        self.register_buffer(
            "ffn_norm_weight",
            torch.randn(hidden, generator=generator).float().mul_(0.1).add_(1.0).bfloat16(),
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ):
        batch, seq, hidden = x.shape
        num_tokens = batch * seq
        mhc_hidden = MHC * hidden

        post = mhc_post_op(x, residual, post_layer_mix, comb_res_mix)
        post_flat = post.view(num_tokens, mhc_hidden)

        if ENABLE_SUPER_KERNEL:
            torch.npu.super_kernel_scope_begin(None)

        out_mul_splitted, sqrsum_splitted = mhc_gemm_sqrsum_op(
            post_flat, self.fn, 1
        )

        if ENABLE_SUPER_KERNEL:
            torch.npu.super_kernel_scope_end(None)

        _, _, mixes = mhc_reduce_rmsnorm_op(
            out_mul_splitted,
            sqrsum_splitted,
            mhc_hidden,
            1e-6,
            1,
        )
        pre_mix, post_mix_next, comb_next = mhc_pre_split_op(
            mixes,
            self.mhc_scale,
            self.mhc_base,
            2.0,
            1e-2,
        )
        comb_out = mhc_sinkhorn_op(comb_next.view(num_tokens, MHC, MHC), 10, 1e-6)
        mixed = mhc_pre_apply_op(post.view(num_tokens, MHC, hidden), pre_mix)
        ffn_input = ffn_norm_op(
            mixed.view(batch, seq, hidden),
            self.ffn_norm_weight,
            1e-6,
        )

        return (
            ffn_input,
            post,
            post_mix_next.view(batch, seq, MHC, 1),
            comb_out.view(batch, seq, MHC, MHC),
        )


class Dsv41FlashMhcModel(nn.Module):
    """A loop of MhcFfnPreBlock layers with per-layer weights."""

    def __init__(self, num_layers: int, hidden: int, seed: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MhcFfnPreBlock(hidden, seed=seed + layer) for layer in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ):
        for block in self.blocks:
            x, residual, post_layer_mix, comb_res_mix = block(
                x, residual, post_layer_mix, comb_res_mix
            )
        return x, residual, post_layer_mix, comb_res_mix


def main():
    torch.manual_seed(SEED)
    device = "npu"
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.bfloat16, device=device)
    residual = torch.randn(BATCH, SEQ_LEN, MHC, HIDDEN, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn(BATCH, SEQ_LEN, MHC, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(BATCH, SEQ_LEN, MHC, MHC, dtype=torch.float32, device=device)

    model = Dsv41FlashMhcModel(NUM_LAYERS, HIDDEN, seed=SEED).npu()
    print(
        f"model: layers={NUM_LAYERS}, hidden={HIDDEN}, "
        f"input x={tuple(x.shape)} residual={tuple(residual.shape)}",
        flush=True,
    )

    # Eager warm-up and reference output.
    eager_out = model(x, residual, post_layer_mix, comb_res_mix)
    torch.npu.synchronize()
    print(
        f"eager output: {eager_out[0].dtype} {tuple(eager_out[0].shape)}; "
        f"residual={tuple(eager_out[1].shape)}",
        flush=True,
    )

    if not RUN_NPUGRAPH_EX:
        print(
            "npugraph_ex path skipped (RUN_NPUGRAPH_EX=False); "
            "SuperKernel eager path executed",
            flush=True,
        )
        print("### exit", flush=True)
        return

    if ENABLE_SUPER_KERNEL:
        options = {"super_kernel_optimize": True}
    else:
        options = {}

    compiled_model = torch.compile(
        model,
        backend="npugraph_ex",
        #options={"super_kernel_optimize": True, "super_kernel_debug_options": {"debug_per_op_max_core_num": 1}},
        options=options,
        dynamic=False,
        fullgraph=True,
    )
    compiled_out = compiled_model(x, residual, post_layer_mix, comb_res_mix)
    torch.npu.synchronize()
    print(
        f"compiled output: {compiled_out[0].dtype} {tuple(compiled_out[0].shape)}; "
        f"residual={tuple(compiled_out[1].shape)}",
        flush=True,
    )
    torch.testing.assert_close(
        compiled_out[0].cpu(), eager_out[0].cpu(), atol=1e-2, rtol=1e-2
    )
    print("eager/compiled outputs match", flush=True)
    print("### exit", flush=True)


if __name__ == "__main__":
    main()
