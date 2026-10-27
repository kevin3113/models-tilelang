#!/usr/bin/env python3
"""Compile one full DecodeFFN model; enable optional profiling with --profile."""
import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch_npu
import cann_ops_nn
import cann_ops_tilelang as ops
from cann_ops_tilelang.config import set_token_alignment
set_token_alignment(1)

E, H, I = 48, 5120, 2304
FP8 = torch.float8_e4m3fn
COMPILE_OPTIONS = {"frozen_parameter": True, "static_kernel_compile": True}
FP4_LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                       0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


@torch.library.custom_op("tileop::swiglu_quant", mutates_args=(), device_types="npu")
def swiglu_quant_op(gate_up: torch.Tensor, psum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = gate_up.shape[0]
    hidden, packed = ops.swiglu_forward_and_per_token_cast(
        gate_up, fmt="e4m3", num_per_channels=32,
        psum_num_tokens_per_expert=psum,
        round_sf=True, clamp_value=10.0, use_packed_ue8m0=True,
    )
    scale = packed.view(torch.uint8).view(torch.float8_e8m0fnu).reshape(t, I // 64, 2)
    return hidden, scale


@swiglu_quant_op.register_fake
def _swiglu_quant_fake(gate_up, psum):
    t = gate_up.shape[0]
    return (
        gate_up.new_empty((t, I), dtype=torch.float8_e4m3fn),
        gate_up.new_empty((t, I // 64, 2), dtype=torch.float8_e8m0fnu),
    )


class DecodeFFN(torch.nn.Module):
    """One local MoE FFN: GMM1 -> SwiGLU+MX quant -> GMM2."""

    def __init__(self, w1, s1, w2, s2):
        super().__init__()
        # Already on NPU and NZ formatted: avoid model.to()/npu() after construction.
        self.register_buffer("w1", w1)
        self.register_buffer("s1", s1)
        self.register_buffer("w2", w2)
        self.register_buffer("s2", s2)
        for buffer in self.buffers():
            torch._dynamo.mark_static_address(buffer)

    @staticmethod
    def _gmm(x, weight, weight_scale, x_scale, group_list):
        return torch_npu.npu_grouped_matmul(
            [x], [weight], antiquant_scale=[weight_scale],
            per_token_scale=[x_scale], group_list=group_list,
            split_item=3, group_type=0, group_list_type=1,
            output_dtype=torch.bfloat16,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            tuning_config=[0],
        )[0]

    def gmm1(self, x, x_scale, group_list):
        return self._gmm(x, self.w1, self.s1, x_scale, group_list)

    def activation(self, gate_up, group_list):
        psum = torch.cumsum(group_list, 0).to(torch.int32)
        return swiglu_quant_op(gate_up, psum)

    def gmm2(self, hidden, hidden_scale, group_list):
        return self._gmm(hidden, self.w2, self.s2, hidden_scale, group_list)

    def forward(self, x, x_scale, group_list):
        gate_up = self.gmm1(x, x_scale, group_list)
        hidden, hidden_scale = self.activation(gate_up, group_list)
        return self.gmm2(hidden, hidden_scale, group_list)


def make_weight(n, k):
    """Random packed FP4 plus scale=2^-7; CPU copy is only for validation."""
    raw = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8)
    nz = torch_npu.npu_format_cast(
        raw.npu(), 29, customize_dtype=FP8,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    )
    scale = torch.full((E, n, k // 64, 2), 120, dtype=torch.uint8, device="npu")
    return raw, nz.transpose(1, 2), scale.transpose(1, 2)


def mxdecode(q, scale):
    value = q.cpu().float()
    # Input MX quant uses a uint8 carrier; cann_ops_nn returns typed E8M0.
    scales = (torch.pow(2., scale.cpu().to(torch.int32) - 127)
              if scale.dtype == torch.uint8 else scale.cpu().float())
    scales = scales.reshape(value.shape[0], -1)
    return value * scales.repeat_interleave(32, dim=-1)[:, :value.shape[1]]


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    return {"max_abs": float(delta.abs().max()),
            "relative_rmse": float(delta.square().mean().sqrt() /
                                   reference.square().mean().sqrt().clamp_min(1e-12)),
            "finite": bool(torch.isfinite(actual).all())}


def check_gmm(x, x_scale, y, raw, counts):
    """CPU FP32: one row/nonempty expert, 16 output channels, full K."""
    xd, yc = mxdecode(x, x_scale), y.cpu().float()
    references, actuals, start = [], [], 0
    columns = torch.linspace(0, raw.shape[1] - 1, 16).long()
    for expert, rows in enumerate(counts):
        if rows:
            packed = raw[expert, columns].long()
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            weight = FP4_LUT[codes] / 128.
            references.append(xd[start] @ weight.T)
            actuals.append(yc[start, columns])
        start += rows
    result = metrics(torch.stack(actuals), torch.stack(references))
    assert result["finite"] and result["relative_rmse"] < .015, result
    return result


def timing(model, inputs, args):
    for _ in range(args.warmup):
        output = model(*inputs)
    torch.npu.synchronize()
    device_samples, host_samples = [], []
    for _ in range(args.rounds):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter()
        for _ in range(args.iterations):
            output = model(*inputs)
        end.record()
        end.synchronize()
        host_samples.append((time.perf_counter() - start) * 1e6 / args.iterations)
        device_samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
    return {"device_us_median": float(np.median(device_samples)),
            "device_us_rounds": device_samples,
            "host_us_median": float(np.median(host_samples)),
            "host_us_rounds": host_samples}


def profile_model(compiled_model, inputs, destination, steps):
    """Independent from timing: one warmup step, then exactly `steps` captures."""
    destination.mkdir(parents=True, exist_ok=True)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=steps, repeat=1),
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(destination)),
    ) as profiler:
        for _ in range(steps + 1):
            with torch.autograd.profiler.record_function("compiled_DecodeFFN"):
                compiled_model(*inputs)
            torch.npu.synchronize()
            profiler.step()
    files = list(destination.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    assert len(files) == 1, files
    return str(files[0].resolve())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 72, 432])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--profile", action="store_true", help="Enable separate NPU profiling (default: off)")
    parser.add_argument("--profile-dir", default="profiler", help="Profiling output root; used only with --profile")
    parser.add_argument("--profile-steps", type=int, default=5)
    args = parser.parse_args()
    if len(args.tokens) != len(set(args.tokens)):
        parser.error("--tokens values must be unique")
    if (min(args.tokens) < 1 or args.warmup < 0 or
            min(args.iterations, args.rounds, args.profile_steps) < 1):
        parser.error("tokens/iterations/rounds/profile-steps must be positive; warmup >= 0")
    return args


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261025)
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print("Allocating full model dimensions...", flush=True)
    w1cpu, w1, s1 = make_weight(2 * I, H)
    w2cpu, w2, s2 = make_weight(H, I)
    model = DecodeFFN(w1, s1, w2, s2).eval()
    # The only torch.compile call: compile the entire nn.Module, never individual operators.
    compiled_model = torch.compile(
        model, backend="npugraph_ex", dynamic=False, fullgraph=True,
        options=COMPILE_OPTIONS,
    )
    print(model, flush=True)
    results = {
        "environment": {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                        "device": torch.npu.get_device_name(0), "python": platform.python_version(),
                        "cann_ops_nn": cann_ops_nn.__file__},
        "config": {"model_class": "DecodeFFN", "compile_scope": "whole_model",
                   "experts": E, "hidden": H, "intermediate": I,
                   "backend": "npugraph_ex", "dynamic": False, "fullgraph": True,
                   "options": COMPILE_OPTIONS, "swiglu": "cann_ops_nn.swiglu_group_quant",
                   "clamp": 10, "seed": 20261025, "warmup": args.warmup,
                   "iterations": args.iterations, "rounds": args.rounds,
                   "profile_enabled": args.profile, "profile_steps": args.profile_steps,
                   "run_id": run_id,
                   "scope": "local FFN only; excludes dispatch/combine, input quantization, weights, compile"},
        "cases": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        print("CASE", tokens, flush=True)
        rng = np.random.default_rng(20261025 + tokens)
        counts = rng.multinomial(tokens, np.full(E, 1 / E)).tolist()
        group_list = torch.tensor(counts, dtype=torch.int64, device="npu")
        original = torch.randn(tokens, H, device="npu", dtype=torch.bfloat16) * .1
        x, x_scale = torch_npu.npu_dynamic_mx_quant(original, dst_type=FP8)
        x_scale = x_scale.reshape(tokens, H // 64, 2)
        inputs = (x, x_scale, group_list)
        # Eager intermediate values are used only for independent stage validation.
        gate_up = model.gmm1(*inputs)
        hidden, hidden_scale = model.activation(gate_up, group_list)
        eager_output = model.gmm2(hidden, hidden_scale, group_list)
        checks = {"gmm1_cpu_fp32": check_gmm(x, x_scale, gate_up, w1cpu, counts),
                  "gmm2_cpu_fp32": check_gmm(hidden, hidden_scale, eager_output, w2cpu, counts)}
        gate, up = gate_up.cpu().float().chunk(2, dim=-1)
        reference = torch.nn.functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)
        checks["swiglu_dequantized_vs_cpu"] = metrics(mxdecode(hidden, hidden_scale), reference)
        assert checks["swiglu_dequantized_vs_cpu"]["relative_rmse"] < .06
        start = time.perf_counter()
        compiled_output = compiled_model(*inputs)
        torch.npu.synchronize()
        compile_seconds = time.perf_counter() - start
        checks["ffn_compiled_vs_eager"] = metrics(compiled_output.cpu(), eager_output.cpu())
        assert checks["ffn_compiled_vs_eager"]["finite"]
        assert checks["ffn_compiled_vs_eager"]["relative_rmse"] < 1e-5
        performance = {"compile_first_call_seconds": compile_seconds,
                       "eager": timing(model, inputs, args),
                       "compiled_static": timing(compiled_model, inputs, args)}
        # Runtime counts are mutable; weights/scales alone have static addresses.
        alternative = list(reversed(counts))
        group_list.copy_(torch.tensor(alternative, device="npu", dtype=torch.int64))
        actual = compiled_model(*inputs)
        expected = model(*inputs)
        torch.npu.synchronize()
        checks["changed_group_list_compiled_vs_eager"] = metrics(actual.cpu(), expected.cpu())
        assert checks["changed_group_list_compiled_vs_eager"]["relative_rmse"] < 1e-5
        # Restore the measured routing distribution before optional profiling.
        group_list.copy_(torch.tensor(counts, device="npu", dtype=torch.int64))
        case = {"T": tokens, "counts": counts,
                "input_shapes": {"x": list(x.shape), "x_scale": list(x_scale.shape),
                                 "gmm1_out": list(gate_up.shape), "swiglu_out": list(hidden.shape),
                                 "swiglu_scale": list(hidden_scale.shape), "gmm2_out": list(eager_output.shape)},
                "validation": checks, "performance": {"ffn": performance}}
        if args.profile:
            destination = Path(args.profile_dir) / run_id / f"T{tokens}"
            case["profile_kernel_csv"] = profile_model(
                compiled_model, inputs, destination, args.profile_steps)
            case["profile_kernel_csv_relative"] = os.path.relpath(
                case["profile_kernel_csv"], output_path.resolve().parent)
        results["cases"].append(case)
        output_path.write_text(json.dumps(results, indent=2))
        print("PASS", tokens, json.dumps(performance), flush=True)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
