#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA3 W4A16 GPTQ kernel (gfx1100).

Exercises ``RDNA3W4A16LinearKernel`` end-to-end: it builds a layer with
GPTQ-format checkpoint parameters, runs ``process_weights_after_loading``
(weight shuffle + zero-point synthesis), then ``apply_weights``, and compares
the result against an fp32 reference dequant-and-matmul.

The kernel is exposed via ``torch.ops._rocm_C.gptq_gemm_rdna3`` and is only
built for gfx11; tests are skipped elsewhere.

Run `pytest tests/kernels/quantization/test_rdna3_w4a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E402
    RDNA3W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

device = "cuda"

WEIGHT_TYPE = scalar_types.uint4b8  # symmetric int4, bias = 8
PACK_FACTOR = 8  # 8 x 4-bit nibbles per int32

# Skip everything in this module unless we are on the only architecture the
# kernel is built/registered for.
gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
    ),
    reason="requires gfx1100 with the _rocm_C.gptq_gemm_rdna3 op built in",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference for the RDNA3 W4A16 op.

    x_mk:       [M, K] fp16/bf16 activations.
    q_int4_kn:  [K, N] int32 raw stored nibbles in [0, 15].
    scales_gn:  [K//G, N] per-group scales (act dtype).
    zeros_gn:   [K//G, N] int32 raw stored zero points in [0, 15], or None
                for the symmetric path (kernel synthesizes stored zero = 7).
    group_size: G.

    The kernel applies the GPTQv1 "+1" zero-point quirk, so the effective
    zero is ``stored_zero + 1`` (symmetric path: 7 + 1 == bias == 8).
    """
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)  # [K,N]
    if zeros_gn is None:
        z_full = torch.full(
            (K, N), float(WEIGHT_TYPE.bias), device=x_mk.device, dtype=torch.float32
        )
    else:
        z_full = (zeros_gn + 1).repeat_interleave(group_size, dim=0).to(torch.float32)
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full  # [K, N]
    out = x_mk.to(torch.float32) @ w_fp  # [M, N]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Layer construction (GPTQ checkpoint format)
# ---------------------------------------------------------------------------


def _build_layer(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Build a dummy layer carrying GPTQ-format params, as the loader would."""
    no_loader = lambda *args, **kwargs: None  # noqa: E731

    # qweight: int4 packed along K into int32 -> [K//8, N].
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales_gn.to(dtype),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
        ),
    )
    if zeros_gn is not None:
        # qzeros: int4 packed along N into int32 -> [K//G, N//8].
        qzeros = pack_quantized_values_into_int32(zeros_gn, WEIGHT_TYPE, packed_dim=1)
        layer.register_parameter(
            "qzeros",
            PackedvLLMParameter(
                data=qzeros,
                weight_loader=no_loader,
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=PACK_FACTOR,
            ),
        )
    return layer


def _run_kernel(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    K, N = q_int4_kn.shape
    has_zp = zeros_gn is not None

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=dtype,
        group_size=group_size,
        zero_points=has_zp,
        has_g_idx=False,
    )
    ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected a supported config: {reason}"

    layer = _build_layer(q_int4_kn, scales_gn, zeros_gn, dtype)
    kernel = RDNA3W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros" if has_zp else None,
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)
    return kernel.apply_weights(layer, x_mk, bias=bias)


# Relative-L2 tolerance per dtype. The bf16 path widens dequantized weights
# to fp32 and accumulates in fp32, so it matches the reference almost exactly
# (<0.4% incl. the WMMA prefill path). The fp16 path uses the exllamav2
# "+1024" bit-trick (see qdq_4_rdna3.cuh): the dequantized weight is recovered
# as the fp16 difference of two ~1024*scale magnitudes, which sheds low-order
# mantissa bits and leaves ~2-3% relative noise that accumulates over K. We
# compare on the relative Frobenius norm rather than elementwise, since the
# bit-trick noise produces large *relative* errors on individual near-zero
# outputs that carry negligible absolute weight.
_REL_L2_TOL = {torch.float16: 5e-2, torch.bfloat16: 1e-2}


def _assert_close(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype):
    rel_l2 = (out.to(torch.float32) - ref.to(torch.float32)).norm() / ref.to(
        torch.float32
    ).norm()
    tol = _REL_L2_TOL[dtype]
    assert rel_l2 < tol, f"relative L2 error {rel_l2:.4f} exceeds {tol} for {dtype}"


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


# (M, K, N, group_size). M spans the scalar decode path (small M) and the
# WMMA prefill path (M >= 16 on the bf16 dispatch). K/N satisfy the kernel's
# divisibility constraints (K % G == 0, K % 8 == 0, N % 8 == 0).
MKNG_SHAPES = [
    (1, 128, 128, 128),  # single group, decode
    (2, 256, 256, 128),  # two groups
    (8, 256, 512, 64),  # M=8 scalar, smaller group
    (16, 512, 256, 128),  # M=16 -> WMMA path for bf16
    (32, 512, 512, 64),  # larger prefill
]


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zp", [False, True], ids=["no_zp", "with_zp"])
@pytest.mark.parametrize(
    "M,K,N,G", MKNG_SHAPES, ids=[f"m{m}_k{k}_n{n}_g{g}" for m, k, n, g in MKNG_SHAPES]
)
def test_rdna3_w4a16_matches_reference(dtype, has_zp, M, K, N, G, dist_init):
    set_random_seed(0)
    assert K % G == 0 and K % PACK_FACTOR == 0 and N % PACK_FACTOR == 0

    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    zeros_gn = (
        torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
        if has_zp
        else None
    )

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None, dtype)
    ref = _reference(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)

    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, ref, dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 32], ids=["decode", "prefill"])
def test_rdna3_w4a16_bias(dtype, M, dist_init):
    """Bias is added on both the scalar (M=1) and WMMA (M=32) paths."""
    set_random_seed(0)
    K, N, G = 512, 256, 128
    groups = K // G

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    bias = (0.1 * torch.randn(N, device=device, dtype=torch.float32)).to(dtype)

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, None, G, bias, dtype)
    ref = _reference(x_mk, q_int4_kn, scales_gn, None, G, bias)

    _assert_close(out, ref, dtype)


# ---------------------------------------------------------------------------
# Asymmetric checkpoints (compressed-tensors `symmetric: false`)
# ---------------------------------------------------------------------------

ASYM_WEIGHT_TYPE = scalar_types.uint4  # asymmetric int4, no bias


def _reference_asym(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference for the asymmetric path.

    Identical to ``_reference`` except that the stored zero *is* the zero:
    there is no GPTQv1 "+1". That is what ``use_v2_format=True`` selects, and
    it is the convention compressed-tensors writes.
    """
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)
    z_full = zeros_gn.repeat_interleave(group_size, dim=0).to(torch.float32)
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full
    out = x_mk.to(torch.float32) @ w_fp
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


def _build_layer_asym(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Build the layer the way ``compressed_tensors_wNa16.py`` registers it.

    The difference from ``_build_layer`` is the one that matters here: an
    asymmetric checkpoint stores zero points as ``[N//8, K//G]`` -- packed
    along N, group index last -- registered with ``input_dim=1, output_dim=0,
    packed_dim=0``. That is the transpose of the layout the kernel reads.
    Building them the other way round, as the symmetric test above does, makes
    the layout question unreachable.
    """
    no_loader = lambda *args, **kwargs: None  # noqa: E731

    qweight = pack_quantized_values_into_int32(
        q_int4_kn, ASYM_WEIGHT_TYPE, packed_dim=0
    )

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_gn.t().contiguous().to(dtype),  # [N, K//G]
            weight_loader=no_loader,
            input_dim=1,
            output_dim=0,
        ),
    )
    # [K//G, N] -> transpose -> [N, K//G] -> pack along N -> [N//8, K//G]
    qzeros = pack_quantized_values_into_int32(
        zeros_gn.t().contiguous(), ASYM_WEIGHT_TYPE, packed_dim=0
    )
    layer.register_parameter(
        "weight_zero_point",
        PackedvLLMParameter(
            data=qzeros,
            weight_loader=no_loader,
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    return layer


def _run_kernel_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, group_size, bias, dtype,
                     g_idx=None):
    K, N = q_int4_kn.shape
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=ASYM_WEIGHT_TYPE,
        act_type=dtype,
        group_size=group_size,
        zero_points=True,
        has_g_idx=g_idx is not None,
    )
    ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected an asymmetric config: {reason}"

    layer = _build_layer_asym(q_int4_kn, scales_gn, zeros_gn, dtype)
    if g_idx is not None:
        layer.register_parameter(
            "weight_g_idx",
            RowvLLMParameter(data=g_idx.clone(), input_dim=0,
                             weight_loader=lambda *a, **k: None))
    kernel = RDNA3W4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
        w_gidx_param_name="weight_g_idx" if g_idx is not None else None,
    )
    kernel.process_weights_after_loading(layer)
    return kernel.apply_weights(layer, x_mk, bias=bias)


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,K,N,G", MKNG_SHAPES, ids=[f"m{m}_k{k}_n{n}_g{g}" for m, k, n, g in MKNG_SHAPES]
)
def test_rdna3_w4a16_asymmetric_matches_reference(dtype, M, K, N, G, dist_init):
    """An asymmetric checkpoint's own zero points, in its own layout."""
    set_random_seed(0)
    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    # spans the full range, including 0, which GPTQv1's stored=real-1 cannot
    # represent and which real AWQ checkpoints do use
    zeros_gn = torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)

    out = _run_kernel_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None, dtype)
    ref = _reference_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)

    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, ref, dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 32], ids=["decode", "prefill"])
def test_rdna3_w4a16_asymmetric_zero_point_of_zero(dtype, M, dist_init):
    """A zero point of exactly 0 is representable and correct.

    Under the GPTQv1 convention the kernel adds 1 to the stored value, so a
    real zero of 0 has no encoding; this checks the v2 path handles it.
    """
    set_random_seed(0)
    K, N, G = 512, 256, 128
    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    zeros_gn = torch.zeros((groups, N), device=device, dtype=torch.int32)

    out = _run_kernel_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None, dtype)
    ref = _reference_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)
    _assert_close(out, ref, dtype)


# ---------------------------------------------------------------------------
# Asymmetric + act-order: the configuration nothing on gfx1100 can serve today
# ---------------------------------------------------------------------------


def _reference_asym_actorder(x_mk, q_int4_kn, scales_gn, zeros_gn, g_idx,
                             group_size, bias):
    """fp32 reference with act-order: row k of the weight uses group g_idx[k]."""
    K, N = q_int4_kn.shape
    s_full = scales_gn[g_idx].to(torch.float32)          # [K, N]
    z_full = zeros_gn[g_idx].to(torch.float32)           # [K, N], v2: no +1
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full
    out = x_mk.to(torch.float32) @ w_fp
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,K,N,G", [(1, 512, 256, 128), (32, 512, 512, 64)],
                         ids=["decode", "prefill"])
def test_rdna3_w4a16_asymmetric_act_order(dtype, M, K, N, G, dist_init):
    """An asymmetric checkpoint with activation reordering.

    Hybrid rejects `has_g_idx` outright and Triton rejects it too, so on
    gfx1100 this configuration currently has no kernel at all and the layer
    fails to load. It is the case the type-gate change is actually worth
    making, as distinct from the group-size-32/64/128 cases where Hybrid
    already serves the layer and serves it faster.
    """
    set_random_seed(0)
    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    zeros_gn = torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
    # a real desc_act permutation: each row assigned to some group, shuffled
    g_idx = torch.arange(K, device=device, dtype=torch.int32) // G
    g_idx = g_idx[torch.randperm(K, device=device)]

    out = _run_kernel_asym(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None, dtype,
                           g_idx=g_idx)
    ref = _reference_asym_actorder(x_mk, q_int4_kn, scales_gn, zeros_gn,
                                   g_idx.long(), G, None)
    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, ref, dtype)
