"""Unit tests for the cuTile backend of flashinfer.gemm.mm_fp4.

The cuTile path lives in ``flashinfer.cutile.fp4_gemm.mm_fp4_cutile``
and is wired into ``flashinfer.gemm.mm_fp4`` with ``backend="cutile"``.

Scoped to the cuTile-only quirks:

* Supports NVFP4 only (``use_nvfp4=True``, ``block_size=16``).
* Supports the 128x4 scale factor layout only (``use_8x4_sf_layout=False``).
* Requires SM >= 100 (Blackwell) and the cuda-tile python package.

The test reference is a pure-PyTorch fp32 implementation that unpacks
the FP4 bytes via the e2m1 lookup table and applies the per-(M, K/16)
and per-(N, K/16) FP8 e4m3 scales — same arithmetic the kernel does,
but in fp32. This avoids the dependency on
``flashinfer.nvfp4_quantize`` which is tested elsewhere.
"""

import math

import pytest
import torch

from flashinfer.gemm import mm_fp4
from flashinfer.utils import get_compute_capability


# NVFP4 e2m1 lookup table.
# Bit layout: [sign(1), exp(2), mantissa(1)]
# 0000=+0, 0001=+0.5, 0010=+1, 0011=+1.5, 0100=+2, 0101=+3, 0110=+4, 0111=+6
# 1000=-0, 1001=-0.5, 1010=-1, 1011=-1.5, 1100=-2, 1101=-3, 1110=-4, 1111=-6
_FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _cutile_available() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return True


def _skip_if_not_supported():
    if not _cutile_available():
        pytest.skip("cuda-tile not installed in this environment.")
    cc = get_compute_capability(torch.device("cuda"))
    cc_num = cc[0] * 10 + cc[1]
    if cc_num < 100:
        pytest.skip(f"cuTile mm_fp4 targets SM >= 100; detected sm{cc_num}.")


def _unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 FP4 e2m1 packed bytes to fp32 along the last dim.

    Input shape: ``(..., K_packed)`` — each byte holds two consecutive
    FP4 values along the original K axis (low nibble first).
    Output shape: ``(..., K_packed * 2)``.
    """
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    lo_idx = (packed & 0x0F).long()
    hi_idx = ((packed >> 4) & 0x0F).long()
    lo = lut[lo_idx]
    hi = lut[hi_idx]
    # Interleave lo, hi along the last dim so output[k] reflects the
    # original-K ordering.
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.flatten(-2)


def _fp4_ref_matmul(
    a_packed: torch.Tensor,   # (M, K_packed=K//2) uint8
    b_packed: torch.Tensor,   # (N, K_packed=K//2) uint8
    a_scale: torch.Tensor,    # (M, K//16) float8_e4m3fn
    b_scale: torch.Tensor,    # (N, K//16) float8_e4m3fn
    alpha: torch.Tensor,      # scalar fp32
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: dequantize + scale + matmul in fp32, then cast to out_dtype."""
    VEC_SIZE = 16
    M, K_packed = a_packed.shape
    N, _ = b_packed.shape
    K = K_packed * 2
    a_unp = _unpack_fp4_e2m1(a_packed)              # (M, K)
    b_unp = _unpack_fp4_e2m1(b_packed)              # (N, K)
    a_scale_f = a_scale.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:, :K]
    b_scale_f = b_scale.to(torch.float32).repeat_interleave(VEC_SIZE, dim=1)[:, :K]
    a_eff = a_unp * a_scale_f                       # (M, K)
    b_eff = b_unp * b_scale_f                       # (N, K)
    ref = (a_eff @ b_eff.T) * alpha.to(torch.float32)
    return ref.to(out_dtype)


def _build_nvfp4_inputs(M, N, K, device="cuda"):
    """Build raw uint8 packed FP4 + FP8 e4m3 scales matching mm_fp4 signature.

    Mirrors the input layout used by ``tests/gemm/test_mm_fp4.py``: the
    natural N-major scale tensor is generated, then transposed via ``.T``
    (a non-contiguous view) for the call into ``mm_fp4``.

    Returns:
    * ``a``: (M, K/2) uint8 row-major — ``mm_fp4`` ``a`` argument
    * ``b``: (K/2, N) uint8 col-major view — ``mm_fp4`` ``b`` argument
    * ``a_descale``: (M, K/16) float8_e4m3fn row-major — ``mm_fp4`` ``a_descale``
    * ``b_descale``: (K/16, N) float8_e4m3fn col-major view — ``mm_fp4`` ``b_descale``
    * ``b_NK``: the row-major (N, K/2) source for ``b`` (for ref math)
    * ``b_descale_NK``: the row-major (N, K/16) source for ``b_descale`` (for ref math)
    """
    torch.manual_seed(0)
    VEC_SIZE = 16
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    b_NK = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    b = b_NK.T  # (K/2, N) col-major view → flashinfer's ``b`` argument
    # Scales — keep values modest so the fp32 reference doesn't overflow fp16.
    a_descale = (torch.rand(M, K // VEC_SIZE, device=device) * 0.5).to(torch.float8_e4m3fn)
    b_descale_NK = (torch.rand(N, K // VEC_SIZE, device=device) * 0.5).to(torch.float8_e4m3fn)
    b_descale = b_descale_NK.T  # (K/16, N) col-major view → flashinfer's ``b_descale``
    return a, b, a_descale, b_descale, b_NK, b_descale_NK


@pytest.mark.parametrize("m", [128, 256])
@pytest.mark.parametrize("n", [1024, 4096])
@pytest.mark.parametrize("k", [2048, 7168])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4_cutile(m, n, k, out_dtype):
    """cuTile NVFP4 mm must agree with the fp32 dequant reference within cos_sim > 0.99."""
    _skip_if_not_supported()

    # Use raw uint8 FP4 + FP8 scales — no dependency on flashinfer.nvfp4_quantize.
    a, b, a_descale, b_descale, b_NK, b_descale_NK = _build_nvfp4_inputs(m, n, k)
    alpha = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    # Compute fp32 reference from N-major layouts.
    ref = _fp4_ref_matmul(a, b_NK, a_descale, b_descale_NK, alpha, out_dtype)

    out = mm_fp4(
        a=a,
        b=b,
        a_descale=a_descale,
        b_descale=b_descale,
        alpha=alpha,
        out_dtype=out_dtype,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutile",
        use_nvfp4=True,
    )

    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    ).item()
    assert cos > 0.99, (
        f"cuTile mm_fp4 cos_sim vs fp32 ref = {cos:.6f} (expected > 0.99) at "
        f"m={m}, n={n}, k={k}, out_dtype={out_dtype}"
    )


def test_mm_fp4_cutile_rejects_8x4_sf_layout():
    """The cuTile path only supports the 128x4 scale layout; 8x4 must raise."""
    _skip_if_not_supported()

    a, b, a_descale, b_descale, _, _ = _build_nvfp4_inputs(128, 1024, 2048)
    alpha = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="128x4 scale factor layout"):
        mm_fp4(
            a=a, b=b, a_descale=a_descale, b_descale=b_descale, alpha=alpha,
            out_dtype=torch.bfloat16, block_size=16,
            use_8x4_sf_layout=True, backend="cutile", use_nvfp4=True,
        )


def test_mm_fp4_cutile_rejects_mxfp4():
    """The cuTile path only supports NVFP4; mxfp4 must raise."""
    _skip_if_not_supported()

    a, b, a_descale, b_descale, _, _ = _build_nvfp4_inputs(128, 1024, 2048)
    alpha = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="NVFP4"):
        mm_fp4(
            a=a, b=b, a_descale=a_descale, b_descale=b_descale, alpha=alpha,
            out_dtype=torch.bfloat16, block_size=32,
            use_8x4_sf_layout=False, backend="cutile", use_nvfp4=False,
        )


def test_mm_fp4_cutile_repeat_uses_tune_cache():
    """Two back-to-back calls at the same shape must hit the cuTile tune cache."""
    _skip_if_not_supported()

    a, b, a_descale, b_descale, _, _ = _build_nvfp4_inputs(128, 1024, 2048)
    alpha = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    from flashinfer.cutile.fp4_gemm import _FP4_MANUAL_TUNE_CACHE
    _FP4_MANUAL_TUNE_CACHE.clear()
    out1 = mm_fp4(
        a=a, b=b, a_descale=a_descale, b_descale=b_descale, alpha=alpha,
        out_dtype=torch.bfloat16, block_size=16,
        use_8x4_sf_layout=False, backend="cutile", use_nvfp4=True,
    )
    assert len(_FP4_MANUAL_TUNE_CACHE) == 1
    out2 = mm_fp4(
        a=a, b=b, a_descale=a_descale, b_descale=b_descale, alpha=alpha,
        out_dtype=torch.bfloat16, block_size=16,
        use_8x4_sf_layout=False, backend="cutile", use_nvfp4=True,
    )
    assert len(_FP4_MANUAL_TUNE_CACHE) == 1
    cos = torch.nn.functional.cosine_similarity(
        out1.reshape(-1).float(), out2.reshape(-1).float(), dim=0
    ).item()
    assert cos > 0.999


if __name__ == "__main__":
    pytest.main([__file__])
