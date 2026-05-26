# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) NVFP4 block-scaled GEMM for FlashInfer.

This module provides ``mm_fp4_cutile`` — a block-scaled NVFP4 GEMM that
plugs into the existing ``flashinfer.gemm.gemm_base.mm_fp4`` dispatcher
alongside the ``cudnn``, ``cutlass``, ``trtllm``, ``cute-dsl``, and
``b12x`` backends. It targets the DeepSeek-R1 / DeepSeek-V3 NVFP4
inference path:

    out = alpha * (a @ b.T) where
        a is (M, K/2) FP4 e2m1fnx2 packed as uint8, row-major
        b is (N, K/2) FP4 e2m1fnx2 packed as uint8, row-major (caller
            passes ``b.T`` from a (K, N) col-major source — same
            convention as the cuDNN/CUTLASS backends)
        a_scale (a_descale) is (M, K//block_size) FP8 e4m3 — 2D, K-major
        b_scale (b_descale) is (K, N//block_size) FP8 e4m3 — 2D; we
            transpose internally to (N, K//block_size) for the kernel
    block_size = 16 for NVFP4

The cuTile kernel and autotune logic are ported verbatim from NVIDIA TileGym
(https://github.com/NVIDIA/TileGym), specifically
``src/tilegym/suites/flashinfer/cutile/gemm/gemm_block_scale.py``
(``bs_gemm_manual_kernel_cutile`` + ``_bs_gemm_manual_autotune_configs``
+ ``_launch_manual_path``). The TileGym kernel handles arbitrary
transpose_a/transpose_b combinations and both FP8/FP4 operands; this
flashinfer port specializes to the NT layout (transpose_a=False,
transpose_b=True) and FP4 operands only — the FP8 path is already
covered by ``flashinfer/cutile/fp8_gemm.py``. TileGym-internal helpers
(``@register_impl``, persistent_mode toggles) are stripped in favor of
equivalent public ``cuda.tile`` APIs so this module has no TileGym
runtime dependency.

Lessons applied from the BF16 cuTile port (MR adding ``mm_bf16(cutile)``)
and FP8 cuTile port (MR adding ``gemm_fp8_nt_groupwise(cutile)``):

* ``from __future__ import annotations`` is NOT used — it would convert
  the ``ct.Constant[int]`` annotations into strings at function-definition
  time and break ``cuda.tile``'s runtime introspection of the
  ``Annotated[int, ConstantAnnotation()]`` metadata.

* ``out.zero_()`` is called before the kernel launch as a defensive
  consistency measure to make the cuTile family behave uniformly with
  respect to uninitialized output buffers; the manual ct.store epilogue
  here is a pure write so it's not strictly required, but zeroing
  removes a class of surprising bugs.

* No TMA / mma_scaled variant in v1. The FP4 fast-path
  (``ct.mma_scaled`` with 5D packed scales) would require an input scale
  layout conversion from flashinfer's 2D layout_128x4 form to TileGym's
  5D packed form; that conversion is a separate kernel and a follow-up
  MR after the baseline is reviewed.
"""

from math import ceil
from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search


# Module-level tune cache:
#   key:   (M, N, K, transpose_a_int, transpose_b_int, a_dtype, str(device))
#   value: (best_cfg, kernel bound to chosen num_ctas/occupancy)
_FP4_MANUAL_TUNE_CACHE: dict = {}

# NVFP4 quantization vector size (one FP8 scale per 16 FP4 elements along K)
_NVFP4_VEC_SIZE = 16


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Ported verbatim from NVIDIA TileGym
# (https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/suites/flashinfer/cutile/gemm/gemm_block_scale.py).
# Specialized to the FP4 path of bs_gemm_manual_kernel_cutile.
@ct.kernel
def _bs_gemm_fp4_manual_kernel_cutile(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,  # 2D: [M, K // VEC_SIZE], float8_e4m3fn
    b_scale_ptr,  # 2D: [N, K // VEC_SIZE], float8_e4m3fn
    M: ct.Constant[int],
    N: ct.Constant[int],
    num_k_tiles: ct.Constant[int],
    num_pid_m: ct.Constant[int],
    num_pid_n: ct.Constant[int],
    total_tiles: ct.Constant[int],
    num_programs: ct.Constant[int],
    VEC_SIZE: ct.Constant[int],
    SCALES_PER_BLOCK_K: ct.Constant[int],  # BLOCK_K // VEC_SIZE
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """Block-scaled NVFP4 GEMM, NT layout (transpose_a=False, transpose_b=True).

    Computes ``C = (A_unp * a_scale) @ (B_unp * b_scale).T`` where ``A_unp``
    and ``B_unp`` are the FP4-unpacked (M, K) and (N, K) tensors. The
    operands ``a_ptr`` / ``b_ptr`` are FP4 e2m1fn values packed as uint8
    bytes (one byte = two FP4 elements along the original K axis).

    Per-(M, K/VEC_SIZE) and per-(N, K/VEC_SIZE) FP8-e4m3 scales are
    applied before the MMA accumulation; the kernel runs the scaled
    products in FP32.
    """
    pid = ct.bid(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    zero_pad = ct.PaddingMode.ZERO

    for current_pid in range(pid, total_tiles, num_programs):
        group_id = current_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (current_pid % group_size_m_actual)
        pid_n = (current_pid % num_pid_in_group) // group_size_m_actual

        acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

        for k in range(num_k_tiles):
            for s in range(SCALES_PER_BLOCK_K):
                # Load per-block scales (2D layout).
                a_scale_s = ct.load(
                    a_scale_ptr,
                    index=(pid_m * BLOCK_M, k * SCALES_PER_BLOCK_K + s),
                    shape=(BLOCK_M, 1),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                b_scale_s = ct.load(
                    b_scale_ptr,
                    index=(pid_n * BLOCK_N, k * SCALES_PER_BLOCK_K + s),
                    shape=(BLOCK_N, 1),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )

                # K-byte offsets for FP4 (each byte packs 2 fp4 along original K).
                k_base_byte = k * (BLOCK_K // 2)
                k_start_byte = s * (VEC_SIZE // 2)

                # ─── Load A sub-block → (BLOCK_M, VEC_SIZE) ───
                # NT layout: A is row-major (M, K_packed); read (BLOCK_M, VEC_SIZE//2) bytes
                # and unpack to (BLOCK_M, VEC_SIZE) FP4 values.
                a_bytes = ct.load(
                    a_ptr,
                    index=(pid_m * BLOCK_M, k_base_byte + k_start_byte),
                    shape=(BLOCK_M, VEC_SIZE // 2),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                a_flat = ct.reshape(a_bytes, (-1,))
                a_unp = ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn)
                a_sub = ct.reshape(a_unp, (BLOCK_M, VEC_SIZE))

                # ─── Load B sub-block → (VEC_SIZE, BLOCK_N) ───
                # NT layout: B is row-major (N, K_packed); read (BLOCK_N, VEC_SIZE//2)
                # bytes, unpack to (BLOCK_N, VEC_SIZE) FP4 values, then transpose to
                # (VEC_SIZE, BLOCK_N) for MMA.
                b_bytes = ct.load(
                    b_ptr,
                    index=(pid_n * BLOCK_N, k_base_byte + k_start_byte),
                    shape=(BLOCK_N, VEC_SIZE // 2),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                b_flat = ct.reshape(b_bytes, (-1,))
                b_unp = ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn)
                b_nv = ct.reshape(b_unp, (BLOCK_N, VEC_SIZE))
                b_sub = ct.permute(b_nv, (1, 0))  # [VEC_SIZE, BLOCK_N]

                # Upcast operands and scales to fp32 and apply scale broadcast.
                a_sub_f32 = ct.astype(a_sub, ct.float32)
                b_sub_f32 = ct.astype(b_sub, ct.float32)
                a_scale_f32 = ct.astype(a_scale_s, ct.float32)
                b_scale_f32 = ct.astype(b_scale_s, ct.float32)

                a_scale_bc = ct.broadcast_to(a_scale_f32, (BLOCK_M, VEC_SIZE))
                b_scale_t = ct.permute(b_scale_f32, (1, 0))  # [1, BLOCK_N]
                b_scale_bc = ct.broadcast_to(b_scale_t, (VEC_SIZE, BLOCK_N))

                a_scaled = a_sub_f32 * a_scale_bc
                b_scaled = b_sub_f32 * b_scale_bc

                acc = ct.mma(a_scaled, b_scaled, acc=acc)

        c_block = ct.astype(acc, c_ptr.dtype)
        ct.store(
            c_ptr,
            index=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            tile=c_block,
            order=(0, 1),
        )


def _bs_gemm_fp4_manual_autotune_configs():
    """Yield autotune configurations for the FP4 manual kernel.

    Conservative tile sizes (FP32 register pressure is higher than
    ct.mma_scaled). BLOCK_K kept at 128 to match cuTile's preferred
    K-axis chunk size for the manual loop.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] >= 10:
        # Blackwell family (B200/B300/B100: sm_100/103/120/121)
        for BM, BN, BK, nc, occ, gsm in [
            (128, 128, 128, 1, 1, 8),
            (128, 128, 128, 1, 2, 8),
            (64, 128, 128, 1, 2, 8),
            (128, 64, 128, 1, 2, 8),
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=BK,
                GROUP_SIZE_M=gsm,
                num_ctas=nc,
                occupancy=occ,
            )
    else:
        # Pre-Blackwell fallback (not officially supported — requirement
        # check rejects SM < 100, but keep entries for cuda.tile autotune
        # validity).
        for BM, BN, BK in [(128, 128, 128), (64, 128, 128)]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )


def _bs_gemm_fp4_autotune_and_launch(
    stream,
    A, B, C, As, Bs,
    M, N, K, VEC_SIZE,
):
    """Launch FP4 manual matmul kernel with exhaustive_search autotuning."""
    cache_key = (
        M, N, K,
        A.dtype, str(A.device),
    )

    if cache_key not in _FP4_MANUAL_TUNE_CACHE:
        configs = list(_bs_gemm_fp4_manual_autotune_configs())

        def grid_fn(cfg):
            num_pid_m = _cdiv(M, cfg.BLOCK_M)
            num_pid_n = _cdiv(N, cfg.BLOCK_N)
            return (num_pid_m * num_pid_n, 1, 1)

        def args_fn(cfg):
            BM = cfg.BLOCK_M
            BN = cfg.BLOCK_N
            BK = cfg.BLOCK_K
            GSM = cfg.GROUP_SIZE_M
            num_pid_m = _cdiv(M, BM)
            num_pid_n = _cdiv(N, BN)
            num_k_tiles = _cdiv(K, BK)
            total_tiles = num_pid_m * num_pid_n
            # persistent_mode == "none" (matches ocean's _compute_num_programs for "none")
            num_programs = total_tiles
            SCALES_PER_BK = BK // VEC_SIZE
            return (
                A, B, C, As, Bs,
                M, N,
                num_k_tiles,
                num_pid_m, num_pid_n,
                total_tiles, num_programs,
                VEC_SIZE, SCALES_PER_BK,
                BM, BN, BK, GSM,
            )

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs, stream, grid_fn,
            _bs_gemm_fp4_manual_kernel_cutile,
            args_fn, hints_fn,
        )
        best_cfg = result.best.config
        tuned_kernel = ct.kernel(
            _bs_gemm_fp4_manual_kernel_cutile._pyfunc,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
        )
        _FP4_MANUAL_TUNE_CACHE[cache_key] = (best_cfg, tuned_kernel)

    best_cfg, tuned_kernel = _FP4_MANUAL_TUNE_CACHE[cache_key]
    BM = best_cfg.BLOCK_M
    BN = best_cfg.BLOCK_N
    BK = best_cfg.BLOCK_K
    GSM = best_cfg.GROUP_SIZE_M
    num_pid_m = _cdiv(M, BM)
    num_pid_n = _cdiv(N, BN)
    num_k_tiles = _cdiv(K, BK)
    total_tiles = num_pid_m * num_pid_n
    num_programs = total_tiles
    SCALES_PER_BK = BK // VEC_SIZE
    ct.launch(
        stream,
        (num_pid_m * num_pid_n, 1, 1),
        tuned_kernel,
        (
            A, B, C, As, Bs,
            M, N,
            num_k_tiles,
            num_pid_m, num_pid_n,
            total_tiles, num_programs,
            VEC_SIZE, SCALES_PER_BK,
            BM, BN, BK, GSM,
        ),
    )


def mm_fp4_cutile(
    a: torch.Tensor,           # (M, K_packed = K//2) uint8 packed FP4
    b: torch.Tensor,           # (K_packed, N) uint8 (caller passes b.T to flashinfer mm_fp4)
    a_descale: torch.Tensor,   # (M, K // block_size) float8_e4m3fn 2D
    b_descale: torch.Tensor,   # (K, N // block_size) float8_e4m3fn 2D
    alpha: Optional[torch.Tensor],  # scalar float32
    out: torch.Tensor,         # (M, N) bf16/fp16
    block_size: int = 16,
) -> torch.Tensor:
    """NVFP4 block-scaled mm via cuTile manual kernel.

    Matches the input semantics of ``flashinfer.gemm.mm_fp4`` for the
    ``backend="cutile"`` path:

    * ``a``: shape (M, K/2), dtype uint8 (FP4 e2m1fnx2 packed).
    * ``b``: shape (K/2, N) — caller passes ``b_quantized.T`` from a
      column-major (K, N) source. After the caller's ``.T``, the memory
      layout is (N, K/2) row-major; we accept this and pass to the kernel.
    * ``a_descale``: shape (M, K // block_size), dtype float8_e4m3fn
      (or uint8 viewed as such), 2D K-major.
    * ``b_descale``: shape (K, N // block_size). flashinfer's docstring
      describes this layout for cuDNN/CUTLASS; we transpose internally to
      the (N, K // block_size) layout the kernel expects.
    * ``alpha``: optional scalar float32 — global scale applied post-MMA.
    * ``out``: (M, N) bf16/fp16 output buffer.
    * ``block_size``: must be 16 for NVFP4.

    Returns ``out`` (modified in place).
    """
    if block_size != _NVFP4_VEC_SIZE:
        raise NotImplementedError(
            f"cuTile mm_fp4 currently supports NVFP4 only (block_size={_NVFP4_VEC_SIZE}); "
            f"got block_size={block_size}."
        )

    # Defensive zero of the output — see module docstring rationale.
    out.zero_()

    # Resolve b to (N, K_packed) row-major. The flashinfer caller's
    # ``mm_fp4(a, b_quant.T, ...)`` produces ``b`` as a (K_packed, N)
    # column-major view; ``.T`` recovers the underlying (N, K_packed)
    # row-major storage we want.
    b_nk = b.T

    # Resolve b_descale to (N, K // block_size) row-major. flashinfer
    # passes ``b_descale`` as (K, N // block_size) (cudnn/cutlass
    # convention); transpose to match the kernel's per-(N, K/VEC_SIZE)
    # layout.
    b_scale_nk = b_descale.T

    # Reinterpret uint8 scales as float8_e4m3fn for the kernel's astype.
    a_scale_view = (
        a_descale.view(torch.float8_e4m3fn)
        if a_descale.dtype == torch.uint8
        else a_descale
    )
    b_scale_view = (
        b_scale_nk.view(torch.float8_e4m3fn)
        if b_scale_nk.dtype == torch.uint8
        else b_scale_nk
    )

    # Force contiguity — the cuTile kernel addresses both operands and
    # scales via row-major (axis 0, axis 1) order.
    if not a.is_contiguous():
        a = a.contiguous()
    if not b_nk.is_contiguous():
        b_nk = b_nk.contiguous()
    if not a_scale_view.is_contiguous():
        a_scale_view = a_scale_view.contiguous()
    if not b_scale_view.is_contiguous():
        b_scale_view = b_scale_view.contiguous()

    # Shape sanity checks.
    if a.dim() != 2 or b_nk.dim() != 2:
        raise ValueError(
            f"mm_fp4_cutile expects 2D a / b after transpose; got {a.shape} / {b_nk.shape}"
        )
    M, KA_packed = a.shape
    N, KB_packed = b_nk.shape
    if KA_packed != KB_packed:
        raise ValueError(
            f"K-packed mismatch: a has K_packed={KA_packed}, b has K_packed={KB_packed}"
        )
    K = KA_packed * 2  # logical K (FP4 is 2 elements per byte)

    if out.shape != (M, N):
        raise ValueError(
            f"out must be ({M}, {N}); got {tuple(out.shape)}"
        )
    if out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"out.dtype must be bfloat16 or float16; got {out.dtype}"
        )
    if a_scale_view.shape != (M, K // _NVFP4_VEC_SIZE):
        raise ValueError(
            f"a_descale must be ({M}, {K // _NVFP4_VEC_SIZE}); got "
            f"{tuple(a_scale_view.shape)}"
        )
    if b_scale_view.shape != (N, K // _NVFP4_VEC_SIZE):
        raise ValueError(
            f"b_descale (after transpose) must be ({N}, {K // _NVFP4_VEC_SIZE}); "
            f"got {tuple(b_scale_view.shape)}"
        )

    _bs_gemm_fp4_autotune_and_launch(
        torch.cuda.current_stream(),
        a, b_nk, out, a_scale_view, b_scale_view,
        M, N, K, _NVFP4_VEC_SIZE,
    )

    # Apply the optional global alpha scale post-MMA.
    if alpha is not None:
        out.mul_(alpha)

    return out
