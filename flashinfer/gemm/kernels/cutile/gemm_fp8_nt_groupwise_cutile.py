# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) FP8 block-scaled GEMM for FlashInfer.

This module provides ``gemm_fp8_nt_groupwise_cutile`` — a block-scaled W8A8
FP8 GEMM that plugs into the existing ``flashinfer.gemm.gemm_base.gemm_fp8_nt_groupwise``
dispatcher alongside the ``cutlass`` and ``trtllm`` backends. It targets the
DeepSeek-R1 / DeepSeek-V3 FP8 inference hot path:

    out = dequant(a @ b.T) where
        a is (M, K) FP8 e4m3 row-major
        b is (N, K) FP8 e4m3 row-major (column-major view as upstream caller
            passes)
        a_scale is (M, K // block_k) FP32 K-major (per-token-group scale)
        b_scale is (N // block_n, K // block_k) FP32 (per-block scale)
    with block_n = block_k = 128 by default
    (i.e. scale_granularity_mnk = (1, 128, 128), scale_major_mode = "K").

The cuTile kernel and autotune logic are ported verbatim from NVIDIA TileGym
(https://github.com/NVIDIA/TileGym), specifically
``src/tilegym/ops/cutile/fp8_quantization_matmul.py``. TileGym-internal
decorators (``@register_impl``) and helpers (``cached_replace_hints``,
``mark_perf_ready``) are stripped in favor of equivalent public
``cuda.tile`` APIs so this module has no TileGym runtime dependency.

Lessons applied from the BF16 cuTile port (MR adding ``mm_bf16(cutile)``):

* ``from __future__ import annotations`` is NOT used — it would convert the
  ``ct.Constant[int]`` annotations into strings at function-definition time
  and break ``cuda.tile``'s runtime introspection of the
  ``Annotated[int, ConstantAnnotation()]`` metadata.

* ``out.zero_()`` is called before the kernel launch. The W8A8 kernel uses
  ``ct.scatter`` to write outputs (not a load-and-blend), so it does not
  have the ``0 * NaN = NaN`` epilogue trap that the alpha-beta kernel has;
  the zeroing here is a defensive consistency measure to make the cuTile
  family behave uniformly with respect to uninitialized output buffers.

* No TMA variant in v1 — the non-TMA path is simpler to verify. A TMA
  follow-up will be a separate MR once the baseline is reviewed.
"""

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search


# Module-level tune caches:
#   key:   (M, N, K, block_n, block_k, output_dtype_int, dtype, str(device))
#   value: (best_cfg, kernel bound to chosen num_ctas/occupancy)
_W8A8_TUNE_CACHE: dict = {}

# Fused group-GEMM tune cache:
#   key:   (G, max_m_per_group, N, K, block_n, block_k, output_dtype_int, dtype, str(device))
#   value: (best_cfg, kernel bound to chosen num_ctas/occupancy)
_W8A8_GROUP_FUSED_TUNE_CACHE: dict = {}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gemm_calculate_pid_ct(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    """Swizzle linear block id into (pid_m, pid_n) for L2 cache locality."""
    num_pid_m = ct.cdiv(M, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# Ported verbatim from NVIDIA TileGym
# (https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/fp8_quantization_matmul.py).
@ct.kernel
def _w8a8_block_fp8_matmul_kernel(
    # Tensors
    A,
    B,
    C,
    As,
    Bs,
    # Dimensions
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    # Quantization block sizes
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Strides
    STRIDE_AM: ct.Constant[int],
    STRIDE_AK: ct.Constant[int],
    STRIDE_BK: ct.Constant[int],
    STRIDE_BN: ct.Constant[int],
    STRIDE_CM: ct.Constant[int],
    STRIDE_CN: ct.Constant[int],
    STRIDE__AS_M: ct.Constant[int],
    STRIDE__AS_K: ct.Constant[int],
    STRIDE__BS_K: ct.Constant[int],
    STRIDE__BS_N: ct.Constant[int],
    # Tile parameters
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
):
    """Gather/scatter W8A8 block-scaled FP8 matmul.

    When swap_ab=1: compute (B @ A^T)^T * scales  (swap operand order).
    When swap_ab=0: compute (A @ B^T) * scales     (normal order).

    A: (M, K)  B: (N, K)  As: (M, K_groups)  Bs: (N_groups, K_groups)  C: (M, N)

    Requires BLOCK_SIZE_N == group_n and BLOCK_SIZE_K == group_k for correct
    scale indexing (one scale per tile).
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N,
        f"Kernel requires BLOCK_SIZE_N == group_n, got {BLOCK_SIZE_N} vs {GROUP_N}",
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K,
        f"Kernel requires BLOCK_SIZE_K == group_k, got {BLOCK_SIZE_K} vs {GROUP_K}",
    )

    pid = ct.bid(0)
    pid_m, pid_n = _gemm_calculate_pid_ct(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_bn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_SIZE_K
        offs_k = offs_k_base + k_start

        # Load A block: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = ct.gather(
            A,
            (offs_am[:, None], offs_k[None, :]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # Load B block: (N, K) -> (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b = ct.gather(
            B,
            (offs_bn[:, None], offs_k[None, :]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # As: (M, K_groups) -> (BLOCK_SIZE_M,)
        a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0)

        # Bs: (N_groups, K_groups) -> scalar
        b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0)
        ab_s = a_s[:, None] * b_s

        # MMA with permute for transpose
        if SWAP_AB:
            zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
            a_t = ct.permute(a, (1, 0))
            dot_result = ct.mma(b, a_t, acc=zero_acc)
            dot_result = ct.permute(dot_result, (1, 0))
        else:
            zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
            b_t = ct.permute(b, (1, 0))
            dot_result = ct.mma(a, b_t, acc=zero_acc)

        accumulator = accumulator + dot_result * ab_s

    # Cast to output dtype
    if OUTPUT_DTYPE == 0:  # torch.float32
        c = accumulator
    elif OUTPUT_DTYPE == 1:  # torch.float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # torch.bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:
        c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_cn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    ct.scatter(C, (offs_cm[:, None], offs_cn[None, :]), c, check_bounds=True)


def _w8a8_autotune_configs(block_n_quant, block_k_quant):
    """Yield autotune configurations for the W8A8 FP8 matmul kernel.

    ``BLOCK_SIZE_N`` and ``BLOCK_SIZE_K`` must equal the quantization block
    sizes for correct scale indexing (one scale per tile), so only
    ``BLOCK_SIZE_M``, ``occupancy``, and ``swap_ab`` are searched.
    """
    for block_m in [16, 32, 64, 128]:
        for occupancy in [1, 2, 4]:
            for swap_ab in [True, False]:
                yield SimpleNamespace(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n_quant,
                    BLOCK_SIZE_K=block_k_quant,
                    GROUP_SIZE_M=16,
                    num_ctas=1,
                    occupancy=occupancy,
                    swap_ab=swap_ab,
                )


def _w8a8_early_config_prune(configs, M):
    """Drop configs whose BLOCK_SIZE_M exceeds the M dimension."""
    pruned = [cfg for cfg in configs if cfg.BLOCK_SIZE_M <= M]
    return pruned if pruned else configs


def _w8a8_autotune_and_launch(
    stream,
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    block_n,
    block_k,
    output_dtype_int,
):
    """Launch W8A8 FP8 matmul kernel with exhaustive_search autotuning."""
    cache_key = (
        M,
        N,
        K,
        block_n,
        block_k,
        output_dtype_int,
        A.dtype,
        str(A.device),
    )

    if cache_key not in _W8A8_TUNE_CACHE:
        configs = _w8a8_early_config_prune(
            list(_w8a8_autotune_configs(block_n, block_k)),
            M,
        )

        def grid_fn(cfg):
            grid_m = _cdiv(M, cfg.BLOCK_SIZE_M)
            grid_n = _cdiv(N, cfg.BLOCK_SIZE_N)
            return (grid_m * grid_n, 1, 1)

        def args_fn(cfg):
            return (
                A,
                B,
                C,
                As,
                Bs,
                M,
                N,
                K,
                block_n,
                block_k,
                A.stride(-2),
                A.stride(-1),
                B.stride(1),
                B.stride(0),
                C.stride(-2),
                C.stride(-1),
                As.stride(-2),
                As.stride(-1),
                Bs.stride(1),
                Bs.stride(0),
                cfg.BLOCK_SIZE_M,
                cfg.BLOCK_SIZE_N,
                cfg.BLOCK_SIZE_K,
                cfg.GROUP_SIZE_M,
                output_dtype_int,
                int(cfg.swap_ab),
            )

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs,
            stream,
            grid_fn,
            _w8a8_block_fp8_matmul_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        tuned_kernel = ct.kernel(
            _w8a8_block_fp8_matmul_kernel._pyfunc,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
        )
        _W8A8_TUNE_CACHE[cache_key] = (best_cfg, tuned_kernel)

    best_cfg, tuned_kernel = _W8A8_TUNE_CACHE[cache_key]
    grid_m = _cdiv(M, best_cfg.BLOCK_SIZE_M)
    grid_n = _cdiv(N, best_cfg.BLOCK_SIZE_N)
    ct.launch(
        stream,
        (grid_m * grid_n, 1, 1),
        tuned_kernel,
        (
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            best_cfg.BLOCK_SIZE_M,
            best_cfg.BLOCK_SIZE_N,
            best_cfg.BLOCK_SIZE_K,
            best_cfg.GROUP_SIZE_M,
            output_dtype_int,
            int(best_cfg.swap_ab),
        ),
    )


_DTYPE_INT_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

# ─────────────────────────────────────────────────────────────────────────────
# Fused group-GEMM kernel
# ─────────────────────────────────────────────────────────────────────────────
# Ported from Ocean's ragged_block_scaled_bmm.py pattern:
#   single grid dispatch over all (group, m_tile, n_tile) tuples;
#   reads m_indptr on-device to determine per-group M slices.
# Pre-conditions (enforced by caller):
#   B_flat  : (G * N, K)           — reshape of (G, N, K)
#   Bs_flat : (G * N//block_n, K//block_k) — reshape of (G, N//block_n, K//block_k)


@ct.kernel
def _w8a8_group_fp8_fused_kernel(
    # Tensors
    A,          # (cum_m, K) FP8
    B,          # (G * N, K) FP8 — pre-reshaped; group g at rows [g*N, (g+1)*N)
    C,          # (cum_m, N) output — must be zero-initialised before launch
    As,         # (cum_m, K // block_k) FP32, K-major
    Bs,         # (G * N // block_n, K // block_k) FP32, K-major — pre-reshaped
    m_indptr,   # (G + 1,) int32 cumulative segment starts
    # Dimensions
    N: ct.Constant[int],
    K: ct.Constant[int],
    # Quantisation block sizes (BLOCK_SIZE_N == GROUP_N and BLOCK_SIZE_K == GROUP_K)
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Strides
    STRIDE_AM: ct.Constant[int],
    STRIDE_AK: ct.Constant[int],
    STRIDE_BN: ct.Constant[int],   # stride of B_flat along dim 0 (= K)
    STRIDE_BK: ct.Constant[int],   # stride of B_flat along dim 1 (= 1)
    STRIDE_CM: ct.Constant[int],
    STRIDE_CN: ct.Constant[int],
    STRIDE__AS_M: ct.Constant[int],
    STRIDE__AS_K: ct.Constant[int],
    STRIDE__BS_N: ct.Constant[int],  # stride of Bs_flat along dim 0 (= K // block_k)
    STRIDE__BS_K: ct.Constant[int],  # stride of Bs_flat along dim 1 (= 1)
    # Grid / tile geometry
    tiles_per_group: ct.Constant[int],   # num_pid_m * num_pid_n
    num_pid_m: ct.Constant[int],         # cdiv(max_m_per_group, BLOCK_SIZE_M)
    num_pid_n: ct.Constant[int],         # cdiv(N, BLOCK_SIZE_N)
    N_per_group: ct.Constant[int],       # N   (offset into B_flat for group g = g * N)
    Ns_per_group: ct.Constant[int],      # N // block_n (offset into Bs_flat for group g)
    # Tile sizes / tuning
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
):
    """Fused group-GEMM W8A8 FP8 kernel.

    Dispatches one grid tile per (group, m_tile, n_tile) triple. Each SM:
    1. Determines its group_id and (pid_m, pid_n) inside that group.
    2. Reads m_start / m_end from m_indptr on-device (one ct.gather per value).
    3. Skips if pid_m * BLOCK_SIZE_M >= (m_end - m_start).
    4. Computes A[m_start + pid_m*BM : , :] @ B[g*N + pid_n*BN : , :]^T scaled.
    5. Masks invalid rows (last partial M tile) to zero before scatter.

    C must be zeroed before launch — the masked scatter writes 0 for invalid
    rows and the next group's tiles will fill their own rows later.

    Reference: Ocean ragged_block_scaled_bmm.py.
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N,
        f"Kernel requires BLOCK_SIZE_N == GROUP_N",
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K,
        f"Kernel requires BLOCK_SIZE_K == GROUP_K",
    )

    pid = ct.bid(0)
    group_id = pid // tiles_per_group
    pid_in_group = pid % tiles_per_group

    # GROUP_SIZE_M tile swizzle within the group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    swizzle_group = pid_in_group // num_pid_in_group
    first_pid_m = swizzle_group * GROUP_SIZE_M
    group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_in_group % group_size_m_actual)
    pid_n = (pid_in_group % num_pid_in_group) // group_size_m_actual

    # Load segment boundaries from m_indptr on-device.
    m_start_t = ct.gather(m_indptr, (group_id,))
    m_end_t = ct.gather(m_indptr, (group_id + 1,))
    m_start = m_start_t.item()
    m_end = m_end_t.item()
    valid_m = m_end - m_start

    if pid_m * BLOCK_SIZE_M < valid_m:
        # Row indices for A and C (absolute within cum_m)
        offs_am = m_start + pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
        # Row indices for B_flat (group offset + tile offset within N)
        offs_bn = group_id * N_per_group + pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
        offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

        accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

        num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
        for k_tile in range(num_k_tiles):
            k_start = k_tile * BLOCK_SIZE_K
            offs_k = offs_k_base + k_start

            # Load A block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
            a_blk = ct.gather(
                A,
                (offs_am[:, None], offs_k[None, :]),
                check_bounds=True,
                padding_value=ct.float8_e4m3fn(0.0),
            )

            # Load B block from B_flat: (BLOCK_SIZE_N, BLOCK_SIZE_K)
            b_blk = ct.gather(
                B,
                (offs_bn[:, None], offs_k[None, :]),
                check_bounds=True,
                padding_value=ct.float8_e4m3fn(0.0),
            )

            # Load A scale: (BLOCK_SIZE_M,) — one scale per M row per K-tile
            a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0)

            # Load B scale from Bs_flat: scalar — one scale per (N-tile, K-tile)
            bs_row_idx = group_id * Ns_per_group + pid_n
            b_s = ct.gather(Bs, (bs_row_idx, k_tile), check_bounds=True, padding_value=0.0)
            ab_s = a_s[:, None] * b_s

            # MMA (with optional operand swap for autotuner exploration)
            if SWAP_AB:
                zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
                a_t = ct.permute(a_blk, (1, 0))
                dot_result = ct.mma(b_blk, a_t, acc=zero_acc)
                dot_result = ct.permute(dot_result, (1, 0))
            else:
                zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
                b_t = ct.permute(b_blk, (1, 0))
                dot_result = ct.mma(a_blk, b_t, acc=zero_acc)

            accumulator = accumulator + dot_result * ab_s

        # Cast accumulator to output dtype
        if OUTPUT_DTYPE == 0:
            c = accumulator
        elif OUTPUT_DTYPE == 1:
            c = ct.astype(accumulator, ct.float16)
        elif OUTPUT_DTYPE == 2:
            c = ct.astype(accumulator, ct.bfloat16)
        else:
            c = accumulator

        # Mask rows that fall beyond m_end (last partial M tile of this group)
        # so the scatter does not overwrite the next group's rows with garbage.
        # C was zero-init before launch, and the next group's tiles will fill
        # their own rows; writing 0 here is safe.
        row_valid = offs_am < m_end
        c_masked = ct.where(row_valid[:, None], c, ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c.dtype))

        offs_cm = offs_am
        offs_cn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
        ct.scatter(C, (offs_cm[:, None], offs_cn[None, :]), c_masked, check_bounds=True)


def _w8a8_group_fused_autotune_configs(block_n_quant, block_k_quant):
    """Yield autotune configs for the fused group-GEMM kernel."""
    for block_m in [16, 32, 64, 128]:
        for occupancy in [1, 2, 4]:
            for swap_ab in [True, False]:
                yield SimpleNamespace(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n_quant,
                    BLOCK_SIZE_K=block_k_quant,
                    GROUP_SIZE_M=16,
                    num_ctas=1,
                    occupancy=occupancy,
                    swap_ab=swap_ab,
                )


def _w8a8_group_fused_autotune_and_launch(
    stream,
    A, B_flat, C, As, Bs_flat, m_indptr,
    G, max_m_per_group, N, K,
    block_n, block_k,
    output_dtype_int,
):
    """Launch the fused group-GEMM kernel with exhaustive_search autotuning."""
    cache_key = (
        G, max_m_per_group, N, K, block_n, block_k, output_dtype_int,
        A.dtype, str(A.device),
    )

    if cache_key not in _W8A8_GROUP_FUSED_TUNE_CACHE:
        configs = [
            cfg for cfg in _w8a8_group_fused_autotune_configs(block_n, block_k)
            if cfg.BLOCK_SIZE_M <= max_m_per_group
        ] or list(_w8a8_group_fused_autotune_configs(block_n, block_k))[:1]

        num_pid_n = _cdiv(N, block_n)
        N_per_group = N
        Ns_per_group = _cdiv(N, block_n)

        def grid_fn(cfg):
            npm = _cdiv(max_m_per_group, cfg.BLOCK_SIZE_M)
            tiles_pg = npm * num_pid_n
            return (G * tiles_pg, 1, 1)

        def args_fn(cfg):
            npm = _cdiv(max_m_per_group, cfg.BLOCK_SIZE_M)
            tiles_pg = npm * num_pid_n
            return (
                A, B_flat, C, As, Bs_flat, m_indptr,
                N, K,
                block_n, block_k,
                A.stride(0), A.stride(1),
                B_flat.stride(0), B_flat.stride(1),
                C.stride(0), C.stride(1),
                As.stride(0), As.stride(1),
                Bs_flat.stride(0), Bs_flat.stride(1),
                tiles_pg, npm, num_pid_n,
                N_per_group, Ns_per_group,
                cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N, cfg.BLOCK_SIZE_K,
                cfg.GROUP_SIZE_M,
                output_dtype_int,
                int(cfg.swap_ab),
            )

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs, stream, grid_fn,
            _w8a8_group_fp8_fused_kernel,
            args_fn, hints_fn,
        )
        best_cfg = result.best.config
        tuned_kernel = ct.kernel(
            _w8a8_group_fp8_fused_kernel._pyfunc,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
        )
        _W8A8_GROUP_FUSED_TUNE_CACHE[cache_key] = (best_cfg, tuned_kernel)

    best_cfg, tuned_kernel = _W8A8_GROUP_FUSED_TUNE_CACHE[cache_key]
    num_pid_n = _cdiv(N, block_n)
    npm = _cdiv(max_m_per_group, best_cfg.BLOCK_SIZE_M)
    tiles_pg = npm * num_pid_n
    N_per_group = N
    Ns_per_group = _cdiv(N, block_n)
    ct.launch(
        stream,
        (G * tiles_pg, 1, 1),
        tuned_kernel,
        (
            A, B_flat, C, As, Bs_flat, m_indptr,
            N, K,
            block_n, block_k,
            A.stride(0), A.stride(1),
            B_flat.stride(0), B_flat.stride(1),
            C.stride(0), C.stride(1),
            As.stride(0), As.stride(1),
            Bs_flat.stride(0), Bs_flat.stride(1),
            tiles_pg, npm, num_pid_n,
            N_per_group, Ns_per_group,
            best_cfg.BLOCK_SIZE_M, best_cfg.BLOCK_SIZE_N, best_cfg.BLOCK_SIZE_K,
            best_cfg.GROUP_SIZE_M,
            output_dtype_int,
            int(best_cfg.swap_ab),
        ),
    )


def gemm_fp8_nt_groupwise_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: tuple = (1, 128, 128),
    scale_major_mode: str = "K",
) -> torch.Tensor:
    """BF16/FP16/FP32-out FP8 block-scaled GEMM via cuTile.

    Computes ``out = a @ b.T`` where ``a`` is FP8 e4m3 (M, K) row-major,
    ``b`` is FP8 e4m3 (N, K) row-major, and the scales ``a_scale`` /
    ``b_scale`` are applied per-block.

    Parameters
    ----------
    a : (M, K) FP8 e4m3, row-major, contiguous.
    b : (N, K) FP8 e4m3, row-major, contiguous.
    a_scale : per-token-group scale for ``a``, shape (M, K // block_k),
        K-major (``scale_major_mode == "K"``).
    b_scale : per-block scale for ``b``, shape (N // block_n, K // block_k),
        K-major.
    out : (M, N) output buffer, bf16/fp16/fp32; must be contiguous.
    scale_granularity_mnk : (m_g, n_g, k_g). The kernel currently supports
        ``m_g == 1`` (per-token-group on M); ``n_g`` becomes ``block_n`` and
        ``k_g`` becomes ``block_k``.
    scale_major_mode : ``"K"`` or ``"MN"``. The underlying cuTile kernel is
        K-major and reads ``As[m, k_group]`` and ``Bs[n_group, k_group]`` in
        that layout. When ``"MN"`` is passed the wrapper transposes the
        incoming scale views — the kernel uses arbitrary strides via its
        ``STRIDE__AS_*`` / ``STRIDE__BS_*`` constants so the non-contiguous
        view is consumed without any kernel modification.

    Returns
    -------
    The same ``out`` tensor (modified in place).
    """
    if scale_major_mode not in ("K", "MN"):
        raise NotImplementedError(
            f"cuTile gemm_fp8_nt_groupwise supports scale_major_mode in "
            f"('K', 'MN'); got {scale_major_mode!r}."
        )

    m_g, n_g, k_g = scale_granularity_mnk
    if m_g != 1:
        raise NotImplementedError(
            f"cuTile gemm_fp8_nt_groupwise requires scale_granularity_mnk[0] == 1 "
            f"(per-token-group on M); got {m_g}."
        )
    block_n, block_k = n_g, k_g

    # Defensive zero of the output: the cuTile family uniformly does
    # out.zero_() in its public entries to guard against propagating NaN/Inf
    # from uninitialized output storage (the alpha-beta family hits this
    # through the 0 * c_load term; for this kernel the ct.scatter epilogue
    # is a pure write so it's not strictly required, but zeroing keeps the
    # behaviour consistent across cuTile entries and removes a class of
    # surprising bugs).
    out.zero_()

    # Shape sanity checks — use explicit ValueErrors instead of `assert` so
    # the validation isn't elided when Python is run with `-O` (which strips
    # assert statements and would let bad inputs reach the cuda.tile kernel).
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous")
    if not (a.dim() == 2 and b.dim() == 2):
        raise ValueError("a and b must be 2D")
    M, KA = a.shape
    N, KB = b.shape
    if KA != KB:
        raise ValueError(f"a.shape[-1] ({KA}) must match b.shape[-1] ({KB})")
    K = KA
    if out.shape != (M, N):
        raise ValueError(f"out must be ({M},{N}); got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if not (a_scale.dim() == 2 and b_scale.dim() == 2):
        raise ValueError("scales must be 2D")
    # Scale shapes:
    #   K-major:  a_scale (M, K//block_k),         b_scale (N//block_n, K//block_k)
    #   MN-major: a_scale (K//block_k, M),         b_scale (K//block_k, N//block_n)
    # The cuTile kernel itself is K-major. We transpose the MN-major views
    # before passing them into the kernel — the kernel respects arbitrary
    # strides via its STRIDE__AS_*/STRIDE__BS_* constants.
    if scale_major_mode == "K":
        if a_scale.shape != (M, _cdiv(K, block_k)):
            raise ValueError(
                f"a_scale must be ({M}, {_cdiv(K, block_k)}); got {tuple(a_scale.shape)}"
            )
        if b_scale.shape != (_cdiv(N, block_n), _cdiv(K, block_k)):
            raise ValueError(
                f"b_scale must be ({_cdiv(N, block_n)}, {_cdiv(K, block_k)}); "
                f"got {tuple(b_scale.shape)}"
            )
        a_scale_k = a_scale
        b_scale_k = b_scale
    else:  # MN-major
        if a_scale.shape != (_cdiv(K, block_k), M):
            raise ValueError(
                f"a_scale must be ({_cdiv(K, block_k)}, {M}); got {tuple(a_scale.shape)}"
            )
        if b_scale.shape != (_cdiv(K, block_k), _cdiv(N, block_n)):
            raise ValueError(
                f"b_scale must be ({_cdiv(K, block_k)}, {_cdiv(N, block_n)}); "
                f"got {tuple(b_scale.shape)}"
            )
        # transpose() returns a stride-only view; no copy. The kernel reads
        # via STRIDE__AS_M / STRIDE__AS_K so non-contiguous strides are fine.
        a_scale_k = a_scale.transpose(0, 1)
        b_scale_k = b_scale.transpose(0, 1)

    out_dtype_int = _DTYPE_INT_MAP.get(out.dtype)
    if out_dtype_int is None:
        raise ValueError(
            f"out.dtype {out.dtype} not supported by cuTile gemm_fp8_nt_groupwise; "
            f"expected bf16 / fp16 / fp32"
        )

    # Pin the stream to ``a.device`` for multi-GPU correctness — same fix as
    # gemm.py / bmm.py.
    _w8a8_autotune_and_launch(
        torch.cuda.current_stream(a.device),
        a,
        b,
        out,
        a_scale_k,
        b_scale_k,
        M,
        N,
        K,
        block_n,
        block_k,
        out_dtype_int,
    )
    return out


def group_gemm_fp8_nt_groupwise_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: tuple = (1, 128, 128),
    scale_major_mode: str = "K",
) -> torch.Tensor:
    """Grouped FP8 block-scaled GEMM via cuTile.

    Computes a stacked sequence of FP8 GEMMs sharing the same K / N but with
    per-group M slices, where the per-group inputs are concatenated along the
    M axis. Used heavily in MoE workloads where each expert is a different
    group.

    Implemented as a *fused single-kernel* dispatch: one CUDA grid covers all
    ``(group, m_tile, n_tile)`` tuples.  Each thread block reads its group's
    ``m_indptr`` boundaries on-device, computes its FP8 MMA tile, masks partial
    M tiles, and scatters the result to the correct rows of ``C``.  This avoids
    the per-group kernel launch overhead of the earlier loop-based baseline and
    enables the autotuner to optimise across groups jointly.

    Design follows the Ocean ``ragged_block_scaled_bmm.py`` persistent-scheduler
    pattern.  ``B`` and ``Bs`` are pre-flattened to ``(G*N, K)`` and
    ``(G*N//block_n, K//block_k)`` before launch.

    Parameters
    ----------
    a : (cum_m, K) FP8 e4m3 / e5m2, row-major, contiguous.
    b : (num_groups, N, K) FP8 e4m3 / e5m2, row-major.
    a_scale :
        K-major: (cum_m, K // block_k) per-token-group scale.
        MN-major: (K // block_k, cum_m).
    b_scale :
        K-major: (num_groups, N // block_n, K // block_k) per-block scale.
        MN-major: (num_groups, K // block_k, N // block_n).
    m_indptr : (num_groups + 1,) int32 cumulative segment starts.
    out : (cum_m, N) bf16 / fp16 / fp32, contiguous (modified in place).
    scale_granularity_mnk : (m_g, n_g, k_g); m_g must be 1.
    scale_major_mode : ``"K"`` or ``"MN"``.

    Returns
    -------
    The same ``out`` tensor (modified in place).
    """
    if scale_major_mode not in ("K", "MN"):
        raise NotImplementedError(
            f"cuTile group_gemm_fp8_nt_groupwise supports scale_major_mode in "
            f"('K', 'MN'); got {scale_major_mode!r}."
        )
    m_g, n_g, k_g = scale_granularity_mnk
    if m_g != 1 or (n_g, k_g) != (128, 128):
        raise NotImplementedError(
            f"cuTile group_gemm_fp8_nt_groupwise requires scale_granularity_mnk=(1, 128, 128); "
            f"got {scale_granularity_mnk}."
        )
    if a.dim() != 2:
        raise ValueError(f"a must be 2D (cum_m, K); got shape {tuple(a.shape)}")
    if b.dim() != 3:
        raise ValueError(f"b must be 3D (num_groups, N, K); got shape {tuple(b.shape)}")
    if a_scale.dim() != 2:
        raise ValueError(
            f"a_scale must be 2D; got shape {tuple(a_scale.shape)}"
        )
    if b_scale.dim() != 3:
        raise ValueError(
            f"b_scale must be 3D; got shape {tuple(b_scale.shape)}"
        )
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32; got {m_indptr.dtype}")
    num_groups = b.shape[0]
    if m_indptr.shape != (num_groups + 1,):
        raise ValueError(
            f"m_indptr shape mismatch: expected ({num_groups + 1},); "
            f"got {tuple(m_indptr.shape)}"
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    out_dtype_int = _DTYPE_INT_MAP.get(out.dtype)
    if out_dtype_int is None:
        raise ValueError(
            f"out.dtype {out.dtype} not supported by cuTile group_gemm_fp8_nt_groupwise; "
            f"expected bf16 / fp16 / fp32"
        )

    G = num_groups
    _cum_m, K = a.shape
    N = b.shape[1]
    block_n, block_k = n_g, k_g

    # Pull m_indptr to host once to compute max_m_per_group (needed for grid
    # sizing) without per-tile GPU->CPU syncs later.
    m_indptr_cpu = m_indptr.cpu().tolist()
    max_m_per_group = max(
        (m_indptr_cpu[g + 1] - m_indptr_cpu[g] for g in range(G)),
        default=0,
    )
    if max_m_per_group == 0:
        # All groups are empty — nothing to compute.
        return out

    # Flatten B and Bs along the group axis so the fused kernel can index them
    # with a single linear row index (group_id * N + pid_n * BN + ...).
    #
    # K-major layout (the kernel's native layout):
    #   a_scale : (cum_m, K//block_k)
    #   b_scale : (G, N//block_n, K//block_k) → reshape to (G*N//block_n, K//block_k)
    #
    # MN-major layout: the incoming tensors have the M and K axes swapped
    # relative to K-major.  We expose K-major *views* to the kernel:
    #   a_scale : (K//block_k, cum_m) → transpose to (cum_m, K//block_k)
    #             The kernel reads via STRIDE__AS_M / STRIDE__AS_K, so the
    #             non-contiguous transposed strides are handled correctly.
    #   b_scale : (G, K//block_k, N//block_n) → permute axes 1↔2 to get
    #             (G, N//block_n, K//block_k), make contiguous, then reshape.
    #             A contiguous copy is unavoidable here because the permuted
    #             view cannot be reshaped without copying.
    B_flat = b.reshape(G * N, K)  # b is (G, N, K) contiguous → safe to reshape

    if scale_major_mode == "K":
        a_scale_k = a_scale  # (cum_m, K//block_k) already K-major
        Bs_flat = b_scale.reshape(
            G * _cdiv(N, block_n), _cdiv(K, block_k)
        )  # (G, N//bn, K//bk) contiguous → safe
    else:  # MN-major
        # Transpose a_scale view (no copy; kernel strides handle the layout).
        a_scale_k = a_scale.transpose(0, 1)  # (cum_m, K//block_k) non-contiguous
        # Permute b_scale (G, K//bk, N//bn) → (G, N//bn, K//bk) then flatten.
        Bs_flat = (
            b_scale.permute(0, 2, 1)
            .contiguous()  # makes (G, N//bn, K//bk) contiguous
            .reshape(G * _cdiv(N, block_n), _cdiv(K, block_k))
        )

    # Zero output once before the fused launch.  Individual rows of C that
    # belong to a valid group tile are overwritten by the kernel; rows beyond
    # m_end for each group are zeroed by the ct.where mask inside the kernel,
    # so a global zero-init here ensures no stale garbage escapes.
    out.zero_()

    _w8a8_group_fused_autotune_and_launch(
        torch.cuda.current_stream(a.device),
        a, B_flat, out, a_scale_k, Bs_flat, m_indptr,
        G, max_m_per_group, N, K,
        block_n, block_k,
        out_dtype_int,
    )
    return out
