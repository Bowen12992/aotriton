#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_offsets, dropout_rng
from masked_load_store import load_fn
from triton.language.extra import libdevice


# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M: tl.constexpr, QDIM: tl.constexpr, KDIM: tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)


@triton.jit
def bwd_inner_dk_dv(
    # I/O Tensor
    dk,
    dv,
    qk_scale,
    bias_scale,
    # Problem Description
    q_ptrs,
    q_stride,
    kt,
    vt,
    B_block_ptr,
    do_ptrs,
    do_stride,
    l_ptrs,
    D_ptrs,
    seqlen_q,
    seqlen_k,
    head_dim,
    # Sub-problem range, (lo, hi) specify the range for seqlen_q
    start_k,
    lo,
    hi,
    overflow_size,
    ## Dropout
    ### max_seqlen_k is put in Dropout section because it is not needed by
    ### anything other than dropout
    dropout_p,
    dropout_scale,
    philox_seed,
    batch_philox_offset,
    max_seqlen_k,
    # constexpr starts here
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # initialize offsets
    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)

    # Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    # DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    q_ptrs += lo * q_stride
    do_ptrs += lo * do_stride
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))

    """
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_k: select k and dV
    start_q: select q and dO
    """
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_q in range(lo, hi, BLOCK_M):
        # TODO: Unify the name, the usage of m/n is very confusing
        offs_q_curr = offs_q[:, None] + start_q  # (BLOCK_M, 1)
        # -- load q, do --
        # TODO: It is more optimal to do OOB check only in the last iter.
        # (BLOCK_M, BLOCK_DMODEL), offs = (BLOCK_M * iter, 0) = (start_q, 0)
        #
        # This common function can be further split into regular and
        # non-regular version, determined by tl.constexpr, just like the fwd kernel.

        # q = tl.load(Q_block_ptr)
        if not FULL_BLOCKS:
            q = load_fn(q_ptrs, offs_q + start_q, ld_offs_d, seqlen_q, head_dim)
        else:
            q = load_fn(q_ptrs, None, ld_offs_d, seqlen_q, head_dim)
        # do = tl.load(DO_block_ptr)
        # TODO: pre_load_do
        if not FULL_BLOCKS:
            do = load_fn(do_ptrs, offs_q + start_q, ld_offs_d, seqlen_q, head_dim)
        else:
            do = load_fn(do_ptrs, None, ld_offs_d, seqlen_q, head_dim)
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # TODO: These two checks can be optimized to occur on the last iter.
        # if not FULL_BLOCKS:
        #     if overflow_size > 0:
        #         boundary_n = tl.full((BLOCK_N, ), seqlen_q, dtype=tl.int32)
        #         mask = offs_q_curr < boundary_n[None, :]
        #         qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_q_curr >= offs_k[None, :], qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: do boundary_check correctly
            bias = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
        # q.offs = (start_q, 0), k.offs = (0, start_k)
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)  # (BLOCK_M, BLOCK_N)
        # Check for OOB accesses on D and LSE
        if FULL_BLOCKS:
            Di = tl.load(D_ptrs + offs_q_curr)
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
            d_lse_padding = tl.full((BLOCK_M,), 0, dtype=tl.float32)
            Di = tl.load(
                D_ptrs + offs_q_curr,
                mask=d_lse_ptrs_mask[:, None],
                other=d_lse_padding[:, None],
            )
            l_i = tl.load(
                l_ptrs + offs_q_curr,
                mask=d_lse_ptrs_mask[:, None],
                other=d_lse_padding[:, None],
            )
        p = tl.math.exp2(qk_scale * qk - l_i)  # (BLOCK_M, BLOCK_N)

        # if not FULL_BLOCKS or CAUSAL:
        #     if qk_scale == 0.0:
        #         p = tl.where(libdevice.isnan(p), 0.0, p)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k
            )
            # CAVEAT: do NOT update p, ds needs the original p
            if BLOCK_M == 1:
                dv += (
                    tl.where(keep, p * dropout_scale, 0.0).to(q_ptrs.dtype.element_ty)
                    * do
                )
            else:
                dv += tl.dot(
                    tl.trans(tl.where(keep, p * dropout_scale, 0.0)).to(
                        q_ptrs.dtype.element_ty
                    ),
                    do,
                )
        else:
            if BLOCK_M == 1:
                dv += p.to(q_ptrs.dtype.element_ty) * do
            else:
                dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # compute dp = dot(do, vt)
        # dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        # do.shape = (BLOCK_M, BLOCK_DMODEL) vt.shape = (BLOCK_DMODEL, BLOCK_N)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp * dropout_scale, 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di)  # (BLOCK_M, BLOCK_N)
        # compute dk
        if BLOCK_M == 1:
            dk += ds.to(q_ptrs.dtype.element_ty) * q
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), q.shape = (BLOCK_M, BLOCK_DMODEL)
            dk += tl.dot(
                tl.trans(ds.to(q_ptrs.dtype.element_ty)), q
            )  # (BLOCK_N, BLOCK_DMODEL)
        # update pointers (block_ptr code was left intentionally as comment)
        # Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        # DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0)) # Debug DO accessing problems
        q_ptrs += q_stride * BLOCK_M
        do_ptrs += do_stride * BLOCK_M
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))
    return dk, dv
