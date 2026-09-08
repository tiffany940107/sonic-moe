import triton
import triton.language as tl


# https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L33
@triton.jit
def _keyed_add(x, y):
    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _bitmatrix_metadata_single_stage(
    topk_indices_ptr,
    topk_scores_ptr,
    grouped_scores_ptr,
    logits_ptr,
    switch_loss_ptr,
    expert_freq_ptr,
    expert_offsets_ptr,
    x_gather_idx_ptr,
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    TK: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    LOGITS_STRIDE_T: tl.constexpr,
    LOGITS_STRIDE_E: tl.constexpr,
    HAS_GROUPED_SCORES: tl.constexpr,
    HAS_SWITCH_AUX: tl.constexpr,
):
    """Stable global expert sort and all metadata for small routing tiles."""
    offsets = tl.arange(0, BLOCK_TK)
    valid = offsets < TK
    expert = tl.load(topk_indices_ptr + offsets, mask=valid, other=0xFFFF).to(tl.uint32)
    kv_pairs = tl.sort((expert << 16) | offsets.to(tl.uint32), 0)
    sorted_expert = kv_pairs >> 16
    entry_idx = kv_pairs & 0xFFFF
    sorted_valid = offsets < TK

    tl.store(s_scatter_idx_ptr + offsets, entry_idx, mask=sorted_valid)
    tl.store(x_gather_idx_ptr + offsets, entry_idx // K, mask=sorted_valid)
    if HAS_GROUPED_SCORES:
        grouped_scores = tl.load(
            topk_scores_ptr + entry_idx,
            mask=sorted_valid,
            other=0.0,
        )
        tl.store(
            grouped_scores_ptr + offsets,
            grouped_scores,
            mask=sorted_valid,
        )
    tl.store(
        s_reverse_scatter_idx_ptr + entry_idx,
        offsets,
        mask=sorted_valid,
    )

    expert_ids = tl.arange(0, BLOCK_E)
    expert_valid = expert_ids < E
    counts = tl.sum(
        (sorted_expert[None, :] == expert_ids[:, None]) & sorted_valid[None, :],
        axis=1,
    ).to(tl.int32)
    expert_offsets = tl.cumsum(counts, axis=0) - counts
    tl.store(expert_freq_ptr + expert_ids, counts, mask=expert_valid)
    tl.store(
        expert_offsets_ptr + expert_ids,
        expert_offsets,
        mask=expert_valid,
    )
    tl.store(expert_offsets_ptr + E, TK)

    if HAS_SWITCH_AUX:
        rows = tl.arange(0, BLOCK_T)
        row_mask = rows < T
        safe_rows = tl.minimum(rows, T - 1)
        logits = tl.load(
            logits_ptr
            + safe_rows[:, None] * LOGITS_STRIDE_T
            + expert_ids[None, :] * LOGITS_STRIDE_E,
            mask=expert_valid[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        logits -= tl.max(logits, axis=1)[:, None]
        numerator = tl.exp(logits)
        probabilities = numerator / tl.sum(numerator, axis=1)[:, None]
        probabilities = tl.where(row_mask[:, None], probabilities, 0.0)
        weighted_per_token = tl.sum(
            probabilities * counts[None, :].to(tl.float32), axis=1
        )
        loss = E * tl.sum(weighted_per_token, axis=0) / (T * TK)
        tl.store(switch_loss_ptr, loss)


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit
def _bitmatrix_metadata_compute_stage1(
    expert_freq_ptr,
    expert_freq_offs_ptr,
    E: tl.constexpr,
    partial_sum_ptr,
    n_tiles,
    TK,
    BLOCK_M: tl.constexpr,  # chunk size for iterating over tiles per expert
    BLOCK_N: tl.constexpr,  # chunk size for iterating over experts in cumsum
):
    # Assume grid size == E + 1

    pid = tl.program_id(0)
    if pid < E:
        # convert partial_sum[e, *] from raw counts to exclusive prefix
        # sums over tiles. After this kernel, partial_sum[e, t] =
        # number of entries for expert e in tiles 0..t-1.

        # This is read by stage2 to locate each entry's position within expert e's contiguous output segment.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(
                expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0
            )
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Exclusive prefix sum of per-expert total counts → expert_offs[e].
        # expert_freq_offset[e] = total entries routed to expert e (from A.sum(dim=1)).
        # expert_offs[e] = sum of expert_freq_offset[0..e-1] = global start of expert e.
        curr_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)
            excl_cumsum = tl.cumsum(expert_freq, 0) - expert_freq + curr_sum
            curr_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_cumsum, mask=offs < E)
    elif pid == E + 1:
        # expert_freq_off[E] = TK (total number of entries)
        tl.store(expert_freq_offs_ptr + E, TK)


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit
def _bitmatrix_metadata_compute_stage2(
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    x_gather_idx_ptr,
    topk_indices_ptr,
    T,
    partial_sum_ptr,
    n_tiles,
    expert_offs_ptr,
    K_POW2: tl.constexpr,  # padded K, == BLOCK_SIZE / BLOCK
    K: tl.constexpr,  # actual experts per token
    TOKENS_PER_BLOCK: tl.constexpr,  # tokens per tile
):
    # One CTA per tile, same tiling as _compute_col_partial_sum_kernel.
    # For each entry (token t, k-slot k) in this tile:
    #   s_reverse_scatter_idx[entry_idx] = output position in expert-sorted order
    #   s_scatter_idx[output_pos]        = entry_idx   (inverse permutation)
    #   x_gather_idx[output_pos]         = token index (= entry_idx // K)
    #
    # Output position = expert_offs[e]          (global start of expert e)
    #                 + partial_sum[tile, e]     (entries for e in earlier tiles, after stage1)
    #                 + within_expert_rank       (position within this tile's group for e)
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2  # fast path: no padding waste
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(
        0, BLOCK_SIZE
    )  # position within this tile's flat [BLOCK*K_POW2] space
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    # Load expert id for each slot. IS_POW2_K fast path reads topk_indices as a
    # flat 1D array (no padding gaps). Non-pow2 path reads 2D with k_slot masking.
    if IS_POW2_K:
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(
            tl.uint32
        )
    else:
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    # Pack (expert, presort_offs) into a uint32 kv pair and sort by expert.
    # Upper 16 bits = expert id (sort key), lower 16 bits = pre-sort local offset.
    # Invalid slots have expert=0xffff (from other=-1 cast to uint32 >> 16).
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF  # exclude padding/OOB slots

    # Segmented scan to compute within-expert rank (0-based exclusive count).
    # scan_input packs expert id in upper 16 bits and count=1 in lower 16 bits.
    # _keyed_add resets the count at each expert boundary.
    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, _keyed_add)
    within_expert_rank = (
        inclusive_run_lengths - 1
    ) & 0xFFFF  # exclusive = inclusive - 1

    # Output position for this entry in the expert-sorted output array.
    # partial_sum layout after stage1: [n_tiles, E], stride (1, n_tiles).
    # So partial_sum[pid_m, expert] = partial_sum_ptr + pid_m*1 + expert*n_tiles.
    s_reverse_scatter_idx = tl.load(
        partial_sum_ptr + pid_m + expert * n_tiles, mask=mask
    )
    s_reverse_scatter_idx += tl.load(expert_offs_ptr + expert, mask=mask)
    s_reverse_scatter_idx += within_expert_rank

    if IS_POW2_K:
        # presort_offs == offs_local before sort; entry_idx is the flat index into
        # topk_router_indices.view(-1), i.e. token * K + k_slot.
        presort_offs = kv_pairs & 0xFFFF
        entry_idx = pid_m * BLOCK_SIZE + presort_offs
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(
            x_gather_idx_ptr + s_reverse_scatter_idx, entry_idx // K_POW2, mask=mask
        )
    else:
        # presort_offs is in K_POW2-padded space; convert to unpadded entry_idx.
        presort_offs = kv_pairs & 0xFFFF
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse_scatter_idx, token_i_global_s, mask=mask)
