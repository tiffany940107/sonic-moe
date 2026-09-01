# Correctness Audit

No correctness-invalid latency is included in the formal tables.

## C0: MXFP8 operands and tails

- block-scale `K=1280`, `granK=32` tested expert segments
  `[0,1,127,128,129,517]` with zero qdata and scale-byte mismatches;
- compute-sanitizer reported `ERROR SUMMARY: 0 errors`;
- Sonic variable-M pack/unpack tests cover empty experts and 127/128/129-row
  boundaries with byte-exact active E8M0 scales;
- a direct old/fixed block-scale A/B guard test launches logical 10 rows from a
  12-row backing allocation at K=1280/granK=32: old `0da85e6` overwrites 1,804
  int32 guard elements, while fixed `06a2b24` overwrites zero;
- the fixed kernel passes the active-byte operand test and compute-sanitizer;
  those two checks alone cannot expose the old logical overrun because the CUDA
  caching allocator places it inside a larger allocation, which is why the
  explicit guard-canary test is retained;
- the formal 8K–45K balanced block-scale matrix passed after the upstream
  K-tail guard, including the lengths that previously produced illegal access.

## C1: routing and placement

The counting route-pack tests check pair conservation and byte equality for
qdata, active blocked scales, receive-token indices, expert IDs, and top-k
weights. The same gates run under contiguous placement, real greedy migration,
and experimental replica metadata. Migration additionally samples transferred
weight/scale storage bytes and verifies the physical stride contract.

## C2: local compute and reduce

FC1 Gate/Up ordering, SwiGLU, requantized post-activation values/scales, and FC2
output are checked against dequantized quantized operands. Atomic, segmented,
hybrid, direct-epilogue, workspace, and heavy-first experiments were tested;
deterministic segmented is the retained default.

## C3: communication

The NCCL wrapper runs a dispatch/combine identity oracle on every invocation.
The MegaMoE NVSHMEM/IBGDA reference passed a strict common-trace test with
token-back byte difference zero on all ranks. Its complete output also had
zero bad elements and zero maximum difference on each rank.

## C4: complete output

An element is bad only when:

```text
abs(actual - reference) > 0.05 + 0.05 * abs(reference)
```

The formal requirement is `bad_count=0` and relative L2 `<=5e-3`. Every
**admitted** Sonic and block-scale latency record satisfies this gate. Four
block-scale attempts do not: Segment-aligned/permuted/joint traces have 7-9 bad
elements, and the isolated single-hot trace has 9 bad elements. Those attempts
are status-only and contribute no latency. The completed Sonic unbalanced
matrix has maximum relative L2 below `2.7e-5`; the isolated single-hot replica
path also passes C4 in all three restarts.

MegaMoE performance records are admitted only after the separate strict
reduced-token/common-trace gate; the p95/p99 change is reporting-only. The 32K
and 45K signed-extent failures are recorded as unsupported without a latency.
The current targeted Sonic/QuACK suite has 23 passes and zero failures,
including transport-workspace and independent-hotspot controller tests.
