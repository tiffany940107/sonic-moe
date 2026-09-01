# M1-M3 Data-Path Decisions

## M1: workspace and CUDA Graph

The reusable local workspace and `_out` APIs are retained because they
establish stable ownership and are prerequisites for route-pack/reduce fusion.
Formal K0 has three process restarts on both nodes: PRO5000-A eager/graph p50
is 10.562/10.560 ms and PRO5000-B is 10.627/10.634 ms. The graph effect is
effectively zero and far below the 3% gate, so graph capture is supported but
disabled by default.

The later caller-owned EP transport workspace removes the remaining caching-
allocator activity: the timed forward changes from 17 allocation/free events
per rank/step to zero, and NSYS launches fall from 32 to 21 per rank/step.
Across the eight A/B balanced cells its incremental geometric p50 gain is only
0.31% (individual cells 0.15-0.75%), so it is not presented as an independent
latency optimization. It is kept as the deterministic-allocation/graph-ready
API and has no formal regression; the cumulative M1-M3 path still clears the
overall E0 gate by a wide margin.

## M2: route-pack

The bounded-expert histogram/prefix/scatter path produces expert-contiguous
qdata, blocked E8M0 scales, receive-token indices, expert IDs, and top-k
weights. Every active operand passed byte equality, including empty experts
and M/K tails. Standalone E0 was effectively flat (31.444 to 31.310 ms in the
8K pilot), so route-pack does not claim an independent performance win. It is
retained as the metadata/layout prerequisite for deterministic segmented
reduce. Heavy-expert-first ordering was also flat (67.576 to 67.543 ms on the
16K segment trace) and remains disabled.

## M3: weighted reduce

The default path is deterministic segmented FP32 weighted reduce. In the 8K
pilot, the original multiply/cast/`index_add_` stage measured 12.935 ms,
atomic scatter 2.794 ms, and segmented reduce 1.053 ms. E0 fell from 31.44 ms
to 19.49 ms in that pilot. The later full-workspace same-NUMA matrix shows
1.78-2.08x legacy-data-path speedup across 8K–45K/rank, with C4 pass and zero
timed allocator deltas in every formal record.

This retained implementation fuses weight multiplication and token reduction
into one kernel, but it still materializes the BF16 FC2 pair output. The QuACK
direct FC2 atomic epilogue experiment removed that write but raised FC2 from
about 3.15 to 5.60 ms and produced roughly 21.20 ms E0, so it is disabled.
Atomic/hybrid policy remains experimental; deterministic segmented is the
default because it is faster and avoids atomic-order variance on the customer
trace.
