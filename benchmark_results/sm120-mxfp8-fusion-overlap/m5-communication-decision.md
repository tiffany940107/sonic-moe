# M5 Communication and IBGDA Decision

The load-aware triple-buffer prototype passed C4, but it did not meet the
retention gate. On the cross-NUMA 8K/rank, rank-ratio 1.227 trace, the best
unchunked fused path measured 16.784 ms p50 and the pipeline measured 16.384
ms: a 2.38% gain. It improved over its own sequential-chunked control by 3.16%,
but the plan requires at least 3% overall and 5% on the target cross-NUMA case.
It also regressed the same-NUMA balanced pilot. The pipeline remains available
only as an experiment and is disabled by default.

This was an earlier diagnostic captured while a symmetric 2+2 allocation was
available. It is deliberately labelled non-formal and does not substitute for
the later full cross-NUMA matrix, which is externally resource-blocked.

The latest MegaMoE NVSHMEM/IBGDA path was then used as a correctness and design
reference. A common 128-token/rank trace passed token-back byte equality and
the full four-rank output check with zero bad elements and zero max difference.
Thus the earlier two-dimensional top-k-weight indexing bug is fixed upstream.

Performance did not justify porting the transport into Sonic. On the same
4K/rank common trace, MegaMoE hybrid measured 61.339 ms p50 while Sonic NCCL
measured 7.088 ms, an 8.65x latency ratio. MegaMoE's same-NUMA 8K p2p result was
also about 43.43 ms, showing that the gap is not explained only by cross-NUMA
traffic. The attempted 8K hybrid run was additionally invalidated by external
resource pressure and is not reported as latency.

Decision: retain deduplicated NCCL as the default Sonic transport. Do not port
NVSHMEM/IBGDA in this branch; revisit only if a future profile shows more than
10% exposed communication and a reference path beats the NCCL baseline under
the same trace and timing boundary.
