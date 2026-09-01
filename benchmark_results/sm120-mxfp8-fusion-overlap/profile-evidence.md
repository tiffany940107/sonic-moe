# M0 Profile Evidence

The frozen 8K/rank EP4 trace showed that the original local weighted-reduce
chain was the largest removable non-communication cost. In the baseline NSYS
capture, FP32 `index_add_` consumed 17.3% of GPU kernel time (5.397 ms average)
and the separate weight multiply consumed another 12.4%. FC1 and FC2 averaged
6.106 ms and 3.114 ms, respectively.

After route-pack and deterministic segmented weighted-reduce were enabled,
the reduce kernel averaged 1.054 ms. FC1/FC2 stayed effectively unchanged at
6.104/3.122 ms, so the gain is attributable to removing the materialized pair
multiply/copy/index-add chain rather than changing GEMM semantics. NCCL's
relative share rose from 22.9% to 36.2%, making communication the next visible
bottleneck.

A second launch-level NSYS audit used three timed iterations on four ranks.
The legacy path issued 79 kernel launches and 11 CUDA memsets per rank/step;
the first fused path issued 32 launches and zero memsets; caller-owned EP
transport buffers and the bounded prefix kernel reduced this again to 21
launches and zero memsets. The profiler recorded no driver allocation/free
calls in any timed interval. The formal E0 records independently report zero
caching-allocator allocation/free/segment deltas for the final path.

NCU measured the segmented reduce at 71.72% of sustained peak DRAM throughput,
83.57% SM throughput, 78.92% active warps, and 42 registers/thread. Its
instrumented 1.351 ms duration must not be mixed with ordinary benchmark
latency.

Raw profiler reports are intentionally not published: they contain host,
device, stream, and process identifiers. The privacy-safe aggregate metrics
and SHA-256 identities are stored in `profile-summary.json` and the three
`nsys-launch-*.json` files in this directory.
