# K0/K1/E0/E1/R0/A0 Timing Levels

The timing levels are deliberately separate; setup/control costs are never
hidden inside steady-state E0.

| Level | Representative measurement | Scope |
|---|---:|---|
| K0 | A: 10.562/10.560 ms eager/graph | native physical layout, preallocated local FC1/requant/FC2 |
| K1 | A: 12.793 ms; B: 12.885 ms | one enclosing interval: row-linear route-pack + local FC1/requant/FC2 + segmented reduce; no EP collectives |
| E0 | A: 13.051 ms | 8K/rank balanced, prequantized EP4 dispatch through combine, caller-owned transport workspace |
| E1 | 13.079 ms | E0 plus source BF16-to-MXFP8 quantization in the timed forward |
| R0 | trace-dependent, about 72-88 ms formal | one planning/migration transaction, separately timed |
| A0 | `optimized E0 + R0 / stable_steps` | amortized latency over a stable routing window |

K0 and K1 each use 20 warmups, 100 timed iterations, and three independent
process restarts. K1 is now a true enclosing CUDA interval, replacing the old
diagnostic sum, and its C4 check is exact for the constructed reference. K1
still excludes EP dispatch/combine, so it must not be added to E0. Its timed
allocator allocation/free/segment deltas are zero on both nodes. E0 and E1
were measured on adjacent implementation revisions and must not be subtracted
as a source-quantization estimate; the earlier 0.5% difference was within
run-to-run/system variation.

For the formal 16K Segment-aligned case, E0 changes from 67.538 to 41.446 ms
and median R0 is 87.75 ms, yielding a 3.36-step break-even. At four stable
steps A0 is about 63.38 ms; at 16 steps it is about 46.93 ms. For rank-ratio
1.698, E0 changes only from 43.476 to 41.380 ms with 86.98 ms R0, so break-even
is about 41.5 steps and a 32-step policy correctly rejects migration.
