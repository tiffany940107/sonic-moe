# 0902 Sonic SM120 MXFP8 Optimization Final Report

## Outcome

The work is isolated on Sonic `feature/sm120-mxfp8-fusion-overlap` at source
commit `31ab794` and QuACK `feature/sm120-mxfp8-fused-route-reduce` at
`d38dd36`. The immutable comparison commits remain Sonic `013f1726` and QuACK
`c87c9d1`; neither baseline branch was overwritten.

The retained Sonic path uses caller-owned local and EP transport workspaces,
bounded-expert route metadata, direct blocked-E8M0 scale scatter, and
deterministic segmented FP32 weighted reduce. It preserves exact top-k routing
and MXFP8 K32 semantics. Formal timed forwards have zero caching-allocator
allocation/free/segment deltas.

## Balanced E0 result

Values are median-of-three restart p50 milliseconds for prequantized EP4
dispatch through weighted combine. MegaMoE p95/p99 was added by a reporting-
only patch; it does not change its kernel.

| node | m/rank | legacy Sonic | final Sonic | block-scale | MegaMoE |
|---|---:|---:|---:|---:|---:|
| PRO5000-A | 8K | 23.269 | 13.051 | 26.082 | 43.301 |
| PRO5000-A | 16K | 46.649 | 25.904 | 51.938 | 85.931 |
| PRO5000-A | 32K | 104.111 | 51.878 | 109.399 | unsupported |
| PRO5000-A | 45K | 147.171 | 73.035 | 156.008 | unsupported |
| PRO5000-B | 8K | 25.353 | 13.908 | 29.003 | 43.458 |
| PRO5000-B | 16K | 50.680 | 27.721 | 57.545 | 86.434 |
| PRO5000-B | 32K | 114.697 | 55.380 | 118.978 | unsupported |
| PRO5000-B | 45K | 161.753 | 77.819 | 168.352 | unsupported |

Final Sonic is 1.78-2.08x faster than the legacy Sonic data path across these
eight cells and has no formal regression versus the prior fused path. The last
EP transport-workspace cleanup contributes only 0.31% geometric p50 by itself;
its main value is deterministic zero-allocation execution. MegaMoE 32K/45K is
rejected before timing by signed-INT32 output-view extents, so those cells are
explicitly unsupported rather than zero or estimated latency. Full p50/p95/p99
tables are in `FORMAL_RESULTS.md`.

## Unbalanced result and controller policy

The six supplied 16K common traces have three-restart Sonic, MegaMoE, and
where correct, block-scale controls. Sonic prior-fused p50 spans 28.899-67.538
ms on PRO5000-B; MegaMoE spans 90.089-148.605 ms on the same trace hashes.
Block-scale passes the rank-ratio 1.106/1.227/1.698 traces, but its Segment-
aligned/permuted/joint attempts have 9/8/7 C4 bad elements and are excluded.

Forced whole-expert placement is intentionally selective:

- rank 1.106: 28.899 -> 41.209 ms, regression;
- rank 1.227: 32.025 -> 41.374 ms, regression;
- rank 1.698: 43.476 -> 41.380 ms, about 41.5-step break-even;
- Segment-aligned: 67.538 -> 41.446 ms, 3.36-step break-even;
- Segment-permuted: 37.049 -> 41.424 ms, regression;
- joint skew: 46.897 -> 38.451 ms, 8.55-step break-even.

The retained windowed controller therefore combines persistence, rank skew,
deduplicated remote-record cost, and a configured amortization horizon. Rank
skew and isolated-hot-expert skew are separate signals.

On the new persistent-single-hot trace, rank max/mean is only 1.088 but expert
0 is 16.0x the expert mean. Sonic contiguous is 33.900 ms; whole-expert greedy
only moves the bottleneck and regresses to 40.579 ms. One replica slot per rank
splits the hot expert across all four owners and reaches 30.445 ms (1.113x),
with C4 pass. Median preload is 359.42 ms, so the measured break-even is 104.0
stable steps. A 64-step controller horizon rejects it; a 160-step horizon
recommends it after persistence. The action remains explicit/experimental and
is never silently applied. MegaMoE measures 89.387 ms on this trace;
block-scale has 9 bad elements and no admitted latency.

## M0-M6 gated decisions

- Formal K0/K1 completed three restarts on both nodes. K0 eager/graph p50 is
  10.562/10.560 ms on A and 10.627/10.634 ms on B, so CUDA Graph stays off.
  K1 is a true enclosing local route/MLP/reduce interval at 12.793/12.885 ms
  on A/B, exact C4, with zero timed allocator activity.
- NSYS launch count falls from 79 for legacy to 32 for the first fused path and
  21 per rank/step with EP transport workspace; memsets fall from 11 to zero.
- Route-pack remains as M3 infrastructure, not a standalone win: it misses its
  independent gate but provides byte-exact qdata, active scale, token, expert,
  and top-k metadata for the retained reduce path.
- Deterministic segmented weighted reduce cuts the 8K pilot local stage from
  12.935 to 1.053 ms. Direct FC2 atomic epilogue is correct but slower and off.
- M4 stopped at the pre-port gate. Upstream same-shape single-launch FC12 is
  42.954 ms versus Sonic separate FC1+FC2 at about 9.304 ms; even the optimistic
  tile-resident lower bound is 15.447 ms. No Sonic FC12 implementation is
  claimed.
- The load-aware pipeline passes C4 but improves its target diagnostic only
  2.38%, below the 5% gate. MegaMoE IBGDA passes strict byte/full-output checks
  but measures 61.339 ms versus Sonic NCCL 7.088 ms on the common 4K trace.
  Both remain experimental/off.
- Static migration is retained as a control-plane action. Replica/hybrid code
  remains opt-in and is selected only when a persistent isolated hot expert,
  measured saving, and amortization horizon all justify it.

## Correctness and boundary coverage

The final targeted Sonic/QuACK suite has 23 passes and zero failures. Segment-M
coverage includes an empty expert and active rows 1/127/128/129/517/2528/16384.
Every admitted Sonic and block-scale formal record has C4 `bad_count=0`; invalid
attempts never contribute latency. The block-scale K=1280 old/fixed canary A/B
finds 1,804 overwritten int32 elements at old `0da85e6` and zero at fixed
`06a2b24`; the fixed build also passes active-byte checks and
compute-sanitizer. MegaMoE formal latency uses the separately recorded strict
common-trace gate.

## Cross-NUMA status

Formal symmetric 2+2 cross-NUMA remains externally resource-blocked. On both
shared nodes, an unrelated workload occupies the second NUMA GPU group and
leaves too little memory for even the 8K expert bank. No external process was
terminated, no same-NUMA value was substituted, and failed-attempt timing is
not reported. The topology-safe scripts can rerun unchanged once that memory
is released. The operational campaign uses a resumable full-matrix runner; the
public `run-cross-numa-when-ready.sh` applies the same capacity rule and requires
60,000 MiB free on both selected devices in each NUMA group; the latest gate
returned 75 after observing only 4,083 MiB on A and 1,129 MiB on B, without
launching a benchmark. Earlier M5 cross-NUMA pipeline numbers are labelled
diagnostic, not formal matrix results.

## Artifacts

- full tables and machine data: `formal-results.md` and `summary.json`;
- fail-closed audit: `completion-audit.md` and `completion-audit.json`;
- profile evidence: `profile-evidence.md` and `profile-summary.json`;
- phase decisions: `m1-m3-data-path-decisions.md`,
  `m4-fc12-negative-results.md`, `m5-communication-decision.md`, and
  `m6-eplb-policy.md`;
- correctness and timing: `correctness-audit.md` and `timing-levels.md`;
- exact sources/environment: `sources.json` and `environment.json`;
- reproducible entry points: `run-formal-ep4.sh`, `run-local-k0-k1.sh`,
  `run-cross-numa-when-ready.sh`, `parse-megamoe-log.py`, and
  `audit-summary.py`.
