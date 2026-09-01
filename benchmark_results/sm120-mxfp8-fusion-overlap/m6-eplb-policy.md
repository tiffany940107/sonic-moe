# M6 Windowed EPLB and Replica Policy

The retained controller is communication-aware. It evaluates the exact source
token/expert trace and predicts deduplicated remote source-token/destination
records, not raw top-k pairs. Reconfiguration, routing-map version changes,
weight movement, and replica preload are control-plane costs and remain outside
E0 steady-state latency.

The default policy is:

- rank max/mean below 1.15: keep placement;
- 1.15–1.30: update the EMA and observe, without migration;
- above 1.30 for the configured persistence window: evaluate static EPLB;
- one expert above the independent hot-expert threshold, sustained for the
  persistence window: evaluate an explicit experimental replica
  recommendation even when rank max/mean is below 1.15; high rank skew without
  an isolated hot expert continues through the static-migration path;
- reject any action whose predicted break-even exceeds the configured
  amortization horizon.

For the supplied 16K/rank `rank max/mean=1.698` trace, real migration changed
384 experts and moved 3,114,270,720 bytes. E0 improved only from 42.831 to
41.333 ms (3.5%), while migration took 80.286 ms. The measured break-even was
53.59 steps. The exact communication model predicts remote records increasing
from 49,152 to 196,608, and a 32-step horizon correctly rejects this action.

For the Segment-M-aligned trace, rank max/mean was 2.356. Real migration
changed 378 experts and moved 3,065,610,240 bytes. E0 fell from 67.576 to
41.386 ms (38.8%); the 72.906 ms migration breaks even after 2.78 steps. The
controller predicts 2.80 steps and applies the placement. It explicitly
accepts the increase from 77,382 to 196,608 remote records because the compute
tail reduction is much larger.

The replica path remains experimental. On the broad Segment-M-aligned trace,
replicating its hot expert to three additional owners passed C4, but 2 and 8
reserved slots measured 62.511 and 62.524 ms with 371/279 ms preload cost.
That is better than the 67.576 ms contiguous path but much worse than the
41.386 ms static migration, so replica must not replace whole-expert EPLB in
that workload.

The isolated persistent-single-hot trace demonstrates the opposite case. Its
rank max/mean is only 1.088 while the hottest expert is 16.0x the expert mean.
Contiguous E0 is 33.900 ms. Whole-expert greedy placement merely moves the
bottleneck and regresses to 40.579 ms; one replica slot per rank splits expert
0 across all four owners and reaches 30.445 ms (1.113x), with C4 pass. Median
preload is 359.42 ms, so the measured break-even is 104.0 stable steps. The
controller therefore treats rank skew and isolated-hot-expert skew as
independent signals: after the persistence gate, a 64-step horizon rejects the
replica and a 160-step horizon recommends it. It never silently applies the
experimental action or increments the stable placement route version.

The later three-restart formal control confirms why the controller must not
equate every rank ratio with a migration. On PRO5000-B, greedy placement
regressed rank-ratio 1.106 from 28.899 to 41.209 ms and rank-ratio 1.227 from
32.025 to 41.374 ms. Rank-ratio 1.698 improved only to 41.380 ms and needed
about 41.5 stable steps to repay the formal median migration cost. In contrast,
Segment-M-aligned traces improved from 33.971 to 20.601 ms at 8K, 67.538 to
41.446 ms at 16K, 134.621 to 82.631 ms at 32K, and 189.102 to 116.049 ms at
45K. Their measured break-even ranges from about 6.0 steps at 8K to 1.1 steps
at 45K. This is the empirical basis for the observe/apply thresholds and the
communication-aware cost gate.
