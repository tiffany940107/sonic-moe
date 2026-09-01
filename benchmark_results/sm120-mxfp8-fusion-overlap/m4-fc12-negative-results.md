# M4/M4E FC12 Decision

M4 and M4E reached the plan's pre-port stop gate and were not implemented in
the default Sonic path. The evidence below is an upstream exact-shape probe
plus a Sonic tile-resident feasibility bound; it is not a claim that a Sonic
FC12 kernel was built.

For the local balanced shape `rows=262144`, `E=128`, `H=2048`, and physical
Gate+Up width 2560, the latest MegaMoE single-launch FC12 static scheduler
measured 42.954 ms p50. The separate Sonic FC1+FC2 path measured approximately
9.304 ms. The upstream atomic scheduler also produced an unspecified launch
failure at a much smaller probe, so it has no valid latency. The static result
is performance-only and is not used as a correctness-qualified ranking point.

The tile-resident feasibility model then tested whether global intermediate
storage could be removed. A 64-token tile has 32 FC2 output tiles, while the
target cluster limit is 16 CTAs. Even an optimistic multicast model therefore
needs at least two waves and retains about 84,480 bytes of qdata/scales per
token tile. Its optimistic lower bound is 15.447 ms; without cross-CTA reuse,
the FC1 recomputation bound is 199.737 ms. Both exceed the measured separate
path before accounting for register/SMEM pressure.

Decision: the measured reference and even the optimistic bound fail the 10%
K0/5% E0 entry gate before a port would be justified. Preserve the safe
separate FC1/requant/FC2 kernels and fused weighted-reduce epilogue. No FC12
experiment changes the default runtime.
