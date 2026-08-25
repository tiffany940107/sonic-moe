# Benchmark harness audit

An early draft of `run-mxfp8-customer-suite.sh` used `GROUPS` as its configurable
expert-group variable. `GROUPS` is a special Bash array containing the current
process's Unix group IDs, so Bash expanded it to host-specific group IDs instead
of the requested value 8.

Impact was limited to the grouped portion of that initial wrapper invocation.
Standalone dense GEMM and runs that explicitly passed `--groups 8` were not
affected. The two PRO 5000 invalid grouped JSON files remain marked `INVALID`
on their shared experiment disk; the transient PRO 6000 invalid output was
excluded. None of these invalid artifacts is present in this GitHub result set.

Commit `a213dc0` changed the interface to `NUM_GROUPS` and local variable
`num_groups`. The final dense and grouped suites were both rerun at commit
`f033656`; every included grouped JSON document records `config.groups == 8`.
The summarizer also rejects any input whose group count is not exactly eight.

Published node names are anonymous aliases (`PRO5000-A`, `PRO5000-B`, and
`PRO6000`). The result files intentionally omit hostnames, IP addresses, GPU
UUIDs, PCI bus IDs, and visible-device indices. The benchmark harness likewise
does not collect those fields in new results.
