#!/usr/bin/env bash
set -euo pipefail

# Collect the requested SM120 MXFP8 dense and grouped workload suites.  The
# Python benchmarks perform their own cold compile, correctness check, warmup,
# repeated event timing, and environment capture.

result_dir=${1:-results/mxfp8-customer-suite}
python_bin=${PYTHON_BIN:-python}
node_label=${NODE_LABEL:-$(hostname -s)}
warmup=${WARMUP:-20}
iterations=${ITERATIONS:-100}
repeats=${REPEATS:-3}
groups=${GROUPS:-8}

mkdir -p "${result_dir}"

"${python_bin}" benchmarks/mxfp8-dense-gemm.py \
  --workloads all \
  --warmup "${warmup}" \
  --iterations "${iterations}" \
  --repeats "${repeats}" \
  --node-label "${node_label}" \
  --output "${result_dir}/dense.json"

"${python_bin}" benchmarks/mxfp8-dense-vs-grouped.py \
  --workloads all \
  --groups "${groups}" \
  --distribution balanced \
  --warmup "${warmup}" \
  --iterations "${iterations}" \
  --repeats "${repeats}" \
  --node-label "${node_label}" \
  --output "${result_dir}/grouped-vs-dense.json"
