#!/usr/bin/env bash
set -euo pipefail

# The caller externally selects one SM120 GPU. This script records only the
# public node alias and never discovers a device ordinal, UUID, or PCI address.
if [[ $# -ne 2 ]]; then
  echo "usage: $0 RESULT_DIR PRO5000-A|PRO5000-B|anonymous" >&2
  exit 2
fi
result_dir=$1
node_label=$2
case ${node_label} in PRO5000-A|PRO5000-B|anonymous) ;; *) exit 2 ;; esac
: "${QUACK_REPO:?set QUACK_REPO to the matching QuACK checkout}"
if [[ -e ${result_dir} ]]; then
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}/k0" "${result_dir}/k1"

export SONIC_COMMIT QUACK_COMMIT PYTHONPATH
SONIC_COMMIT=$(git rev-parse HEAD)
QUACK_COMMIT=$(git -C "${QUACK_REPO}" rev-parse HEAD)
PYTHONPATH="$PWD:${QUACK_REPO}:$PWD/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
python_bin=${PYTHON_BIN:-python}

for restart in 1 2 3; do
  "${python_bin}" benchmarks/benchmark_mxfp8_workspace_graph.py \
    --rows 262144 --experts 128 --hidden 2048 --intermediate 1280 \
    --warmup 20 --iters 100 --node-label "${node_label}" \
    --run-label "formal_r${restart}" \
    --output "${result_dir}/k0/r${restart}.jsonl"
  "${python_bin}" benchmarks/benchmark_mxfp8_k1.py \
    --recv-tokens 32768 --top-k 32 --local-routes 8 \
    --experts 128 --hidden 2048 --intermediate 1280 \
    --warmup 20 --iters 100 --node-label "${node_label}" \
    --run-label "formal_r${restart}" \
    --output "${result_dir}/k1/r${restart}.jsonl"
done
