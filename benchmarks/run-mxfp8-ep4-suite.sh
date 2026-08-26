#!/usr/bin/env bash
set -euo pipefail

# Run the public Sonic-MXFP8 EP4 customer case on the four GPUs already made
# visible by the caller. This harness never discovers or records physical GPU
# indices, UUIDs, PCI addresses, hostnames, or IP addresses.

result_dir=${1:-results/mxfp8-ep4}
python_bin=${PYTHON_BIN:-python}
node_label=${NODE_LABEL:-anonymous}
topology_label=${TOPOLOGY_LABEL:-unspecified}
warmup=${WARMUP:-20}
iterations=${ITERATIONS:-100}
restarts=${RESTARTS:-3}

if [[ ! "${node_label}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "NODE_LABEL must be a privacy-safe public alias" >&2
  exit 2
fi
case "${topology_label}" in
  same_numa|cross_numa_2plus2|unspecified) ;;
  *) echo "TOPOLOGY_LABEL must be same_numa, cross_numa_2plus2, or unspecified" >&2; exit 2 ;;
esac
if [[ -z "${QUACK_REPO:-}" ]]; then
  echo "QUACK_REPO is required for reportable commit provenance" >&2
  exit 2
fi
if [[ -e "${result_dir}" ]]; then
  echo "result directory already exists; choose a new path: ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}"
sonic_commit=$(git rev-parse HEAD)
quack_commit=$(git -C "${QUACK_REPO}" rev-parse HEAD)
export SONIC_COMMIT="${sonic_commit}"
export QUACK_COMMIT="${quack_commit}"

run_case() {
  local case_name=$1
  local routing=$2
  local placement=$3
  local restart
  for ((restart = 0; restart < restarts; restart++)); do
    "${python_bin}" -m torch.distributed.run \
      --standalone \
      --nproc-per-node=4 \
      benchmarks/mxfp8-ep4-e2e.py \
      --tokens 4096 \
      --top-k 24 \
      --experts 768 \
      --hidden 2560 \
      --intermediate 1024 \
      --routing "${routing}" \
      --zipf-alpha 1.2 \
      --placement "${placement}" \
      --activation-transport mxfp8 \
      --warmup "${warmup}" \
      --iters "${iterations}" \
      --node-label "${node_label}" \
      --topology-label "${topology_label}" \
      --run-label "${case_name}_restart${restart}" \
      --output "${result_dir}/${case_name}.jsonl"
  done
}

run_case balanced uniform contiguous
run_case zipf-contiguous zipf contiguous
run_case zipf-greedy zipf greedy

"${python_bin}" benchmarks/summarize-mxfp8-ep4-suite.py \
  --result-dir "${result_dir}" \
  --output "${result_dir}/SUMMARY.md"
