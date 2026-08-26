#!/usr/bin/env bash
set -euo pipefail

# Run the historical FlashInfer FP8 EP4 wrapper on the four GPUs already made
# visible by the caller. Physical GPU and host identifiers are never collected.

result_dir=${1:-results/flashinfer-fp8-ep4}
python_bin=${PYTHON_BIN:-python}
node_label=${NODE_LABEL:-anonymous}
topology_label=${TOPOLOGY_LABEL:-unspecified}
flashinfer_commit=${FLASHINFER_COMMIT:-}
restarts=${RESTARTS:-3}

if [[ ! "${node_label}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "NODE_LABEL must be a privacy-safe public alias" >&2
  exit 2
fi
case "${topology_label}" in
  same_numa|cross_numa_2plus2|unspecified) ;;
  *) echo "TOPOLOGY_LABEL must be same_numa, cross_numa_2plus2, or unspecified" >&2; exit 2 ;;
esac
if [[ -z "${flashinfer_commit}" ]]; then
  echo "FLASHINFER_COMMIT is required for reportable runs" >&2
  exit 2
fi
if [[ -e "${result_dir}" ]]; then
  echo "result directory already exists; choose a new path: ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}"

run_case() {
  local case_name=$1
  local routing=$2
  local placement=$3
  local warmup=$4
  local iterations=$5
  local restart
  for ((restart = 0; restart < restarts; restart++)); do
    "${python_bin}" -m torch.distributed.run \
      --standalone \
      --nproc-per-node=4 \
      benchmarks/flashinfer-fp8-ep4-e2e.py \
      --tokens 4096 \
      --top-k 24 \
      --experts 768 \
      --hidden 2560 \
      --intermediate 1024 \
      --routing "${routing}" \
      --zipf-alpha 1.2 \
      --placement "${placement}" \
      --activation-transport fp8 \
      --warmup "${warmup}" \
      --iters "${iterations}" \
      --seed 42 \
      --node-label "${node_label}" \
      --topology-label "${topology_label}" \
      --run-label "${case_name}_restart${restart}" \
      --output "${result_dir}/${case_name}.jsonl"
  done
}

# Historical balanced runs used the longer 50/200 protocol. Zipf/EPLB used
# 20/100. Each case still reports the median of three restarted process P50s.
run_case balanced uniform contiguous 50 200
run_case zipf-contiguous zipf contiguous 20 100
run_case zipf-greedy zipf greedy 20 100
