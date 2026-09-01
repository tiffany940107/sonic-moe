#!/usr/bin/env bash
set -euo pipefail

# Run the 512-expert Sonic MXFP8 EP4 + static-EPLB campaign on four GPUs that
# were made visible by the caller.  This script never discovers or records
# physical GPU IDs, PCI addresses, hostnames, or network addresses.

result_dir=${1:-results/mxfp8-eplb}
python_bin=${PYTHON_BIN:-python}
node_label=${NODE_LABEL:-anonymous}
topology_label=${TOPOLOGY_LABEL:-unspecified}
warmup=${WARMUP:-20}
iterations=${ITERATIONS:-100}
restarts=${RESTARTS:-3}
trace_dir=${TRACE_DIR:-${result_dir}/traces}
migration_limit=${MIGRATION_LIMIT:-0}

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
if [[ ! -f "${trace_dir}/trace_manifest.json" ]]; then
  "${python_bin}" benchmarks/generate-mxfp8-eplb-workloads.py \
    --output-dir "${trace_dir}"
fi

export SONIC_COMMIT
export QUACK_COMMIT
SONIC_COMMIT=$(git rev-parse HEAD)
QUACK_COMMIT=$(git -C "${QUACK_REPO}" rev-parse HEAD)

run_case() {
  local case_name=$1
  local tokens=$2
  local scenario=$3
  local placement=$4
  local migration_args=()
  local restart
  if [[ "${placement}" != contiguous ]]; then
    migration_args+=(--real-weight-migration --migration-limit "${migration_limit}")
  fi
  for ((restart = 0; restart < restarts; restart++)); do
    "${python_bin}" -m torch.distributed.run \
      --standalone \
      --nproc-per-node=4 \
      benchmarks/mxfp8-ep4-e2e.py \
      --tokens "${tokens}" \
      --top-k 32 \
      --experts 512 \
      --hidden 2048 \
      --intermediate 1280 \
      --routing-trace "${trace_dir}/m${tokens}_${scenario}.pt" \
      --placement "${placement}" \
      --activation-transport mxfp8 \
      --warmup "${warmup}" \
      --iters "${iterations}" \
      --node-label "${node_label}" \
      --topology-label "${topology_label}" \
      --run-label "${case_name}_restart${restart}" \
      --output "${result_dir}/${case_name}.jsonl" \
      "${migration_args[@]}"
  done
}

# All requested token lengths, plus the supplied p90 rank-skew operating point.
for tokens in 8192 16384 32768 46080; do
  run_case "m${tokens}_balanced_contiguous" "${tokens}" balanced contiguous
  run_case "m${tokens}_p90_contiguous" "${tokens}" rank_r1.227 contiguous
  run_case "m${tokens}_p90_greedy" "${tokens}" rank_r1.227 greedy
done

# Detailed 16K threshold/long-tail and segment-skew cases.
run_case m16384_p50_contiguous 16384 rank_r1.106 contiguous
run_case m16384_max_contiguous 16384 rank_r1.698 contiguous
run_case m16384_max_greedy 16384 rank_r1.698 greedy
run_case m16384_joint_contiguous 16384 joint_aligned_max contiguous
run_case m16384_joint_greedy 16384 joint_aligned_max greedy
run_case m16384_segment_contiguous 16384 segment_aligned contiguous
run_case m16384_segment_greedy 16384 segment_aligned greedy

echo "Raw JSONL results are in ${result_dir}."
echo "Replica/hybrid is experimental and is intentionally not run by this stable suite."
