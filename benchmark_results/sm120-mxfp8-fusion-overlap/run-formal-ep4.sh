#!/usr/bin/env bash
set -euo pipefail

# The caller selects four physical GPUs before invoking this script. Device
# ordinals and machine identifiers are intentionally neither discovered nor
# recorded here.
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 RESULT_DIR NODE_LABEL TOPOLOGY [balanced|unbalanced|eplb|all]" >&2
  exit 2
fi
result_dir=$1
node_label=$2
topology=$3
mode=${4:-all}
case ${node_label} in PRO5000-A|PRO5000-B|anonymous) ;; *) exit 2 ;; esac
case ${topology} in same_numa|cross_numa_2plus2|unspecified) ;; *) exit 2 ;; esac
case ${mode} in balanced|unbalanced|eplb|all) ;; *) exit 2 ;; esac
: "${QUACK_REPO:?set QUACK_REPO to the matching QuACK checkout}"

if [[ -e ${result_dir} ]]; then
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}/traces" "${result_dir}/raw"
python benchmarks/generate-mxfp8-eplb-workloads.py \
  --output-dir "${result_dir}/traces"

export SONIC_COMMIT QUACK_COMMIT
SONIC_COMMIT=$(git rev-parse HEAD)
QUACK_COMMIT=$(git -C "${QUACK_REPO}" rev-parse HEAD)
python_bin=${PYTHON_BIN:-python}

run_case() {
  local tokens=$1 scenario=$2 data_path=$3 placement=$4 restart=$5
  local replica_slots=${6:-0}
  local replica_suffix=
  if (( replica_slots > 0 )); then replica_suffix=_replica${replica_slots}; fi
  local variant=${data_path}_transport_workspace_${placement}${replica_suffix}
  local output=${result_dir}/raw/m${tokens}_${scenario}_${variant}_r${restart}.jsonl
  local migration_args=() replica_args=()
  if [[ ${placement} == greedy ]]; then migration_args+=(--real-weight-migration); fi
  if (( replica_slots > 0 )); then
    replica_args+=(--experimental-replica-slots "${replica_slots}")
  fi
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=4 \
    benchmarks/mxfp8-ep4-e2e.py \
    --tokens "${tokens}" --top-k 32 --experts 512 --hidden 2048 \
    --intermediate 1280 \
    --routing-trace "${result_dir}/traces/m${tokens}_${scenario}.pt" \
    --placement "${placement}" --activation-transport mxfp8 \
    --data-path "${data_path}" --prequantized-source \
    --implementation-label transport_workspace \
    --warmup 20 --iters 100 --atomic-variance-runs 5 \
    --node-label "${node_label}" --topology-label "${topology}" \
    --run-label "formal_r${restart}" --output "${output}" \
    "${migration_args[@]}" "${replica_args[@]}"
}

if [[ ${mode} == balanced || ${mode} == all ]]; then
  for data_path in baseline fused_segmented; do
    for tokens in 8192 16384 32768 46080; do
      for restart in 1 2 3; do
        run_case "${tokens}" balanced "${data_path}" contiguous "${restart}"
      done
    done
  done
fi

unbalanced_cases=(
  "8192 rank_r1.227" "8192 segment_aligned"
  "16384 rank_r1.106" "16384 rank_r1.227" "16384 rank_r1.698"
  "16384 segment_aligned" "16384 segment_permuted" "16384 joint_aligned_max"
  "16384 persistent_single_hot"
  "32768 rank_r1.227" "32768 segment_aligned"
  "46080 rank_r1.227" "46080 segment_aligned"
)
if [[ ${mode} == unbalanced || ${mode} == all ]]; then
  for spec in "${unbalanced_cases[@]}"; do
    read -r tokens scenario <<<"${spec}"
    for restart in 1 2 3; do
      run_case "${tokens}" "${scenario}" fused_segmented contiguous "${restart}"
    done
  done
fi
if [[ ${mode} == eplb || ${mode} == all ]]; then
  for spec in "${unbalanced_cases[@]}"; do
    read -r tokens scenario <<<"${spec}"
    for restart in 1 2 3; do
      run_case "${tokens}" "${scenario}" fused_segmented greedy "${restart}"
    done
  done
  for restart in 1 2 3; do
    run_case 16384 persistent_single_hot fused_segmented contiguous "${restart}" 1
  done
fi
