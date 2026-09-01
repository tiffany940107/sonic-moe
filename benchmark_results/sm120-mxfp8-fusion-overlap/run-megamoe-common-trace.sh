#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 RESULT_DIR NODE TOPOLOGY M TRACE p2p_direct|nvshmem_hybrid RESTART" >&2
  exit 2
fi
result_dir=$1
node_label=$2
topology=$3
tokens=$4
trace=$5
transport=$6
restart=$7
case ${node_label} in PRO5000-A|PRO5000-B) ;; *) exit 2 ;; esac
case ${topology} in same_numa|cross_numa_2plus2) ;; *) exit 2 ;; esac
case ${tokens} in 8192|16384|32768|46080) ;; *) exit 2 ;; esac
case ${transport} in p2p_direct|nvshmem_hybrid) ;; *) exit 2 ;; esac
case ${restart} in 1|2|3) ;; *) exit 2 ;; esac
: "${MEGAMOE_REPO:?set MEGAMOE_REPO to commit 8512aed522e14e4ee67e08cb54533ab7ec038a92}"

script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trace_dir=$(cd -- "$(dirname -- "${trace}")" && pwd)
trace_name=$(basename -- "${trace}")
if [[ ! ${trace_name} =~ ^m${tokens}_[A-Za-z0-9_.-]+\.pt$ ]]; then
  echo "trace basename must start with m${tokens}_ and end in .pt" >&2
  exit 2
fi
mkdir -p "${result_dir}"
result_dir=$(cd -- "${result_dir}" && pwd)

python_bin=${PYTHON_BIN:-python}
torchrun_bin=${TORCHRUN_BIN:-torchrun}
source_commit=${MEGAMOE_COMMIT:-8512aed522e14e4ee67e08cb54533ab7ec038a92}
log=${result_dir}/r${restart}.log
output=${result_dir}/r${restart}.jsonl
if [[ -e ${output} || -e ${log} ]]; then
  echo "restart output already exists: ${output} or ${log}" >&2
  exit 2
fi

(
  cd -- "${trace_dir}"
  PYTHONPATH="${MEGAMOE_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${torchrun_bin}" --standalone --nproc-per-node=4 \
    -m moe_sm120_mxfp8_split.mega_runner \
    --num_tokens_per_rank "${tokens}" --num_topk 32 \
    --num_total_experts 512 --hidden 2048 --intermediate 2560 \
    --route_distribution balanced --routing_trace "${trace_name}" \
    --comm_backend "${transport}" --perf_run --skip_ref_check \
    --use_cuda_events --perf_warmup 20 --perf_iters 100
) >"${log}" 2>&1

"${python_bin}" "${script_root}/parse-megamoe-log.py" \
  --log "${log}" --output "${output}" --node-label "${node_label}" \
  --topology "${topology}" --transport "${transport}" \
  --run-label "formal_r${restart}" --restart "${restart}" \
  --source-commit "${source_commit}" \
  --patch "${script_root}/megamoe-common-routing-trace.patch" \
  --reporting-patch "${script_root}/megamoe-p95-p99-reporting.patch" \
  --implementation-label p95_p99
