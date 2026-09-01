#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RESULT_DIR NODE TOPOLOGY M TRACE RESTART" >&2
  exit 2
fi
result_dir=$1
node_label=$2
topology=$3
tokens=$4
trace=$5
restart=$6
case ${node_label} in PRO5000-A|PRO5000-B) ;; *) exit 2 ;; esac
case ${topology} in same_numa|cross_numa_2plus2) ;; *) exit 2 ;; esac
case ${tokens} in 8192|16384|32768|46080) ;; *) exit 2 ;; esac
case ${restart} in 1|2|3) ;; *) exit 2 ;; esac
: "${BLOCKSCALE_REPO:?set BLOCKSCALE_REPO to commit 06a2b245634381af8f297f7f415862a6e5bc3e99}"
: "${BLOCKSCALE_LIB:?set BLOCKSCALE_LIB to the built libth_op.so}"

script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sonic_root=$(git rev-parse --show-toplevel)
trace_dir=$(cd -- "$(dirname -- "${trace}")" && pwd)
trace_name=$(basename -- "${trace}")
if [[ ! ${trace_name} =~ ^m${tokens}_[A-Za-z0-9_.-]+\.pt$ ]]; then
  echo "trace basename must start with m${tokens}_ and end in .pt" >&2
  exit 2
fi
mkdir -p "${result_dir}"
result_dir=$(cd -- "${result_dir}" && pwd)
output=${result_dir}/r${restart}.jsonl
log=${result_dir}/r${restart}.log
if [[ -e ${output} || -e ${log} ]]; then
  echo "restart output already exists: ${output} or ${log}" >&2
  exit 2
fi

torchrun_bin=${TORCHRUN_BIN:-torchrun}
source_commit=${BLOCKSCALE_COMMIT:-06a2b245634381af8f297f7f415862a6e5bc3e99}
(
  cd -- "${trace_dir}"
  PYTHONPATH="${sonic_root}/benchmarks:${script_root}:${BLOCKSCALE_REPO}/test${PYTHONPATH:+:${PYTHONPATH}}" \
  BLOCKSCALE_LIB="${BLOCKSCALE_LIB}" BLOCKSCALE_COMMIT="${source_commit}" \
    "${torchrun_bin}" --standalone --nproc-per-node=4 \
    "${script_root}/bench-blockscale-mxfp8-ep.py" \
    --tokens "${tokens}" --top-k 32 --experts 512 --hidden 2048 \
    --intermediate 1280 --routing-trace "${trace_name}" \
    --placement contiguous --source-mode prequantized \
    --activation-transport mxfp8 --warmup 20 --iters 100 \
    --node-label "${node_label}" --topology-label "${topology}" \
    --run-label "formal_r${restart}" \
    --output "${output}"
) 2>&1 | tee "${log}"
