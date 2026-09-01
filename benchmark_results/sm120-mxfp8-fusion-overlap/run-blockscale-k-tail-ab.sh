#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EMPTY_WORK_DIRECTORY" >&2
  exit 2
fi

work_root=$1
old_commit=0da85e635caaf727b7f6e27fa78afdcc2288903d
new_commit=06a2b245634381af8f297f7f415862a6e5bc3e99
image=${BLOCKSCALE_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -e ${work_root} ]]; then
  echo "work directory already exists: ${work_root}" >&2
  exit 2
fi
mkdir -p "${work_root}"
git clone https://github.com/CarstyYou/sm120_block_scale_gemm.git \
  "${work_root}/repo"
git -C "${work_root}/repo" worktree add --detach "${work_root}/old" "${old_commit}"
git -C "${work_root}/repo" worktree add --detach "${work_root}/new" "${new_commit}"

docker run --rm --gpus all \
  -v "${work_root}:/workspace/ab" \
  -v "${script_root}:/workspace/repro:ro" \
  -w /workspace/ab \
  "${image}" bash -euo pipefail -c '
    for spec in old:'"${old_commit}"' new:'"${new_commit}"'; do
      variant=${spec%%:*}
      commit=${spec#*:}
      repo=/workspace/ab/${variant}
      BUILD_JOBS=${BUILD_JOBS:-16} python "${repo}/build.py"
      nvcc -std=c++17 -O2 -arch=sm_120a \
        -I"${repo}/kernels/include" \
        -I"${repo}/3rdparty/cutlass/include" \
        /workspace/repro/blockscale-k-tail-guard.cu \
        -o "/workspace/ab/guard_${variant}"
      "/workspace/ab/guard_${variant}" "${variant}" "${commit}"
      python /workspace/repro/blockscale-k-tail-ab.py \
        --repo "${repo}" --source-commit "${commit}" \
        --variant "${variant}" --warmup 0 --iters 0
    done
    compute-sanitizer --error-exitcode=1 \
      python /workspace/repro/blockscale-k-tail-ab.py \
      --repo /workspace/ab/new --source-commit '"${new_commit}"' \
      --variant new --warmup 0 --iters 0
    python /workspace/repro/blockscale-k-tail-ab.py \
      --repo /workspace/ab/new --source-commit '"${new_commit}"' \
      --variant new --warmup 20 --iters 100
  '
