#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MIN_FREE_MIB command..." >&2
  exit 2
fi
minimum_free_mib=$1
shift
if [[ ! ${minimum_free_mib} =~ ^[0-9]+$ ]] || (( minimum_free_mib < 24000 )); then
  echo "MIN_FREE_MIB must be an integer of at least 24000" >&2
  exit 2
fi

declare -A numa_gpus=()
while IFS=',' read -r ordinal pci_bus free_mib; do
  ordinal=${ordinal//[[:space:]]/}
  pci_bus=${pci_bus//[[:space:]]/}
  free_mib=${free_mib//[[:space:]]/}
  pci_bus=${pci_bus,,}
  pci_tail=${pci_bus#00000000:}
  sysfs_path=/sys/bus/pci/devices/0000:${pci_tail}/numa_node
  if [[ ! -r ${sysfs_path} ]]; then
    echo "cannot resolve GPU NUMA topology" >&2
    exit 3
  fi
  numa_node=$(<"${sysfs_path}")
  numa_gpus[${numa_node}]+="${free_mib}:${ordinal} "
done < <(nvidia-smi --query-gpu=index,pci.bus_id,memory.free --format=csv,noheader,nounits)

mapfile -t numa_nodes < <(printf '%s\n' "${!numa_gpus[@]}" | sort -n)
if (( ${#numa_nodes[@]} < 2 )); then
  echo "cross-NUMA requires two GPU NUMA groups" >&2
  exit 4
fi
selected=()
minimum_observed=2147483647
for numa_node in "${numa_nodes[@]:0:2}"; do
  mapfile -t candidates < <(
    printf '%s\n' ${numa_gpus[${numa_node}]} | sort -t: -k1,1nr | head -n 2
  )
  if (( ${#candidates[@]} < 2 )); then
    echo "a NUMA group has fewer than two GPUs" >&2
    exit 4
  fi
  second_free=${candidates[1]%%:*}
  if (( second_free < minimum_observed )); then minimum_observed=${second_free}; fi
  selected+=("${candidates[0]#*:}" "${candidates[1]#*:}")
done
if (( minimum_observed < minimum_free_mib )); then
  printf 'resource_not_ready minimum_top2_free_mib=%d required_mib=%d\n' \
    "${minimum_observed}" "${minimum_free_mib}"
  exit 75
fi

visible=$(IFS=,; printf '%s' "${selected[*]}")
export CUDA_VISIBLE_DEVICES=${visible}
exec "$@"
