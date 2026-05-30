#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <resnet|mobilenet|shufflenet|yolo>" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="$1"

case "${model}" in
  resnet) script="eval_resnet20.py" ;;
  mobilenet) script="eval_mobilenet.py" ;;
  shufflenet) script="eval_shufflenet.py" ;;
  yolo) script="eval_yolo.py" ;;
  *)
    echo "Unknown model: ${model}" >&2
    exit 2
    ;;
esac

export USE_STANDALONE_CUTLASS_CONV=1
export USE_TORCH_COMPILE=0
export STANDALONE_CUTLASS_STRICT="${STANDALONE_CUTLASS_STRICT:-1}"
export CUTLASS_DIR="${CUTLASS_DIR:-/nethome/yjia305/USERSCRATCH/cutlass_test}"
export STANDALONE_CUTLASS_DIR="${STANDALONE_CUTLASS_DIR:-${CUTLASS_DIR}}"
export STANDALONE_CUTLASS_BUILD_DIR="${STANDALONE_CUTLASS_BUILD_DIR:-/tmp/ml_bench_cutlass_test_standalone_conv_build}"
export PYTHONPATH="${root_dir}/custom_conv_cutlass_test:${PYTHONPATH:-}"

cd "${root_dir}/custom_conv_cutlass_test"
python "${script}"
