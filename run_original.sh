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

export USE_STANDALONE_CUTLASS_CONV=0
export USE_TORCH_COMPILE=0
cd "${root_dir}/pytorch_original"
python "${script}"
