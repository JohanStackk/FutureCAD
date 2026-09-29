#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grounding_brep.evaluate \
  --config configs/brepground-inference.yaml \
  --checkpoint "${GROUNDING_CHECKPOINT:-weights/brepground/model.safetensors}" \
  --split "${SPLIT:-test}" --output "${OUTPUT:-outputs/grounding-test.json}" \
  data.split_path=data/splits/grounding.json \
  "data.dataset_path=${GROUNDING_DATA:-datasets/grounding_cache}" "$@"
