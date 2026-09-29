#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grounding_brep.serve --config configs/brepground-inference.yaml \
  --checkpoint "${GROUNDING_CHECKPOINT:-weights/brepground/model.safetensors}" "$@"
