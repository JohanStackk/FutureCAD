#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GT_PLY_DIR:?Set GT_PLY_DIR to normalized reference PLYs}"
PREDICTIONS="${PREDICTIONS:-outputs/predictions.json}"
SPLIT_FILE="${SPLIT_FILE:-data/splits/futurecad.json}"
SPLIT="${SPLIT:-test}"
if [[ ! -f "$PREDICTIONS" ]]; then
  description_args=(--description-level "${DESCRIPTION_LEVEL:-detail}")
  if [[ -n "${DESCRIPTIONS:-}" ]]; then
    description_args+=(--descriptions "$DESCRIPTIONS")
  fi
  "${LLM_PYTHON:-python}" -m futurecad.generate --model "${LLM_MODEL:-weights/futurecad-llm}" \
    "${description_args[@]}" --split-file "$SPLIT_FILE" --split "$SPLIT" \
    --output "$PREDICTIONS" --n "${N_SAMPLES:-2}" --seed "${SEED:-42}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" --enforce-eager
fi
"${CAD_PYTHON:-python}" -m futurecad.evaluation --predictions "$PREDICTIONS" \
  --split-file "$SPLIT_FILE" --split "$SPLIT" --gt-ply-dir "$GT_PLY_DIR" \
  --gt-counts "${GT_COUNTS:-data/evaluation/futurecad_operation_counts.json}" \
  --advanced-ids "${ADVANCED_IDS:-data/splits/advanced_ids.json}" \
  --output "${OUTPUT:-outputs/llm-test.json}" --workers "${WORKERS:-4}" --seed "${SEED:-42}" "$@"
