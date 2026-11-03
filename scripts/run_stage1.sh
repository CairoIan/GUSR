#!/usr/bin/env bash
set -euo pipefail

# Stage 1: multimodal self-supervised pretraining.
# Usage: bash scripts/run_stage1.sh CITY MERGED_JSON POI_CSV OSM_CSV OUTPUT_DIR [DEVICE]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CITY=${1:-zhengzhou}
MERGED_JSON=${2:-$ROOT/Dataset/cities/$CITY/merged_by_grid.json}
POI_CSV=${3:-$ROOT/Dataset/cities/$CITY/stage1/grid_poi_records.csv}
OSM_CSV=${4:-$ROOT/Dataset/cities/$CITY/stage1/grid_osm_features.csv}
OUTPUT_DIR=${5:-$ROOT/outputs/stage1_$CITY}
DEVICE=${6:-cuda:0}
PYTHON_BIN=${PYTHON_BIN:-python}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-$ROOT/Dataset/models/bert-base-uncased}
export TEXT_ENCODER_PATH
STAGE1_CONFIG=${STAGE1_CONFIG:-$ROOT/configs/stage1_paper.json}

cd "$ROOT"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" -m Model.train_self_supervised \
  --config "$STAGE1_CONFIG" \
  --device "$DEVICE" \
  --city "$CITY" \
  --merged_json "$MERGED_JSON" \
  --poi_csv "$POI_CSV" \
  --osm_csv "$OSM_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --sat_category_path "$ROOT/Model/priors/sat_cat.json" \
  --street_category_path "$ROOT/Model/priors/street_cat.json" \
  --text_encoder_path "$TEXT_ENCODER_PATH" \
  --text_contrast_weight 0.2 \
  --same_grid_contrast_weight 0.5
