#!/usr/bin/env bash
set -euo pipefail

# Stage 2: frozen Stage 1 encoders + graph predictor.
# Usage: bash scripts/run_stage2.sh CITY MERGED_JSON OSM_PATH TARGET_PATH SPLIT_JSON SSL_CKPT METRICS_CSV [DEVICE] [OSM_MODE]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CITY=${1:-zhengzhou}
MERGED_JSON=${2:-$ROOT/Dataset/cities/$CITY/merged_by_grid.json}
OSM_PATH=${3:-$ROOT/Dataset/cities/$CITY/map.osm}
TARGET_PATH=${4:-$ROOT/gt_annotations/$CITY/gt_targets.csv}
SPLIT_JSON=${5:-$ROOT/splits/$CITY.json}
SSL_CKPT=${6:-$ROOT/Dataset/checkpoints/zhengzhou/stage1/ssl_step_010000.pt}
METRICS_PATH=${7:-${METRICS_PATH:-$ROOT/outputs/stage2_$CITY/stage2_metrics.csv}}
DEVICE=${8:-cuda:0}
OSM_MODE=${9:-${OSM_MODE:-nonroad}}
PYTHON_BIN=${PYTHON_BIN:-python}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-$ROOT/Dataset/models/bert-base-uncased}
export TEXT_ENCODER_PATH
EPOCHS=${EPOCHS:-70}
RUN_NAME=${RUN_NAME:-$(basename "$(dirname "$METRICS_PATH")")}
OSM_CACHE_PATH=${OSM_CACHE_PATH:-$(dirname "$METRICS_PATH")/osm_features_${OSM_MODE}.csv}
case "$CITY" in
  zhengzhou) TARGETS="GDP_2023_grid,Employment_2023_grid,Wage_2023_grid,Retail_2023_grid,Construction_area_grid,Finished_area_grid,Finished_value_grid,Total population" ;;
  beijing) TARGETS="GDP_2023_grid,Employment_2023_grid,Wage_2023_grid,Retail_2023_grid,Total population" ;;
  changsha|wuhan) TARGETS="GDP_2023_grid,Retail_2023_grid,Total population" ;;
  shanghai) TARGETS="GDP_2023_grid,Total population" ;;
  xian) TARGETS="GDP_2023_grid,Retail_2023_grid,Construction_area_grid,Finished_area_grid,Finished_value_grid,Total population" ;;
  *) printf 'Unsupported city: %s\n' "$CITY" >&2; exit 2 ;;
esac
DETERMINISTIC=${DETERMINISTIC:-1}
DETERMINISTIC_ARGS=()
if [[ "$DETERMINISTIC" == 1 ]]; then
  DETERMINISTIC_ARGS+=(--deterministic)
fi

cd "$ROOT"
mkdir -p "$(dirname "$METRICS_PATH")"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON_BIN" -m Model.train_stage2_gnn \
  --ssl_ckpt "$SSL_CKPT" \
  --device "$DEVICE" \
  --hidden_dim 256 \
  --epochs "$EPOCHS" \
  --city "$CITY" \
  --merged_json "$MERGED_JSON" \
  --osm_path "$OSM_PATH" \
  --osm_mode "$OSM_MODE" \
  --osm_cache_path "$OSM_CACHE_PATH" \
  --target_path "$TARGET_PATH" \
  --run_name "$RUN_NAME" \
  --split_path "$SPLIT_JSON" \
  --region_viz_path "$(dirname "$METRICS_PATH")/region_split.png" \
  --metrics_path "$METRICS_PATH" \
  --modal_mode both \
  --targets "$TARGETS" \
  --sat_func_prior "$ROOT/Model/priors/sat_func_prior_bigcat.json" \
  --street_func_prior "$ROOT/Model/priors/street_func_prior_bigcat.json" \
  --sat_category_path "$ROOT/Model/priors/sat_cat.json" \
  --street_category_path "$ROOT/Model/priors/street_cat.json" \
  --func_temp 0.15 \
  --func_div_weight 5.0 \
  --func_div_target_var 0.003 \
  --func_topk 4 \
  --seed "${SEED:-192}" \
  --split_seed "${SPLIT_SEED:-42}" \
  "${DETERMINISTIC_ARGS[@]}" \
  --tokenizer_path "$TEXT_ENCODER_PATH" \
  --text_encoder_path "$TEXT_ENCODER_PATH"
