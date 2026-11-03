# The uneven visual observability of urban socioeconomic conditions

This repository provides training code and data preparation instructions for *The uneven visual observability of urban socioeconomic conditions*. The paper introduces Grounded Urban Scene Representation (GUSR), which organizes descriptions, entities, relations and their locations in satellite and street-view imagery as visual evidence for analysis. A two-stage framework combines image representations, GUSR, OSM features and a spatial-functional graph to compare the recoverability of eight socioeconomic proxies across six Chinese cities, examine spatial differences in visual associations and assess cross-city transfer.

The repository contains training code, six-city target labels and fixed spatial splits. The companion dataset provides GUSR records, OSM inputs and selected weights. Original images are prepared as described below.

## Reported performance

Test-set $R^2$ reported in the paper. Zhengzhou is the mean of five runs; the other rows show transfer from Zhengzhou after supervised target-city adaptation. A dash means that the city has no corresponding target label.

| City / protocol | GDP | Employment | Wages | Retail | Construction area | Completed area | Completion value | Total population | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zhengzhou (5-seed mean) | 0.614 | 0.575 | 0.443 | 0.667 | 0.743 | 0.745 | 0.741 | 0.657 | 0.648 |
| Beijing transfer | 0.670 | 0.648 | 0.667 | 0.535 | — | — | — | 0.310 | 0.566 |
| Changsha transfer | 0.420 | — | — | 0.484 | — | — | — | 0.396 | 0.433 |
| Shanghai transfer | 0.075 | — | — | — | — | — | — | 0.272 | 0.174 |
| Wuhan transfer | 0.537 | — | — | 0.513 | — | — | — | 0.455 | 0.502 |
| Xi'an transfer | 0.634 | — | — | 0.613 | 0.560 | 0.500 | 0.514 | 0.454 | 0.546 |

## Environment

- Linux, Python 3.10 and a CUDA-capable GPU for the reported training workflow.
- Install the pinned training dependencies from `requirements-core.txt`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

## Data download and layout

**Dataset download:** The [GUSR dataset on ModelScope](https://www.modelscope.cn/datasets/CCairo/GUSR) provides seven checkpoints and six city archives containing the GUSR records, OSM inputs and training manifests. Original satellite and street-view images are prepared separately using the final image request list below.

```text
GUSR/
├── Model/                         # Stage 1/2 code and functional priors
├── scripts/                       # Training entry points
├── configs/stage1_paper.json
├── gt_annotations/<city>/gt_targets.csv
├── splits/<city>.json
└── Dataset/                       # Downloaded and prepared training inputs
    ├── cities/<city>/
    │   ├── merged_by_grid.json
    │   ├── map.osm
    │   ├── images/{sat,street}/
    │   └── rle/{sat,street}/
    ├── cities/zhengzhou/stage1/
    │   ├── grid_poi_records.csv
    │   └── grid_osm_features.csv
    ├── checkpoints/
    │   ├── zhengzhou/stage1/ssl_step_010000.pt
    │   ├── zhengzhou/stage2/best.pt
    │   └── <transfer-city>/stage2/best.pt
    └── models/bert-base-uncased/
```

`<city>` is one of `zhengzhou`, `beijing`, `changsha`, `shanghai`, `wuhan` or `xian`. Image and RLE paths inside each manifest are relative to that city's directory. `gt_annotations/` and `splits/` are already in this repository. Stage 1 uses the two Zhengzhou feature CSV files; Stage 2 uses each city's `map.osm`.

The seven published weights are Zhengzhou Stage 1 (step 10,000), Zhengzhou Stage 2 (seed 192, epoch 66), and Beijing, Changsha, Shanghai, Wuhan and Xi'an transfer Stage 2 (seed 42; epochs 65, 37, 58, 58 and 59, respectively). They are distributed with the data bundle because the Stage 1 weight alone is about 2 GB. The Stage 2 weights are optional when training Stage 2 from scratch.

**Download and place the non-image inputs.** Install the ModelScope downloader, then download the [dataset](https://www.modelscope.cn/datasets/CCairo/GUSR) into `Dataset/`. Each city archive unpacks its `merged_by_grid.json`, `rle/` GUSR masks and `map.osm` under `Dataset/cities/<city>/`; the Zhengzhou archive also contains the two Stage 1 feature CSV files. The `merged_by_grid.json` records contain the GUSR image descriptions and paths to the separate image and mask files. GUSR records are annotations of images, not the images themselves.

```bash
pip install modelscope==1.37.1
modelscope download --dataset CCairo/GUSR --local_dir Dataset
for archive in Dataset/training_inputs/*_training_inputs.tar.gz; do
  tar -xzf "$archive" -C Dataset
done
```

**Download BERT.** The frozen text encoder is [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased). Place its PyTorch weight, configuration and tokenizer files in `Dataset/models/bert-base-uncased/`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir="Dataset/models/bert-base-uncased",
    allow_patterns=["config.json", "pytorch_model.bin", "tokenizer.json",
                    "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"],
)
PY
```

The seven checkpoints arrive under `Dataset/checkpoints/` with the ModelScope download. Follow the image preparation section below to fill `Dataset/cities/<city>/images/` before starting training.

## Prepare the final images

`download/image_requests.csv` identifies the 28,248 satellite and 49,408 street-view images retained across the 9,764 final grids. Each row gives the city, grid ID, modality, WGS84 longitude (`lon`) and latitude (`lat`), and the exact repository-relative output filename (`image_path`). Street-view rows also give the selected direction (`view`) and any available historical-version metadata.

- **Satellite imagery:** Use the [Google imagery attribution](https://about.google/brand-resource-center/products-and-services/geo-guidelines/) to identify the imagery rights holder and request coverage for the six cities, required dates and resolution. Obtain the images through the rights holder's provided interface at the coordinates in the request list.
- **Street-view imagery:** Request panorama service and interface access through [Baidu Maps business contact](https://lbs.baidu.com/survey/lbscontact). After approval, [create a server-side AK](https://lbs.baidu.com/faq/details?id=2298&title=2639) and use the approved interface. The public [Panorama Static API](https://lbs.baidu.com/docs/webapi?title=viewstatic/viewstatic-base) accepts `heading` and `fov=90` to return a directional view directly.

The original five-city satellite downloads used names such as `115.873225_40.449811_z18.jpg`; original street views used names such as `115.872600,40.450500_east.jpg`. The prepared training manifests instead use numbered filenames. The request list maps each coordinate and view to its required filename. For example:

- Beijing satellite at `115.873225, 40.449811` → `Dataset/cities/beijing/images/sat/000000.jpg` (700 × 700 RGB).
- Beijing east-facing street view at `115.872600, 40.450500` → `Dataset/cities/beijing/images/street/000004.jpg` (720 × 720 RGB).

Save or rename every downloaded image to the **exact `image_path` in its request row**. Use that column as the naming rule for all six cities, including Zhengzhou; do not construct a training filename from the coordinates. GUSR descriptions and RLE masks are separate training inputs.

## Training

**Zhengzhou Stage 1.** The supplied Stage 1 checkpoint can be used directly for Stage 2. To train Stage 1 from the prepared data instead:

```bash
bash scripts/run_stage1.sh
```

**Zhengzhou Stage 2.** The default command runs seed 192 for 70 epochs with both image modalities and `nonroad` OSM features:

```bash
bash scripts/run_stage2.sh zhengzhou
```

**Five-city transfer.** Each city automatically selects its available targets and its matching manifest, OSM graph, GT CSV and split. Run the five cities with:

```bash
for city in beijing changsha shanghai wuhan xian; do
  SEED=42 DETERMINISTIC=0 OSM_MODE=nonroad \
    METRICS_PATH="$PWD/outputs/${city}_transfer/stage2_metrics.csv" \
    bash scripts/run_transfer_city.sh "$city"
done
```

## Release status

The GitHub repository contains training code, fixed splits, priors and the six GT tables. ModelScope provides the seven selected weights, GUSR records and OSM/POI inputs. The BERT encoder comes from Hugging Face; the original images are obtained separately as described above.

The code is released under the [MIT License](LICENSE).
