# Six-city GT supervision labels

Each city directory contains `gt_targets.csv`: one row per final grid, with `city`, `grid_id` and the available grid-level socioeconomic proxy labels. The labels were exported from the city-specific corrected road-allocation target layer used by the model. Geometry, county names, road lengths, road weights, image paths, GUSR records and POI/OSM features are excluded.

The target columns, including `Total population`, are read directly by the training code.

`manifest.json` lists row counts and fields for each city. The fixed train/validation/test indices are in `../splits/`.

These files publish the supervision targets only. The GUSR records and OSM/POI inputs are available from the companion ModelScope dataset; the original imagery is prepared separately.
