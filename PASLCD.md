# Reproducing the PASLCD Results

This branch (`paslcd`) contains the code and settings used for the PASLCD results
(Pose-Agnostic Scene-Level Change Detection, from [MV3DCD](https://chumsy0725.github.io/MV-3DCD/)).
The precomputed SAM masks we used are released separately, so the results can be
reproduced without regenerating masks. The pipeline is not fully deterministic on GPU, so
results can vary slightly between runs.

## Setup

Follow [Installation](README.md#installation) and [Download Checkpoints](README.md#download-checkpoints)
in the main README, then switch to this branch:

```bash
git checkout paslcd
git submodule update --init --recursive
```

Set the DINOv3 `weight_url` in `configs/paslcd_config.yml` as described in the main README.
It also accepts a local path to `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`.

## Data

1. Download PASLCD from the [MV3DCD project page](https://chumsy0725.github.io/MV-3DCD/) and place it
   at `data/PASLCD`, so that each scene instance is `data/PASLCD/<Scene>/<Instance>/` containing
   `images/` and `gt_mask/`. Reference views are the images without `test` in their file name;
   query views contain `test`.
2. Download the SAM masks (`PASLCD_masks.tar.gz`, ~50 MB) from **<MASKS_URL>** and extract them
   into the same folder:

   ```bash
   tar -xzf PASLCD_masks.tar.gz -C data/   # adds data/PASLCD/<Scene>/<Instance>/sam_masks.npz
   cd data/PASLCD && md5sum -c md5sums.txt && cd -
   ```

Each `sam_masks.npz` holds `file_names` (the frames used: every second reference view and all
query views) and `masks` (one int16 label map per frame, `-1` for unassigned pixels).
`modules/mask_model.py` loads it automatically from the scene directory.

## Run

```bash
for scene in Cantina Lounge Printing_area Lunch_room Meeting_room Garden Pots Zen Playground Porch; do
  for instance in Instance_1 Instance_2; do
    python scripts/predict_paslcd.py \
      --scene_dir data/PASLCD/$scene/$instance \
      --output output/paslcd/$scene/$instance
  done
done
```

`scripts/predict_paslcd.py` uses `configs/paslcd_config.yml` and keeps every second reference view
(`--resample_rate 2`) by default. Each instance takes about 3–7 minutes on an A40. The predictions
are per-frame binary change masks for the query views in
`output/paslcd/<Scene>/<Instance>/video_2_difference_masks/`. They include changed regions detected
in the query views and the projections of objects that were removed since the reference views.

## Evaluate

```bash
python scripts/evaluate_all_scenes_instances_iou.py \
  --pred_root output/paslcd \
  --data_root data/PASLCD \
  --output_dir output/paslcd_eval
```

For each instance, IoU is computed over all query frames together (intersections and unions summed
over frames). The scene score is the mean over its two instances, and the reported number is the mean
over the 10 scenes (`overall_scene_miou` in `scene_level_results.json`). The per-frame average
(`mean_frame_iou`) is also saved.

## Settings

All pipeline settings are in `configs/paslcd_config.yml`. Compared with the SceneDiff benchmark
configuration, the main differences are a processing size of 644 (`processing.max_size`),
400 voxels for scene normalization, `min_detection_pixel` 400, and a region-match occlusion
threshold of 0.6.

### SAM masks

We found mask generation to be sensitive to the scene type, and use different SAM settings for
indoor and outdoor scenes:

| | Indoor (Cantina, Lounge, Lunch Room, Meeting Room, Printing Area) | Outdoor (Garden, Playground, Porch, Pots, Zen) |
|---|---|---|
| Script | `scripts/generate_paslcd_masks_indoor.py` | `scripts/generate_paslcd_masks_outdoor.py` |
| SAM generator | single-level (`modules/sam_single_level_generator.py`) | two-level LangSplat (medium, large) |
| SAM input size | 2072×1176 | 1554×882 |
| `points_per_side` | 32 | 16 |
| `pred_iou_thresh` | 0.78 | 0.8 |
| `stability_score_thresh` | 0.85 | 0.9 |
| `box_nms_thresh` | 0.7 | 0.8 |
| `crop_n_layers` | 1 | 1 |
| `min_mask_region_area` | 0 | 100 |
| Painting order | small regions on top | medium level first, then large; largest regions first |
| Stored mask size | 2072×1176 | 518×294 |
| Small-region removal | < 50 px | < 50 px |

To regenerate the masks instead of downloading them (needs the SAM checkpoint):

```bash
python scripts/generate_paslcd_masks_indoor.py --scene_dir data/PASLCD/Cantina/Instance_1 \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth
python scripts/generate_paslcd_masks_outdoor.py --scene_dir data/PASLCD/Zen/Instance_1
```

The indoor script writes `sam_masks_additional.pkl` and the outdoor script `sam_masks.npz`; both are
picked up by `modules/mask_model.py`. If a scene directory contains `sam_masks.npz`, it takes
precedence over the other mask files.
