#!/usr/bin/env python3
"""Generate SAM masks for the PASLCD outdoor scenes (Garden, Playground, Porch, Pots, Zen).

Settings used for the released masks: two-level LangSplat SAM
(submodules/segment-anything-langsplat-modified) on images resized to 1554x882 (3x the
518x294 working size), 16 points per side, pred_iou 0.8, box NMS 0.8, stability 0.9,
one crop layer; medium level painted first, then large, largest regions first; masks
downsampled to 518x294 and regions under 50 pixels removed.
Writes <scene_dir>/sam_masks.npz, which modules/mask_model.py reads.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "submodules" / "segment-anything-langsplat-modified"))
sys.path.insert(0, str(REPO_ROOT))
from segment_anything_ls import SamAutomaticMaskGenerator, build_sam  # noqa: E402
from utils import masks_update, remove_small_masks, reorder_mask  # noqa: E402

H, W = 294, 518  # Working size the masks are stored at.
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SAM_CFG = dict(points_per_side=16, pred_iou_thresh=0.8, box_nms_thresh=0.8, stability_score_thresh=0.9,
               crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=100)


def frame_list(scene_dir, resample_rate):
    """Same selection as demo.py: every resample_rate-th train view + all test views."""
    images = sorted(p for p in (scene_dir / "images").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return [p for p in images if "test" not in p.name][::resample_rate] + [p for p in images if "test" in p.name]


def predict_mask(image, generator, min_size=100):
    """Non-overlapping regions: medium level first, then large, largest regions first."""
    masks_m, masks_l = generator.generate(image)
    masks_m, masks_l = masks_update(masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    group_ids = np.full(image.shape[:2], -1, dtype=int)
    counter = 0
    for level in (masks_m, masks_l):
        if (group_ids == -1).sum() < min_size:
            break
        for d in sorted(level, key=lambda x: x["area"], reverse=True):
            to_mask = d["segmentation"] & (group_ids == -1)
            if to_mask.sum() < min_size:
                continue
            group_ids[to_mask] = counter
            counter += 1
            if (group_ids == -1).sum() < min_size:
                break
    return group_ids


def main():
    parser = argparse.ArgumentParser(description="Generate SAM masks for PASLCD outdoor scenes.")
    parser.add_argument("--scene_dir", required=True, help="Scene instance directory containing images/.")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints/sam_vit_h_4b8939.pth"))
    parser.add_argument("--resample_rate", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    generator = SamAutomaticMaskGenerator(model=build_sam(checkpoint=args.checkpoint).to(args.device), **SAM_CFG)
    files = frame_list(scene_dir, args.resample_rate)
    masks = []
    for idx, path in enumerate(files):
        image = np.array(Image.open(path).resize((W * 3, H * 3))).astype(np.uint8)
        mask = predict_mask(image, generator)
        mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        masks.append(reorder_mask(remove_small_masks(mask, min_size=50)))
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(files)} images", flush=True)

    out_path = scene_dir / "sam_masks.npz"
    np.savez_compressed(out_path, file_names=np.array([p.name for p in files]),
                        masks=np.stack(masks).astype(np.int16))
    print(f"Saved {len(masks)} masks to {out_path}")


if __name__ == "__main__":
    main()
