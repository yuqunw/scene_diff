"""
Compute IoU between predicted video_2 difference masks and GT masks.

Expected prediction layout (per scene):
  <pred_scene_dir>/video_2_difference_masks/frame_00000.png

Expected GT layout:
  <scene_dir>/Instance_1/gt_mask/*.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def load_binary_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
    return mask > 0


def list_mask_files(mask_dir: Path):
    return sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_EXTS])


def compute_iou_for_scene(pred_mask_dir: Path, gt_mask_dir: Path):
    pred_files = list_mask_files(pred_mask_dir)
    gt_files = list_mask_files(gt_mask_dir)

    if len(pred_files) == 0:
        raise ValueError(f"No prediction masks found in: {pred_mask_dir}")
    if len(gt_files) == 0:
        raise ValueError(f"No GT masks found in: {gt_mask_dir}")

    if len(pred_files) != len(gt_files):
        raise ValueError(
            f"Prediction/GT mask count mismatch: {len(pred_files)} vs {len(gt_files)}"
        )

    total_intersection = 0
    total_union = 0
    frame_ious = []

    for idx, (pred_path, gt_path) in enumerate(zip(pred_files, gt_files)):
        pred_mask = load_binary_mask(pred_path)
        gt_mask = load_binary_mask(gt_path)

        if pred_mask.shape != gt_mask.shape:
            pred_mask_u8 = pred_mask.astype(np.uint8)
            pred_mask = cv2.resize(
                pred_mask_u8,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = float(intersection / union) if union > 0 else 1.0

        total_intersection += int(intersection)
        total_union += int(union)
        frame_ious.append(
            {
                "pair_idx": idx,
                "pred_file": pred_path.name,
                "gt_file": gt_path.name,
                "iou": iou,
            }
        )

    overall_iou = float(total_intersection / total_union) if total_union > 0 else 1.0
    mean_frame_iou = float(np.mean([x["iou"] for x in frame_ious]))

    return {
        "overall_iou": overall_iou,
        "mean_frame_iou": mean_frame_iou,
        "num_compared_frames": len(pred_files),
        "num_pred_frames": len(pred_files),
        "num_gt_frames": len(gt_files),
        "frame_ious": frame_ious,
    }


def resolve_scene_paths(scene_dir: Path, pred_scene_dir: Path = None):
    if pred_scene_dir is None:
        pred_scene_dir = scene_dir
    pred_mask_dir = pred_scene_dir / "video_2_difference_masks"
    gt_mask_dir = scene_dir / "Instance_1" / "gt_mask"
    return pred_mask_dir, gt_mask_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU for video_2 difference masks."
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="data/PASLCD/Cantina/",
        help="Path to scene directory containing Instance_1/gt_mask",
    )
    parser.add_argument(
        "--pred_scene_dir",
        type=str,
        default="outputs/MV3DCD/Cantina/Instance_1",
        help="Optional path to scene prediction directory (defaults to scene_dir)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/test_MV3DCD",
        help="Optional output json path for detailed per-frame IoU",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    pred_scene_dir = Path(args.pred_scene_dir) if args.pred_scene_dir else None
    pred_mask_dir, gt_mask_dir = resolve_scene_paths(scene_dir, pred_scene_dir)

    results = compute_iou_for_scene(pred_mask_dir, gt_mask_dir)

    print(f"Prediction masks: {pred_mask_dir}")
    print(f"GT masks: {gt_mask_dir}")
    print(f"Compared frames: {results['num_compared_frames']}")
    print(f"Overall IoU: {results['overall_iou']:.6f}")
    print(f"Mean Frame IoU: {results['mean_frame_iou']:.6f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved details to: {output_path}")


if __name__ == "__main__":
    main()
