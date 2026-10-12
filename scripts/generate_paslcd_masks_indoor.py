#!/usr/bin/env python3
"""Generate SAM masks for the PASLCD indoor scenes (Cantina, Lounge, Lunch_room, Meeting_room,
Printing_area). The defaults are the settings used for the released masks: single-level SAM
(modules/sam_single_level_generator.py) on images resized to 2072x1176, 32 points per side,
pred_iou 0.78, stability 0.85, box NMS 0.7, one crop layer, small regions painted on top.
Writes <scene_dir>/sam_masks_additional.pkl, which modules/mask_model.py reads.
"""
import argparse
import fcntl
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _add_repo_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    sam_root = repo_root / "submodules" / "segment-anything-langsplat-modified"
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def extract_frames_from_scene(scene_dir: Path, resample_rate: int) -> list[Path]:
    video_1_dir = scene_dir / "video1_frames"
    video_2_dir = scene_dir / "video2_frames"
    if video_1_dir.exists() and video_2_dir.exists():
        video_1_files = sorted(
            [p for p in video_1_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )[::resample_rate]
        video_2_files = sorted(
            [p for p in video_2_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )[::resample_rate]
        return video_1_files + video_2_files

    frame_dir = scene_dir / "images"
    if not frame_dir.exists():
        return []

    image_files = sorted(
        [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    video_1_files = [p for p in image_files if "test" not in p.name][::resample_rate]
    video_2_files = [p for p in image_files if "test" in p.name]
    return video_1_files + video_2_files


def _build_generator(args: argparse.Namespace):
    from segment_anything_ls import sam_model_registry
    from modules.sam_single_level_generator import SamAutomaticMaskGenerator

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam = sam.to(device=args.device)
    return SamAutomaticMaskGenerator(
        model=sam,
        output_mode="binary_mask",
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )


def _normalize_levels(
    raw_masks: Any,
    use_masks_update: bool = True,
) -> List[Sequence[dict[str, Any]]]:
    if isinstance(raw_masks, tuple):
        levels = [list(raw_masks[0]), list(raw_masks[1])]
        if use_masks_update:
            try:
                from utils import masks_update

                levels = list(masks_update(levels[0], levels[1], iou_thr=0.8, score_thr=0.7, inner_thr=0.5))
            except Exception as exc:
                print(f"[WARN] masks_update unavailable; continuing without it: {exc}")
        return levels
    if isinstance(raw_masks, list):
        return [raw_masks]
    raise TypeError(f"Unsupported generator output type: {type(raw_masks)}")


def build_non_overlapping_groups(
    image_shape: tuple[int, int],
    mask_levels: List[Sequence[dict[str, Any]]],
    min_size: int,
    small_to_large: bool,
) -> np.ndarray:
    h, w = image_shape
    group_ids = np.full((h, w), -1, dtype=np.int32)
    group_counter = 0

    for mask_level in mask_levels:
        if int((group_ids == -1).sum()) < min_size:
            break

        ordered = sorted(mask_level, key=lambda x: x["area"], reverse=not small_to_large)
        for mask_dict in ordered:
            mask_single = np.asarray(mask_dict["segmentation"], dtype=bool)
            non_mask_area = group_ids == -1
            to_mask_area = mask_single & non_mask_area
            if int(to_mask_area.sum()) < min_size:
                continue

            group_ids[to_mask_area] = group_counter
            group_counter += 1

            if int((group_ids == -1).sum()) < min_size:
                break

    return group_ids


def _load_existing_name_to_mask(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}

    if isinstance(data, dict):
        if "file_names" in data and "masks" in data:
            return {
                n: np.asarray(m)
                for n, m in zip(data.get("file_names", []), data.get("masks", []))
            }
        return {str(k): np.asarray(v) for k, v in data.items()}

    return {}


def _atomic_pickle_dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)
        pickle.dump(obj, tmp_f)
        tmp_f.flush()
        os.fsync(tmp_f.fileno())
    os.replace(tmp_path, path)


def _upsert_name_mask_locked(path: Path, name: str, mask: np.ndarray) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        existing = _load_existing_name_to_mask(path)
        existing[name] = mask
        _atomic_pickle_dump(path, existing)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _exists_name_locked(path: Path, name: str) -> bool:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        exists = name in _load_existing_name_to_mask(path)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    return exists


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM masks for PASLCD indoor scenes."
    )
    parser.add_argument("--scene_dir", required=True, help="Scene directory path.")
    parser.add_argument(
        "--additional-path",
        default="",
        help="Default: <scene_dir>/sam_masks_additional.pkl (concurrency-safe incremental cache).",
    )
    parser.add_argument(
        "--output-pkl",
        default="",
        help="Optional primary output path (name->mask dict). If set, it is updated incrementally too.",
    )
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resample-rate", type=int, default=2)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--fixed-width", type=int, default=518 * 4)
    parser.add_argument("--fixed-height", type=int, default=294 * 4)

    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--small-to-large", default=True, type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"})
    parser.add_argument("--disable-masks-update", action="store_true")

    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.78)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--min-mask-region-area", type=int, default=0)

    args = parser.parse_args()
    _add_repo_paths()

    scene_dir = Path(args.scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"scene_dir does not exist: {scene_dir}")

    file_list = extract_frames_from_scene(scene_dir, resample_rate=args.resample_rate)
    if not file_list:
        raise RuntimeError(f"No valid frames found under scene_dir: {scene_dir}")

    total_files = len(file_list)
    start_idx = max(0, int(args.start_idx))
    end_idx = total_files - 1 if int(args.end_idx) < 0 else min(int(args.end_idx), total_files - 1)
    if start_idx > end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}, total={total_files}")

    shard_file_list = file_list[start_idx : end_idx + 1]
    additional_path = (
        Path(args.additional_path) if args.additional_path else (scene_dir / "sam_masks_additional.pkl")
    )
    output_pkl = Path(args.output_pkl) if args.output_pkl else None

    print("Loading SAM...")
    generator = _build_generator(args)

    print(f"[Masks] scene_dir={scene_dir}")
    print(f"[Masks] additional_path={additional_path}")
    if output_pkl is not None:
        print(f"[Masks] output_pkl={output_pkl}")
    print(f"[Masks] frames={total_files}, range={start_idx}:{end_idx}, count={len(shard_file_list)}")

    generated = 0
    skipped = 0
    for local_idx, image_path in enumerate(shard_file_list, start=1):
        name = image_path.name
        global_idx = start_idx + local_idx - 1
        if (not args.overwrite) and _exists_name_locked(additional_path, name):
            skipped += 1
            print(f"[{global_idx + 1}/{total_files}] skip {name} (exists in additional)")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[{global_idx + 1}/{total_files}] skip {name} (failed to read)")
            continue

        image_bgr = cv2.resize(
            image_bgr,
            (args.fixed_width, args.fixed_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        raw_masks = generator.generate(image_rgb)
        mask_levels = _normalize_levels(raw_masks, use_masks_update=not args.disable_masks_update)
        group_ids = build_non_overlapping_groups(
            image_shape=image_rgb.shape[:2],
            mask_levels=mask_levels,
            min_size=args.min_size,
            small_to_large=args.small_to_large,
        )

        _upsert_name_mask_locked(additional_path, name, group_ids)
        if output_pkl is not None:
            _upsert_name_mask_locked(output_pkl, name, group_ids)
        generated += 1
        target = additional_path.name if output_pkl is None else f"{additional_path.name} (+ {output_pkl.name})"
        print(f"[{global_idx + 1}/{total_files}] generated {name} -> {target}")

    print(f"[Masks] done. generated={generated}, skipped={skipped}, total_in_range={len(shard_file_list)}")


if __name__ == "__main__":
    main()
