"""
Batch-evaluate video_2 difference-mask IoU over all scenes and both instances.

This script scans scene folders (e.g., data/PASLCD/<scene>/Instance_1|Instance_2),
finds corresponding prediction folders (including nested layouts), computes IoU,
and saves separate reports for Instance_1 and Instance_2.
"""

import argparse
import json
from pathlib import Path

from compute_video2_difference_iou import compute_iou_for_scene


def discover_scenes(data_root: Path):
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])


def resolve_prediction_mask_dir(pred_root: Path, scene: str, instance: str):
    """
    Find prediction mask directory for a scene/instance.

    Supports:
    - <pred_root>/<scene>/<instance>/video_2_difference_masks
    - Nested variants under <pred_root>/<scene>/<instance>/**/video_2_difference_masks
    """
    scene_dir = pred_root / scene
    if not scene_dir.exists():
        return None

    direct = scene_dir / instance / "video_2_difference_masks"
    if direct.exists():
        return direct

    instance_root = scene_dir / instance
    if not instance_root.exists():
        return None

    matches = sorted(instance_root.glob("**/video_2_difference_masks"))
    if not matches:
        return None

    # Prefer shallower candidate in case multiple are present.
    return min(matches, key=lambda p: len(p.relative_to(instance_root).parts))


def evaluate_one(scene_root: Path, pred_root: Path, scene: str, instance: str):
    gt_mask_dir = scene_root / scene / instance / "gt_mask"
    if not gt_mask_dir.exists():
        return {
            "scene": scene,
            "instance": instance,
            "status": "missing_gt",
            "gt_mask_dir": str(gt_mask_dir),
        }

    pred_mask_dir = resolve_prediction_mask_dir(pred_root, scene, instance)
    if pred_mask_dir is None:
        return {
            "scene": scene,
            "instance": instance,
            "status": "missing_pred",
        }

    try:
        metrics = compute_iou_for_scene(pred_mask_dir, gt_mask_dir)
        return {
            "scene": scene,
            "instance": instance,
            "status": "ok",
            "pred_mask_dir": str(pred_mask_dir),
            "gt_mask_dir": str(gt_mask_dir),
            "overall_iou": metrics["overall_iou"],
            "mean_frame_iou": metrics["mean_frame_iou"],
            "num_compared_frames": metrics["num_compared_frames"],
        }
    except Exception as exc:
        return {
            "scene": scene,
            "instance": instance,
            "status": "error",
            "pred_mask_dir": str(pred_mask_dir),
            "gt_mask_dir": str(gt_mask_dir),
            "error": str(exc),
        }


def summarize(records):
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    if ok:
        mean_overall = sum(r["overall_iou"] for r in ok) / len(ok)  # mIoU over scenes
        mean_frame = sum(r["mean_frame_iou"] for r in ok) / len(ok)
    else:
        mean_overall = None
        mean_frame = None
    return {
        "total": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "miou": mean_overall,
        "mean_frame_iou_of_scenes": mean_frame,
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all scenes for Instance_1 and Instance_2 and save results separately."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/PASLCD",
        help="Dataset root containing <scene>/Instance_1 and <scene>/Instance_2.",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="outputs/MV3DCD",
        help="Prediction root containing per-scene outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/MV3DCD_eval",
        help="Directory to save evaluation reports.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    output_dir = Path(args.output_dir)

    instances = ["Instance_1", "Instance_2"]
    scenes = discover_scenes(data_root)

    all_records = []
    per_instance = {inst: [] for inst in instances}

    for scene in scenes:
        for inst in instances:
            record = evaluate_one(data_root, pred_root, scene, inst)
            all_records.append(record)
            per_instance[inst].append(record)

    # Build scene-level table: one row per scene with both instances side-by-side.
    idx_i1 = {r["scene"]: r for r in per_instance["Instance_1"]}
    idx_i2 = {r["scene"]: r for r in per_instance["Instance_2"]}
    scene_level_rows = []
    for scene in scenes:
        r1 = idx_i1.get(scene, {"status": "missing"})
        r2 = idx_i2.get(scene, {"status": "missing"})
        i1 = r1.get("overall_iou") if r1.get("status") == "ok" else None
        i2 = r2.get("overall_iou") if r2.get("status") == "ok" else None
        valid = [x for x in [i1, i2] if x is not None]
        scene_miou = (sum(valid) / len(valid)) if valid else None
        scene_level_rows.append(
            {
                "scene": scene,
                "instance_1_iou": i1,
                "instance_2_iou": i2,
                "scene_miou": scene_miou,  # averaged IoU for this scene over inst1/inst2
                "instance_1_status": r1.get("status"),
                "instance_2_status": r2.get("status"),
            }
        )

    for inst in instances:
        summary = summarize(per_instance[inst])
        payload = {"instance": inst, "summary": summary, "results": per_instance[inst]}
        out_path = output_dir / f"{inst.lower()}_results.json"
        save_json(out_path, payload)
        print(f"[{inst}] saved: {out_path}")
        print(
            f"[{inst}] total={summary['total']} ok={summary['ok']} failed={summary['failed']} "
            f"mIoU={summary['miou']}"
        )

    combined_summary = {
        inst: summarize(per_instance[inst]) for inst in instances
    }
    valid_scene_mious = [r["scene_miou"] for r in scene_level_rows if r["scene_miou"] is not None]
    overall_scene_miou = (
        sum(valid_scene_mious) / len(valid_scene_mious) if valid_scene_mious else None
    )

    print("[Scene-level] averaged IoU per scene (inst1+inst2):")
    for row in scene_level_rows:
        print(
            f"  {row['scene']}: avg_iou={row['scene_miou']} "
            f"(inst1={row['instance_1_iou']}, inst2={row['instance_2_iou']})"
        )

    save_json(
        output_dir / "scene_level_results.json",
        {"rows": scene_level_rows, "overall_scene_miou": overall_scene_miou},
    )

    save_json(
        output_dir / "combined_summary.json",
        {
            "data_root": str(data_root),
            "pred_root": str(pred_root),
            "scenes": scenes,
            "summary_by_instance": combined_summary,
            "overall_scene_miou": overall_scene_miou,
            "all_results": all_records,
        },
    )
    print(f"[Scene-level] saved: {output_dir / 'scene_level_results.json'}")
    print(f"[All scenes] avg_iou_all_scenes: {overall_scene_miou}")
    print(f"[Combined] saved: {output_dir / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
