"""
SceneDiff on PASLCD
===================

Runs change detection on one PASLCD scene instance (a directory with images/, where
reference views are files without "test" in the name and query views contain "test").
Writes per-frame binary change masks for the query views to
<output>/video_2_difference_masks, which scripts/evaluate_all_scenes_instances_iou.py scores.

Usage:
    python scripts/predict_paslcd.py \
        --scene_dir data/PASLCD/Cantina/Instance_1 \
        --output output/paslcd/Cantina/Instance_1
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import SceneDiff

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def str2bool(v):
    """Parse common bool strings for argparse."""
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"true", "1", "yes", "y", "on"}:
        return True
    if val in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def load_config(config_path):
    """Load and prepare configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    config = expand_env_vars(config)

    # Keep demo lightweight by default (no large point-cloud outputs).
    config['visualization']['vis_pc'] = False
    config['visualization']['save_gt_visualization'] = False
    config['visualization']['save_detections'] = False  # Disable other vis

    return config


def extract_frames_from_scene(scene_dir, resample_rate=1):
    """
    Extract frame lists for video1/video2 from a scene directory.

    Args:
        scene_dir: Scene directory path
        resample_rate: Keep one frame every N frames

    Returns:
        Tuple of (video_1_files, video_2_files)
    """
    scene_dir = Path(scene_dir)

    if resample_rate < 1:
        raise ValueError("resample_rate must be >= 1")

    # Layout A: extracted frame folders
    video_1_dir = scene_dir / 'video1_frames'
    video_2_dir = scene_dir / 'video2_frames'
    if video_1_dir.exists() and video_2_dir.exists():
        video_1_files = sorted(
            [p for p in video_1_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )[::resample_rate]
        video_2_files = sorted(
            [p for p in video_2_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )[::resample_rate]
        return video_1_files, video_2_files

    frame_dir = scene_dir / 'images'
    if not frame_dir.exists():
        return [], []

    image_files = sorted(
        [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    video_1_files = [p for p in image_files if 'test' not in p.name][::resample_rate]
    video_2_files = [p for p in image_files if 'test' in p.name]
    if video_1_files and video_2_files:
        return video_1_files, video_2_files

    return [], []


def _find_existing_prediction(scene_dir, output_dir):
    """Return existing prediction pickle path if found, else None."""
    output_dir = Path(output_dir)
    candidate_files = ['object_masks.pkl', 'object.pkl']

    # Support both call styles:
    # 1) output_dir points to scene dir
    # 2) output_dir points to parent dir and scene name is appended later
    candidate_roots = [output_dir, output_dir / scene_dir.name]
    for root in candidate_roots:
        for candidate in candidate_files:
            direct_path = root / candidate
            if direct_path.exists():
                return direct_path
        if root.exists():
            for candidate in candidate_files:
                matches = sorted(root.rglob(candidate))
                if matches:
                    return matches[0]

    # Fallback: search under output_dir itself
    if output_dir.exists():
        for candidate in candidate_files:
            matches = sorted(output_dir.rglob(candidate))
            if matches:
                return matches[0]

    return None


def run_scenediff_demo(
    scene_dir,
    config_path,
    output_dir,
    resample_rate=30,
    overwrite=False,
    max_size=None,
    filter_percentage_before_threshold=None,
    lambda_dino_region_match=None,
    use_cache=None,
    save_cache=None,
    vis_pc=None,
):
    """
    Run SceneDiff on one scene directory.

    Args:
        scene_dir: Path to scene directory
        config_path: Path to configuration file
        output_dir: Output directory for results
        resample_rate: Frame sampling rate (1 frame every N frames)
        overwrite: Whether to overwrite existing predictions
    """
    scene_dir = Path(scene_dir)

    print("=" * 80)
    print("SceneDiff - PASLCD Scene Change Detection")
    print("=" * 80)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)

    # Override resample rate if specified
    if resample_rate != config['dataset']['resample_rate']:
        config['dataset']['resample_rate'] = resample_rate
        print(f"Overriding resample_rate to {resample_rate}")
    if max_size is not None:
        config['processing']['max_size'] = int(max_size)
        print(f"Overriding max_size to {config['processing']['max_size']}")
    if filter_percentage_before_threshold is not None:
        config['detection']['filter_percentage_before_threshold'] = float(filter_percentage_before_threshold)
        print("Overriding filter_percentage_before_threshold to "
              f"{config['detection']['filter_percentage_before_threshold']}")
    if lambda_dino_region_match is not None:
        config['costs']['lambda_dino_region_match'] = float(lambda_dino_region_match)
        print(f"Overriding lambda_dino_region_match to {config['costs']['lambda_dino_region_match']}")
    if use_cache is not None:
        config['cache']['use_cache'] = bool(use_cache)
        print(f"Overriding use_cache to {config['cache']['use_cache']}")
    if save_cache is not None:
        config['cache']['save_cache'] = bool(save_cache)
        print(f"Overriding save_cache to {config['cache']['save_cache']}")
    if vis_pc is not None:
        config['visualization']['vis_pc'] = bool(vis_pc)
        print(f"Overriding vis_pc to {config['visualization']['vis_pc']}")

    # Extract frames
    print("\n" + "-" * 80)
    print("Step 1: Loading frames from scene")
    print("-" * 80)

    video1_frames, video2_frames = extract_frames_from_scene(scene_dir, resample_rate)

    if len(video1_frames) == 0 or len(video2_frames) == 0:
        print("\nError: Could not find valid frame lists for both videos in scene_dir.")
        return

    # Prepare file list
    file_list = video1_frames + video2_frames

    print("\nTotal frames to process:")
    print(f"  Video 1: {len(video1_frames)} frames")
    print(f"  Video 2: {len(video2_frames)} frames")
    print(f"  Total: {len(file_list)} frames")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    existing_pred = _find_existing_prediction(scene_dir, output_dir)
    if existing_pred is not None and not overwrite:
        print("\n" + "=" * 80)
        print("Skipping processing: existing prediction found")
        print("=" * 80)
        print(f"Prediction file: {existing_pred}")
        print("Use --overwrite to recompute predictions for this scene.")
        return

    # Initialize SceneDiff
    print("\n" + "-" * 80)
    print("Step 2: Initializing SceneDiff pipeline")
    print("-" * 80)

    scene_diff = SceneDiff(config, device)

    # Process scene
    print("\n" + "-" * 80)
    print("Step 3: Running change detection")
    print("-" * 80)

    try:
        results = scene_diff.process_scene(
            file_list,
            img_1_length=len(video1_frames),
            img_2_length=len(video2_frames),
            output_dir=Path(output_dir),
        )

        print("\n" + "=" * 80)
        print("Processing complete")
        print("=" * 80)

        # Print output locations
        results_dir = Path(output_dir)
        print("\nOutput files:")
        print(f"  Object masks: {results_dir / 'object_masks.pkl'}")
        print("\nPoint cloud visualizations:")
        print(f"  Combined cost map: {results_dir / 'cost_map_merged.ply'}")
        print(f"  Video 1 cost map: {results_dir / 'cost_map_1.ply'}")
        print(f"  Video 2 cost map: {results_dir / 'cost_map_2.ply'}")
        print(f"  Video 1 RGB point cloud: {results_dir / 'rgb_pc_1.ply'}")
        print(f"  Video 2 RGB point cloud: {results_dir / 'rgb_pc_2.ply'}")
        print(f"  Object instances (Video 1): {results_dir / 'object_instances_1.ply'}")
        print(f"  Object instances (Video 2): {results_dir / 'object_instances_2.ply'}")

        # Print detected objects summary
        num_objects = len([k for k in results.keys() if k not in ['H', 'W']])
        print(f"\nDetected {num_objects} changed objects")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        scene_diff.cleanup()

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='SceneDiff - change detection on one PASLCD scene instance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
        python scripts/predict_paslcd.py \\
            --scene_dir data/PASLCD/Cantina/Instance_1 \\
            --output output/paslcd/Cantina/Instance_1
                """,
    )

    # Required arguments
    parser.add_argument(
        '--scene_dir',
        type=str,
        required=True,
        help='Path to scene directory containing frame data',
    )

    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='configs/paslcd_config.yml',
        help='Path to configuration file (default: configs/paslcd_config.yml)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/paslcd',
        help='Output directory for results (default: output/paslcd)',
    )
    parser.add_argument(
        '--resample_rate',
        type=int,
        default=2,
        help='Keep every N-th reference view; all query views are kept (default: 2).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing predictions. If unset, skip when object_masks.pkl/object.pkl exists.',
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=None,
        help='Override processing.max_size from config.',
    )
    parser.add_argument(
        '--filter_percentage_before_threshold',
        type=float,
        default=None,
        help='Override detection.filter_percentage_before_threshold from config.',
    )
    parser.add_argument(
        '--lambda_dino_region_match',
        type=float,
        default=None,
        help='Override costs.lambda_dino_region_match from config.',
    )
    parser.add_argument(
        '--use_cache',
        type=str2bool,
        default=None,
        help='Override cache.use_cache from config.',
    )
    parser.add_argument(
        '--save_cache',
        type=str2bool,
        default=None,
        help='Override cache.save_cache from config.',
    )
    parser.add_argument(
        '--vis_pc',
        type=str2bool,
        default=None,
        help='Override visualization.vis_pc from config.',
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.scene_dir):
        print(f"Error: Scene directory not found: {args.scene_dir}")
        return

    if not os.path.isdir(args.scene_dir):
        print(f"Error: scene_dir is not a directory: {args.scene_dir}")
        return

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    # Run demo
    run_scenediff_demo(
        args.scene_dir,
        args.config,
        args.output,
        args.resample_rate,
        args.overwrite,
        args.max_size,
        args.filter_percentage_before_threshold,
        args.lambda_dino_region_match,
        args.use_cache,
        args.save_cache,
        args.vis_pc,
    )


if __name__ == "__main__":
    main()
