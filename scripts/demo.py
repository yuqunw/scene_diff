"""
SceneDiff Demo Script
=====================

Simple demo for processing two videos to detect scene changes.
Outputs point cloud visualizations for inspection.

Usage:
    python demo.py \
        --video1 path/to/video1.mp4 \
        --video2 path/to/video2.mp4 \
        --config scenediff_config.yml \
        --output demo_output
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
from utils import process_video_to_frames


def load_config(config_path):
    """Load and prepare configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return os.path.expandvars(obj)
        else:
            return obj
    
    config = expand_env_vars(config)
    
    # Force enable point cloud visualization
    config['visualization']['vis_pc'] = True
    config['visualization']['save_detections'] = False  # Disable other vis
    
    return config


def extract_frames_from_video(video_path, output_dir, resample_rate=1):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        resample_rate: Extract 1 frame every N frames
        
    Returns:
        List of frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    print(f"Extracting frames from {video_path}...")
    process_video_to_frames(str(video_path), str(output_dir))
    
    # Get frame list with resampling
    frame_list = sorted(output_dir.glob('*.jpg'))
    if resample_rate > 1:
        frame_list = frame_list[::resample_rate]
    
    print(f"  Extracted {len(frame_list)} frames (resample rate: {resample_rate})")
    
    return frame_list


def run_scenediff_demo(video1_path, video2_path, config_path, output_dir, resample_rate=30):
    """
    Run SceneDiff on two videos.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        config_path: Path to configuration file
        output_dir: Output directory for results
        resample_rate: Frame sampling rate (1 frame every N frames)
    """
    print("="*80)
    print("SceneDiff Demo - Two Video Change Detection")
    print("="*80)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)
    
    # Override resample rate if specified
    if resample_rate != config['dataset']['resample_rate']:
        config['dataset']['resample_rate'] = resample_rate
        print(f"Overriding resample_rate to {resample_rate}")
    
    # Create temporary directories for frame extraction
    temp_dir = Path(output_dir) / 'temp_frames'
    video1_frames_dir = temp_dir / 'video1'
    video2_frames_dir = temp_dir / 'video2'
    
    # Extract frames
    print("\n" + "-"*80)
    print("Step 1: Extracting frames from videos")
    print("-"*80)
    
    video1_frames = extract_frames_from_video(video1_path, video1_frames_dir, resample_rate)
    video2_frames = extract_frames_from_video(video2_path, video2_frames_dir, resample_rate)
    
    if len(video1_frames) == 0 or len(video2_frames) == 0:
        print("\nError: No frames extracted from videos!")
        return
    
    # Prepare file list
    file_list = video1_frames + video2_frames
    
    print(f"\nTotal frames to process:")
    print(f"  Video 1: {len(video1_frames)} frames")
    print(f"  Video 2: {len(video2_frames)} frames")
    print(f"  Total: {len(file_list)} frames")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize SceneDiff
    print("\n" + "-"*80)
    print("Step 2: Initializing SceneDiff pipeline")
    print("-"*80)
    
    scene_diff = SceneDiff(config, device)
    
    # Process scene
    print("\n" + "-"*80)
    print("Step 3: Running change detection")
    print("-"*80)
    
    try:
        results = scene_diff.process_scene(
            file_list,
            img_1_length=len(video1_frames),
            img_2_length=len(video2_frames),
            output_dir=Path(output_dir) / 'results'
        )
        
        print("\n" + "="*80)
        print("✓ Processing complete!")
        print("="*80)
        
        # Print output locations
        results_dir = Path(output_dir) / 'results'
        print(f"\nOutput files:")
        print(f"  Object masks: {results_dir / 'object_masks.pkl'}")
        print(f"\nPoint cloud visualizations:")
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
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        scene_diff.cleanup()
        
        # Optionally clean up temporary frames
        # import shutil
        # shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='SceneDiff Demo - Simple two-video change detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Basic usage
        python demo.py --video1 before.mp4 --video2 after.mp4
        
        # With custom config and output
        python demo.py \\
            --video1 video1.mp4 \\
            --video2 video2.mp4 \\
            --config my_config.yml \\
            --output my_demo_output
        
        # Process more frames (slower but more accurate)
        python demo.py \\
            --video1 video1.mp4 \\
            --video2 video2.mp4 \\
            --resample_rate 15
                """
    )
    
    # Required arguments
    parser.add_argument(
        '--video1',
        type=str,
        required=True,
        help='Path to first video (before scene change)'
    )
    parser.add_argument(
        '--video2',
        type=str,
        required=True,
        help='Path to second video (after scene change)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default='scenediff_config.yml',
        help='Path to configuration file (default: scenediff_config.yml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='demo_output',
        help='Output directory for results (default: demo_output)'
    )
    parser.add_argument(
        '--resample_rate',
        type=int,
        default=30,
        help='Sample 1 frame every N frames assuming FPS is 30 (default: 30).'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video1):
        print(f"Error: Video 1 not found: {args.video1}")
        return
    
    if not os.path.exists(args.video2):
        print(f"Error: Video 2 not found: {args.video2}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Run demo
    run_scenediff_demo(
        args.video1,
        args.video2,
        args.config,
        args.output,
        args.resample_rate
    )


if __name__ == "__main__":
    main()
