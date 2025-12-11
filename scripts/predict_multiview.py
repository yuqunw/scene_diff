"""
SceneDiff: Multi-view Scene Change Detection (Refactored)
==========================================================

Clean, class-based implementation with modular architecture.

Usage:
    python predict_multiview_refactored.py --config scenediff_config.yml --splits val --sets All
"""

import os
import sys
import json
import argparse
from pathlib import Path

import yaml
import torch

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after paths are set
from modules import SceneDiff
from utils import process_video_to_frames


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables in paths
    for section in ['models', 'dataset']:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, str):
                    config[section][key] = os.path.expandvars(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            config[section][key][subkey] = os.path.expandvars(subvalue)
    
    return config


def get_scenes_to_process(config, args):
    """
    Determine which scenes to process based on splits and sets.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        List of scene names to process
    """
    # Load dataset splits
    val_split_path = config['dataset']['splits']['val']
    test_split_path = config['dataset']['splits']['test']
    
    val_split = json.load(open(val_split_path, 'r'))
    val_split['all'] = val_split.get('varied', []) + val_split.get('kitchen', [])
    
    test_split = json.load(open(test_split_path, 'r'))
    test_split['all'] = test_split.get('varied', []) + test_split.get('kitchen', [])
    
    # Determine scenes to process
    if args.splits == 'val':
        valid_scenes = val_split[args.sets]
    elif args.splits == 'test':
        valid_scenes = test_split[args.sets]
    elif args.splits == 'all':
        valid_scenes = val_split[args.sets] + test_split[args.sets]
    else:
        raise ValueError(f"Invalid split: {args.splits}")
    
    return valid_scenes


def filter_already_processed_scenes(valid_scenes, output_dir):
    """
    Remove scenes that have already been processed.
    
    Args:
        valid_scenes: List of scene names
        output_dir: Output directory path
        
    Returns:
        Tuple of (scenes_to_process, finished_scenes)
    """
    output_dir = Path(output_dir)
    finished_scenes = []
    
    for scene_name in list(valid_scenes):
        for output_subdir in output_dir.glob(f'{scene_name}*'):
            if output_subdir.is_dir() and (output_subdir / 'object_masks.pkl').exists():
                print(f'{output_subdir} already processed')
                valid_scenes.remove(scene_name)
                finished_scenes.append(scene_name)
                break
    
    return valid_scenes, finished_scenes


def prepare_scene_data(scene_dir, resample_rate):
    """
    Prepare scene data including video frame extraction.
    
    Args:
        scene_dir: Path to scene directory
        resample_rate: Sample 1 frame every N frames
        
    Returns:
        Tuple of (img_1_list, img_2_list) or None if no frames found
    """
    video_1_dir = scene_dir / 'video1_frames'
    video_2_dir = scene_dir / 'video2_frames'
    
    # Extract frames from videos if needed
    video_files = sorted([
        x for x in os.listdir(scene_dir)
        if x.lower().endswith(('.mp4', '.mov', '.avi'))
    ])
    
    if len(video_files) >= 2:
        video_1_path = scene_dir / video_files[0]
        video_2_path = scene_dir / video_files[1]
        
        if not video_1_dir.exists() or not video_2_dir.exists():
            print(f"  Extracting frames from videos...")
            process_video_to_frames(str(video_1_path), str(video_1_dir))
            process_video_to_frames(str(video_2_path), str(video_2_dir))
    
    # Load frame lists
    img_1_list = sorted(video_1_dir.glob('*.jpg'))[::resample_rate]
    img_2_list = sorted(video_2_dir.glob('*.jpg'))[::resample_rate]
    
    if len(img_1_list) == 0 or len(img_2_list) == 0:
        return None
    
    return img_1_list, img_2_list


def process_single_scene(scene_name, scene_dir, scene_diff_model, config):
    """
    Process a single scene for change detection.
    
    Args:
        scene_name: Name of the scene
        scene_dir: Path to scene directory
        scene_diff_model: SceneDiff instance
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*80}\n")
    
    # Prepare scene data
    resample_rate = config['dataset']['resample_rate']
    result = prepare_scene_data(scene_dir, resample_rate)
    
    if result is None:
        print(f"Skipping {scene_name}: no frames found")
        return False
    
    img_1_list, img_2_list = result
    file_list = img_1_list + img_2_list
    
    print(f"  Video 1: {len(img_1_list)} frames")
    print(f"  Video 2: {len(img_2_list)} frames")
    
    # Process scene
    # try:
    output_dir = Path(config['output']['output_dir'])
    scene_diff_model.process_scene(
        file_list,
        len(img_1_list),
        len(img_2_list),
        output_dir
    )
    return True
    # except Exception as e:
    #     print(f"Error processing {scene_name}: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SceneDiff: Multi-view Scene Change Detection (Refactored)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/scenediff_config.yml',
        help='Path to configuration file'
    )
    
    # Dataset selection
    parser.add_argument(
        '--splits',
        type=str,
        default='all',
        choices=['val', 'test', 'all'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--sets',
        type=str,
        default='all',
        choices=['varied', 'kitchen', 'all'],
        help='Dataset subset to process'
    )
    
    # Override options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Override output directory from config'
    )
    parser.add_argument(
        '--use_cache',
        type=eval,
        default=False,
        help='Override use_cache from config'
    )
    parser.add_argument(
        '--save_cache',
        type=eval,
        default=False,
        help='Override save_cache from config'
    )
    parser.add_argument(
        '--vis_pc',
        type=eval,
        default=False,
        help='Override vis_pc from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Apply command line overrides
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.use_cache is not None:
        config['cache']['use_cache'] = args.use_cache
    if args.save_cache is not None:
        config['cache']['save_cache'] = args.save_cache
    if args.vis_pc is not None:
        config['visualization']['vis_pc'] = args.vis_pc
    
    # Determine scenes to process
    valid_scenes = get_scenes_to_process(config, args)
    print(f"\nProcessing: {args.splits} split, {args.sets} set")
    print(f"Total scenes in selection: {len(valid_scenes)}")
    
    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter already processed scenes
    scenes_to_process, finished_scenes = filter_already_processed_scenes(
        valid_scenes.copy(),
        output_dir
    )

    
    print(f'Scenes to process: {len(scenes_to_process)}')
    print(f'Already processed: {len(finished_scenes)}')
    
    if len(scenes_to_process) == 0:
        print("\nNo scenes to process. Exiting.")
        return
    
    # Initialize SceneDiff pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    scene_diff_model = SceneDiff(config, device)
    
    # Process each scene
    gt_dir = Path(config['dataset']['gt_dir'])
    success_count = 0
    failure_count = 0
    
    for scene_name in scenes_to_process:
        scene_dir = gt_dir / scene_name
        
        if not scene_dir.exists():
            print(f"Scene directory not found: {scene_dir}")
            continue
        
        success = process_single_scene(scene_name, scene_dir, scene_diff_model, config)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Clean up
    scene_diff_model.cleanup()
    
    # Print summary
    print(f"\n{'='*80}")
    print("Processing Summary")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count} scenes")
    print(f"Failed: {failure_count} scenes")
    print(f"Already processed: {len(finished_scenes)} scenes")
    print(f"Total: {len(valid_scenes)} scenes")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
