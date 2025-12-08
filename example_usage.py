"""
Example Usage of Refactored SceneDiff
======================================

Demonstrates various ways to use the refactored SceneDiff pipeline.
"""

import yaml
import torch
from pathlib import Path

from modules import SceneDiff, ConfigManager


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Load configuration
    with open('scenediff_config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_diff = SceneDiff(config, device)
    
    # Process a scene (example with dummy paths)
    file_list = [
        # Video 1 frames
        "path/to/video1/frame_0.jpg",
        "path/to/video1/frame_1.jpg",
        # ... more frames
        # Video 2 frames
        "path/to/video2/frame_0.jpg",
        "path/to/video2/frame_1.jpg",
        # ... more frames
    ]
    
    try:
        results = scene_diff.process_scene(
            file_list=file_list,
            img_1_length=10,  # Number of frames in video 1
            img_2_length=10,  # Number of frames in video 2
            output_dir='output/my_scene'
        )
        print(f"Processing complete. Results saved to output/my_scene")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        scene_diff.cleanup()
    
    print()


def example_2_using_config_manager():
    """Example 2: Using ConfigManager for advanced configuration."""
    print("="*80)
    print("Example 2: Using ConfigManager")
    print("="*80)
    
    # Initialize config manager
    config_mgr = ConfigManager('scenediff_config.yml')
    
    # Modify configuration
    config_mgr.set('costs.lambda_geo', 1.5)
    config_mgr.set('detection.threshold_method', 'max_entropy')
    config_mgr.set('visualization.vis_pc', True)
    
    # Validate configuration
    errors = config_mgr.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Print configuration summary
    config_mgr.print_summary()
    
    # Save modified configuration
    config_mgr.save('my_custom_config.yml')
    print("Saved custom configuration to my_custom_config.yml")
    
    # Use in SceneDiff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_diff = SceneDiff(config_mgr.config, device)
    
    # Process scenes...
    scene_diff.cleanup()
    
    print()


def example_3_individual_model_usage():
    """Example 3: Using individual models separately."""
    print("="*80)
    print("Example 3: Individual Model Usage")
    print("="*80)
    
    from modules import GeometryModel, MaskModel, SemanticModel
    
    # Load configuration
    config_mgr = ConfigManager('scenediff_config.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize individual models
    print("Initializing models...")
    geometry_model = GeometryModel(config_mgr.config, device)
    mask_model = MaskModel(config_mgr.config, device)
    semantic_model = SemanticModel(config_mgr.config, device)
    
    # Example: Use geometry model
    print("\n1. Geometry Model:")
    print("   - Estimate depth and poses")
    print("   - Compute intrinsics")
    print("   - Calculate geometric costs")
    
    # Example: Use mask model
    print("\n2. Mask Model:")
    print("   - Generate SAM segmentation masks")
    print("   - Aggregate pixel values to regions")
    print("   - Cache masks for reuse")
    
    # Example: Use semantic model
    print("\n3. Semantic Model:")
    print("   - Extract DINOv3 features")
    print("   - Compute region matching costs")
    print("   - Compute rendering-based costs")
    
    # Clean up
    geometry_model.cleanup()
    mask_model.cleanup()
    semantic_model.cleanup()
    
    print("\nModels cleaned up successfully")
    print()


def example_4_custom_processing():
    """Example 4: Custom processing with modified parameters."""
    print("="*80)
    print("Example 4: Custom Processing")
    print("="*80)
    
    # Create custom configuration
    config = {
        'models': {
            'pi3': {'name': 'yyfz233/Pi3', 'device': 'cuda'},
            'sam': {
                'checkpoint': '/path/to/sam.pth',
                'points_per_side': 16,  # Faster but less detailed
            },
            'dinov3': {
                'path': '/path/to/dinov3',
                'model_name': 'dinov3_vith16plus',
                'feature_dim': 1280,
            }
        },
        'processing': {
            'visible_percentage': 0.6,  # More strict overlap requirement
            'subsample_size': 500000,   # Smaller for faster processing
            'total_voxel_number': 150,
            'occlusion_threshold': -0.03,
        },
        'costs': {
            'lambda_geo': 1.2,      # Emphasize geometry
            'lambda_dino': 0.2,
            'lambda_dino_region_match': 0.05,
        },
        'detection': {
            'min_detection_pixel': 150,  # Larger minimum size
            'threshold_method': 'otsu',
            'change_region_threshold': None,
            'filter_percentage_before_threshold': 0.7,
            'max_score': 0.5,
        },
        'merging': {
            'geometry_distance_threshold_of_voxel_size': 3,  # More lenient merging
            'visual_threshold_ratio': 0.65,
            'geometry_threshold_ratio': 0.45,
            'general_threshold': 1.2,
            'object_similarity_threshold': 0.75,
        },
        'cache': {'use_cache': True, 'save_cache': True},
        'visualization': {'vis_pc': True, 'save_detections': True},
        'output': {'output_dir': 'output_custom'},
        'dataset': {'resample_rate': 30}
    }
    
    print("Custom configuration:")
    print(f"  - Geometry weight: {config['costs']['lambda_geo']}")
    print(f"  - Min detection pixels: {config['detection']['min_detection_pixel']}")
    print(f"  - Object similarity threshold: {config['merging']['object_similarity_threshold']}")
    print(f"  - Using cache: {config['cache']['use_cache']}")
    
    # Initialize with custom config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_diff = SceneDiff(config, device)
    
    # Process with custom parameters...
    scene_diff.cleanup()
    
    print("Custom processing setup complete")
    print()


def example_5_batch_processing():
    """Example 5: Batch processing multiple scenes."""
    print("="*80)
    print("Example 5: Batch Processing")
    print("="*80)
    
    # Load configuration
    config_mgr = ConfigManager('scenediff_config.yml')
    config_mgr.set('cache.use_cache', True)
    config_mgr.set('cache.save_cache', True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize once for all scenes
    scene_diff = SceneDiff(config_mgr.config, device)
    
    # List of scenes to process
    scenes = [
        {
            'name': 'scene_1',
            'video1_frames': ['path/to/scene1/v1_frame_*.jpg'],
            'video2_frames': ['path/to/scene1/v2_frame_*.jpg'],
        },
        {
            'name': 'scene_2',
            'video1_frames': ['path/to/scene2/v1_frame_*.jpg'],
            'video2_frames': ['path/to/scene2/v2_frame_*.jpg'],
        },
        # ... more scenes
    ]
    
    print(f"Processing {len(scenes)} scenes...")
    
    for i, scene_info in enumerate(scenes):
        print(f"\nProcessing scene {i+1}/{len(scenes)}: {scene_info['name']}")
        
        # Prepare file list
        # file_list = load_frame_paths(scene_info)
        
        try:
            # Process scene
            # results = scene_diff.process_scene(
            #     file_list=file_list,
            #     img_1_length=len(scene_info['video1_frames']),
            #     img_2_length=len(scene_info['video2_frames']),
            #     output_dir=f"output/{scene_info['name']}"
            # )
            print(f"  ✓ Scene {scene_info['name']} processed successfully")
        except Exception as e:
            print(f"  ✗ Error processing scene {scene_info['name']}: {e}")
            continue
    
    # Clean up once at the end
    scene_diff.cleanup()
    print("\nBatch processing complete")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("SceneDiff Refactored - Usage Examples")
    print("="*80 + "\n")
    
    # Note: These are demonstration examples
    # Uncomment the ones you want to try
    
    # example_1_basic_usage()
    # example_2_using_config_manager()
    example_3_individual_model_usage()
    # example_4_custom_processing()
    # example_5_batch_processing()
    
    print("\nNote: Examples 1, 4, and 5 require actual image paths to run.")
    print("Examples 2 and 3 demonstrate configuration and model initialization.")


if __name__ == "__main__":
    main()
