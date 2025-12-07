# SceneDiff: Multi-view Scene Change Detection (Refactored)

A clean, modular implementation of multi-view scene change detection using depth estimation, segmentation, and semantic features.

## Architecture Overview

The refactored codebase follows a clean separation of concerns with three main model components:

```
SceneDiff/
├── scenediff_config.yml          # Main configuration file
├── predict_multiview_refactored.py  # Entry point
├── modules/
│   ├── __init__.py
│   ├── geometry_model.py         # Depth & pose estimation (Pi3)
│   ├── mask_model.py            # Segmentation (SAM)
│   ├── semantic_model.py        # Feature extraction (DINOv3)
│   ├── scene_diff.py            # Main orchestrator
│   └── config_manager.py        # Configuration utilities
└── utils.py                     # Shared utilities
```

## Components

### 1. Geometry Model (`modules/geometry_model.py`)
Handles all geometric computations:
- Depth map estimation using Pi3
- Camera pose computation
- Intrinsic matrix calculation
- Scene scale normalization
- Geometric cost computation (foreground distance)
- Confidence map generation

**Key Methods:**
- `estimate_depth_and_poses()`: Estimate depth and poses for all views
- `compute_intrinsics()`: Compute camera intrinsics
- `normalize_scene_scale()`: Normalize scene to consistent scale
- `compute_geometric_costs()`: Compute geometric costs between views

### 2. Mask Model (`modules/mask_model.py`)
Handles segmentation:
- SAM (Segment Anything Model) initialization
- Mask generation and caching
- Region-based aggregation
- Mask statistics

**Key Methods:**
- `load_or_generate_masks()`: Load cached or generate new segmentation masks
- `aggregate_to_regions()`: Aggregate pixel values to segmentation regions
- `get_mask_statistics()`: Get statistics about masks

### 3. Semantic Model (`modules/semantic_model.py`)
Handles appearance features:
- DINOv3 feature extraction
- Region matching costs
- Rendering-based appearance costs
- Feature aggregation to voxels

**Key Methods:**
- `extract_features()`: Extract DINOv3 features from image
- `compute_region_matching_cost()`: Compute region-level matching cost
- `compute_rendering_cost()`: Compute rendering-based cost
- `aggregate_features_to_voxels()`: Aggregate features to 3D voxels

### 4. SceneDiff Main Pipeline (`modules/scene_diff.py`)
Orchestrates the entire pipeline:
- Initializes all sub-models
- Manages processing stages
- Coordinates cost computation
- Handles object detection and merging
- Generates visualizations and outputs

**Pipeline Stages:**
1. Load and preprocess images
2. Depth and pose estimation
3. Compute inter-view similarity
4. Compute per-image confidence maps
5. Generate/load SAM masks
6. Compute multi-view cost maps
7. Voxelize and extract features
8. Detect and merge objects
9. Associate objects and save masks
10. Generate visualizations

## Configuration

All parameters are specified in `scenediff_config.yml`:

```yaml
models:
  pi3:
    name: "yyfz233/Pi3"
  sam:
    checkpoint: "/path/to/sam_checkpoint.pth"
    points_per_side: 32
  dinov3:
    path: "/path/to/dinov3/"
    model_name: "dinov3_vith16plus"

processing:
  visible_percentage: 0.5
  occlusion_threshold: -0.02

costs:
  lambda_geo: 1.0
  lambda_dino: 0.3
  lambda_dino_region_match: 0.1

detection:
  min_detection_pixel: 100
  threshold_method: "otsu"  # or "max_entropy"
  
merging:
  geometry_distance_threshold_of_voxel_size: 2
  visual_threshold_ratio: 0.7
  object_similarity_threshold: 0.7
```

## Usage

### Basic Usage

```bash
# Process validation set
python predict_multiview_refactored.py \
    --config scenediff_config.yml \
    --splits val \
    --sets All

# Process test set with caching
python predict_multiview_refactored.py \
    --config scenediff_config.yml \
    --splits test \
    --sets Kitchen \
    --use_cache true \
    --save_cache true

# Enable point cloud visualization
python predict_multiview_refactored.py \
    --config scenediff_config.yml \
    --splits all \
    --sets Diverse \
    --vis_pc true
```

### Configuration Overrides

Command line arguments can override config file settings:

```bash
python predict_multiview_refactored.py \
    --config scenediff_config.yml \
    --output_dir my_output \
    --use_cache true \
    --vis_pc true
```

### Environment Variables

The submodules (Pi3, DINOv3, segment_ls) are automatically loaded from `submodules/` directory.
You only need to set the SAM checkpoint path:

```bash
export SAM_CHECKPOINT=/path/to/sam_checkpoint.pth
```

## Programmatic Usage

```python
import yaml
from modules.scene_diff import SceneDiff

# Load configuration
with open('scenediff_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
scene_diff = SceneDiff(config, device='cuda')

# Process a scene
file_list = [...]  # List of image paths
results = scene_diff.process_scene(
    file_list,
    img_1_length=10,
    img_2_length=10,
    output_dir='output/scene_name'
)

# Clean up
scene_diff.cleanup()
```

## Key Improvements Over Original

### 1. **Modular Architecture**
- Clear separation of geometry, segmentation, and semantic models
- Each model can be developed/tested independently
- Easy to swap out components (e.g., replace SAM with another segmentation model)

### 2. **Configuration Management**
- All parameters in one YAML file
- Environment variable support
- Easy to create experiment configurations
- Command line overrides

### 3. **Better Code Organization**
- Clear class responsibilities
- Reduced function parameter lists
- Less code duplication
- Easier to understand flow

### 4. **Improved Maintainability**
- Model lifecycle management (initialization, cleanup)
- Better error handling
- Progress tracking and logging
- Resource management

### 5. **Flexibility**
- Easy to add new cost functions
- Simple to modify processing pipeline
- Straightforward to add new models
- Clean extension points

## Output Structure

```
output/
└── scene_name/
    ├── object_masks.pkl           # Detected object masks
    ├── video_1_detection/         # Video 1 visualizations
    │   ├── video_1_frame_0_obj_0.png
    │   └── ...
    ├── video_2_detection/         # Video 2 visualizations
    │   ├── video_2_frame_0_obj_1.png
    │   └── ...
    ├── cost_map_merged.ply        # (if vis_pc=true)
    ├── object_instances_1.ply     # (if vis_pc=true)
    └── object_instances_2.ply     # (if vis_pc=true)
```

## Development

### Adding a New Model

1. Create new model file in `modules/`:
```python
# modules/my_new_model.py
class MyNewModel:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        # Initialize model
    
    def process(self, data):
        # Process data
        return results
    
    def cleanup(self):
        # Clean up resources
        pass
```

2. Add configuration to `scenediff_config.yml`:
```yaml
models:
  my_new_model:
    param1: value1
    param2: value2
```

3. Integrate in `scene_diff.py`:
```python
from .my_new_model import MyNewModel

class SceneDiff:
    def __init__(self, config, device='cuda'):
        # ... existing code ...
        self.my_model = MyNewModel(config, device)
```

### Adding a New Cost Function

1. Implement in appropriate model (e.g., `semantic_model.py`)
2. Add weight to config:
```yaml
costs:
  lambda_my_cost: 0.5
```
3. Update cost aggregation in `scene_diff.py`

## Performance Considerations

- **Caching**: Enable `use_cache` to reuse computed cost maps
- **Batch Processing**: Models process multiple views efficiently
- **Memory Management**: Automatic cleanup of models between scenes
- **GPU Utilization**: All heavy computations use CUDA when available

## Troubleshooting

### Out of Memory
- Reduce `subsample_size` in config
- Reduce `total_voxel_number`
- Process fewer frames at once

### Slow Processing
- Enable caching: `--use_cache true --save_cache true`
- Reduce `points_per_side` for SAM
- Increase `resample_rate` to process fewer frames

### Missing Dependencies
- Ensure Pi3, SAM, and DINOv3 paths are correct
- Check environment variables
- Verify all Python packages are installed

## Migration from Original Code

The refactored code maintains the same algorithm but with better structure:

| Original | Refactored |
|----------|-----------|
| `predict_changed_objects()` | `SceneDiff.process_scene()` |
| Inline Pi3 usage | `GeometryModel` |
| Inline SAM usage | `MaskModel` |
| Inline DINOv3 usage | `SemanticModel` |
| Command line args | YAML config + args |
| Global functions | Class methods |

To migrate:
1. Update imports to use new modules
2. Replace function calls with class methods
3. Use configuration file instead of many arguments
4. Update paths and environment variables

## License

[Your License Here]

## Citation

If you use this code, please cite:

```bibtex
@article{scenediff2024,
  title={SceneDiff: Multi-view Scene Change Detection},
  author={Your Name},
  year={2024}
}
```
