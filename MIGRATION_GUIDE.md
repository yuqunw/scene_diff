# Migration Guide: Original → Refactored SceneDiff

This guide helps you transition from the original `predict_multiview.py` to the refactored class-based implementation.

## Quick Comparison

| Aspect | Original | Refactored |
|--------|----------|-----------|
| **Structure** | Monolithic script | Modular classes |
| **Configuration** | 40+ command-line args | YAML config file |
| **Code Organization** | ~1500 lines in 1 file | 4 specialized modules |
| **Model Management** | Global instances | Class-based lifecycle |
| **Reusability** | Limited | High |
| **Testability** | Difficult | Easy |
| **Maintainability** | Complex | Clear |

## File Structure Comparison

### Original
```
SceneDiff/
├── predict_multiview.py  (1491 lines)
├── utils.py
└── config.yml  (for another purpose)
```

### Refactored
```
SceneDiff/
├── predict_multiview_refactored.py  (Entry point)
├── scenediff_config.yml             (Configuration)
├── modules/
│   ├── __init__.py
│   ├── geometry_model.py     (Depth & pose)
│   ├── mask_model.py        (Segmentation)
│   ├── semantic_model.py    (Features)
│   ├── scene_diff.py        (Orchestrator)
│   └── config_manager.py    (Config utils)
├── utils.py                 (Shared utilities)
├── example_usage.py         (Examples)
└── README_REFACTORED.md     (Documentation)
```

## Step-by-Step Migration

### Step 1: Install Dependencies

No new dependencies! The refactored code uses the same packages:

```bash
pip install -r requirements.txt
```

### Step 2: Create Configuration File

Convert your command-line arguments to a YAML configuration file:

**Original command:**
```bash
python predict_multiview.py \
    --output_dir output \
    --splits val \
    --sets All \
    --resample_rate 30 \
    --lambda_geo 1.0 \
    --lambda_dino 0.3 \
    --lambda_dino_region_match 0.1 \
    --occlusion_threshold -0.02 \
    --min_detection_pixel 100 \
    --threshold_method otsu \
    --geometry_distance_threshold_of_voxel_size 2 \
    --visual_threshold_ratio 0.7 \
    --object_similarity_threshold 0.7 \
    --use_cache False \
    --save_cache False \
    --vis_pc False
```

**Refactored config file (`my_config.yml`):**
```yaml
models:
  pi3:
    name: "yyfz233/Pi3"
  sam:
    checkpoint: "${SAM_CHECKPOINT}"
  dinov3:
    path: "${DINOV3_PATH}"

dataset:
  resample_rate: 30

processing:
  occlusion_threshold: -0.02

costs:
  lambda_geo: 1.0
  lambda_dino: 0.3
  lambda_dino_region_match: 0.1

detection:
  min_detection_pixel: 100
  threshold_method: "otsu"

merging:
  geometry_distance_threshold_of_voxel_size: 2
  visual_threshold_ratio: 0.7
  object_similarity_threshold: 0.7

cache:
  use_cache: false
  save_cache: false

visualization:
  vis_pc: false

output:
  output_dir: "output"
```

**Refactored command:**
```bash
python predict_multiview_refactored.py \
    --config my_config.yml \
    --splits val \
    --sets All
```

### Step 3: Update Your Scripts

#### Original Usage Pattern:
```python
# In original predict_multiview.py
model = Pi3.from_pretrained("yyfz233/Pi3").to(DEVICE).eval()

# Call main function with many arguments
predict_changed_objects(
    file_list, 
    img_1_length, 
    img_2_length, 
    output_dir,
    visible_percentage=0.5,
    args=args,
    model=model
)
```

#### Refactored Usage Pattern:
```python
import yaml
from modules import SceneDiff

# Load configuration
with open('scenediff_config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
scene_diff = SceneDiff(config, device='cuda')

# Process scene
results = scene_diff.process_scene(
    file_list,
    img_1_length,
    img_2_length,
    output_dir
)

# Clean up
scene_diff.cleanup()
```

### Step 4: Update Function Calls

| Original Function | Refactored Equivalent |
|-------------------|----------------------|
| `predict_changed_objects()` | `SceneDiff.process_scene()` |
| `load_and_preprocess_images()` | `SceneDiff.process_scene()` (internal) |
| `estimate_depth_and_poses()` | `GeometryModel.estimate_depth_and_poses()` |
| `load_or_generate_sam_masks()` | `MaskModel.load_or_generate_masks()` |
| `predict_dinov3_feat()` | `SemanticModel.extract_features()` |
| `compute_cost_maps_for_image()` | `SceneDiff._compute_cost_maps_for_image()` |

### Step 5: Access Individual Models (If Needed)

If you need direct access to specific models:

```python
from modules import GeometryModel, MaskModel, SemanticModel

# Initialize individual models
geometry_model = GeometryModel(config, device='cuda')
mask_model = MaskModel(config, device='cuda')
semantic_model = SemanticModel(config, device='cuda')

# Use them independently
depth_results = geometry_model.estimate_depth_and_poses(images)
masks = mask_model.load_or_generate_masks(file_list, H, W)
features = semantic_model.extract_features(image_np)

# Clean up
geometry_model.cleanup()
mask_model.cleanup()
semantic_model.cleanup()
```

## Configuration Parameter Mapping

### Command-Line Args → Config File

| Original Argument | Config Path |
|-------------------|-------------|
| `--lambda_geo` | `costs.lambda_geo` |
| `--lambda_dino` | `costs.lambda_dino` |
| `--lambda_dino_region_match` | `costs.lambda_dino_region_match` |
| `--occlusion_threshold` | `processing.occlusion_threshold` |
| `--visible_percentage` | `processing.visible_percentage` |
| `--min_detection_pixel` | `detection.min_detection_pixel` |
| `--threshold_method` | `detection.threshold_method` |
| `--object_threshold` | `detection.object_threshold` |
| `--geometry_distance_threshold_of_voxel_size` | `merging.geometry_distance_threshold_of_voxel_size` |
| `--visual_threshold_ratio` | `merging.visual_threshold_ratio` |
| `--object_similarity_threshold` | `merging.object_similarity_threshold` |
| `--use_cache` | `cache.use_cache` |
| `--save_cache` | `cache.save_cache` |
| `--vis_pc` | `visualization.vis_pc` |
| `--output_dir` | `output.output_dir` |
| `--resample_rate` | `dataset.resample_rate` |

## Common Migration Scenarios

### Scenario 1: Running Existing Experiments

**Original:**
```bash
python predict_multiview.py \
    --output_dir experiments/exp_001 \
    --splits val \
    --lambda_geo 1.5 \
    --lambda_dino 0.5
```

**Refactored:**

1. Create config file `experiments/exp_001_config.yml`:
```yaml
# ... base config ...
costs:
  lambda_geo: 1.5
  lambda_dino: 0.5
output:
  output_dir: experiments/exp_001
```

2. Run:
```bash
python predict_multiview_refactored.py \
    --config experiments/exp_001_config.yml \
    --splits val
```

### Scenario 2: Testing Different Parameters

**Original:** Create multiple scripts or use loops

**Refactored:** Create multiple config files

```bash
# Create config variants
cp scenediff_config.yml exp_geo_heavy.yml
# Edit exp_geo_heavy.yml to increase lambda_geo

cp scenediff_config.yml exp_semantic_heavy.yml
# Edit exp_semantic_heavy.yml to increase lambda_dino

# Run experiments
python predict_multiview_refactored.py --config exp_geo_heavy.yml --splits val
python predict_multiview_refactored.py --config exp_semantic_heavy.yml --splits val
```

### Scenario 3: Custom Processing Pipeline

**Original:** Modify the main function directly

**Refactored:** Extend or modify model classes

```python
from modules import SceneDiff

class MyCustomSceneDiff(SceneDiff):
    def _compute_cost_maps_for_image(self, *args, **kwargs):
        # Add custom cost computation
        result = super()._compute_cost_maps_for_image(*args, **kwargs)
        # Add your custom logic
        return result

# Use custom pipeline
scene_diff = MyCustomSceneDiff(config, device='cuda')
results = scene_diff.process_scene(...)
```

### Scenario 4: Batch Processing

**Original:** Run script multiple times

**Refactored:** Initialize once, process multiple scenes

```python
from modules import SceneDiff

scene_diff = SceneDiff(config, device='cuda')

for scene_info in scenes:
    try:
        results = scene_diff.process_scene(
            scene_info['files'],
            scene_info['n_frames_1'],
            scene_info['n_frames_2'],
            f"output/{scene_info['name']}"
        )
    except Exception as e:
        print(f"Error: {e}")
        continue

scene_diff.cleanup()
```

## Benefits You'll See

### 1. **Cleaner Code**
- Reduced function parameter lists (50+ params → config dict)
- Clear separation of concerns
- Better code organization

### 2. **Easier Experimentation**
- Quick parameter changes via config file
- Multiple configurations without code changes
- Easy comparison of experiments

### 3. **Better Resource Management**
- Automatic cleanup of models
- Reusable model instances
- Clear initialization/destruction

### 4. **Improved Debugging**
- Isolated model testing
- Clear error messages
- Better logging

### 5. **Enhanced Extensibility**
- Add new models easily
- Modify individual components
- Custom processing pipelines

## Backwards Compatibility

The original `predict_multiview.py` remains functional. You can:

1. Use both versions side-by-side
2. Gradually migrate code
3. Compare results between versions

To keep both:
```bash
# Original
python predict_multiview.py [args...]

# Refactored
python predict_multiview_refactored.py --config [config.yml]
```

## Troubleshooting

### Issue: "Module not found"

**Solution:** Ensure `modules/` directory has `__init__.py`:
```bash
touch modules/__init__.py
```

### Issue: "Configuration key not found"

**Solution:** Check your config file against `scenediff_config.yml` template. Use `ConfigManager.validate()`:
```python
from modules import ConfigManager
config_mgr = ConfigManager('my_config.yml')
errors = config_mgr.validate()
print(errors)
```

### Issue: "Results differ from original"

**Cause:** Default parameter differences

**Solution:** Ensure config matches your original command-line arguments exactly.

### Issue: "Out of memory"

**Solution:** Reduce parameters in config:
```yaml
processing:
  subsample_size: 500000  # Reduce from 1000000
  total_voxel_number: 150  # Reduce from 200
```

## Need Help?

1. Check `example_usage.py` for code examples
2. Review `README_REFACTORED.md` for detailed documentation
3. Compare output with original to verify correctness
4. Use `ConfigManager.print_summary()` to inspect configuration

## Summary

The refactored code provides:
- ✅ Same algorithm and results
- ✅ Better code organization
- ✅ Easier configuration management
- ✅ Improved extensibility
- ✅ Better resource management
- ✅ Enhanced maintainability

Start migrating today for a cleaner, more maintainable codebase!
