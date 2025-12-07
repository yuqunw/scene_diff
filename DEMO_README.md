# SceneDiff Demo - Quick Start

Simple demo script for detecting scene changes between two videos with point cloud visualization.

## Quick Usage

```bash
# Basic usage - just provide two videos
python demo.py --video1 before.mp4 --video2 after.mp4
```

This will:
1. ✅ Extract frames from both videos
2. ✅ Run SceneDiff change detection
3. ✅ Generate point cloud visualizations
4. ✅ Save results to `demo_output/`

## Output Files

After running, you'll find:

```
demo_output/
├── temp_frames/           # Extracted frames (can be deleted after)
│   ├── video1/
│   └── video2/
└── results/
    ├── object_masks.pkl             # Detected object masks (main results)
    ├── cost_map_merged.ply          # Combined cost map visualization
    ├── cost_map_1.ply              # Video 1 cost map
    ├── cost_map_2.ply              # Video 2 cost map
    ├── rgb_pc_1.ply                # Video 1 RGB point cloud
    ├── rgb_pc_2.ply                # Video 2 RGB point cloud
    ├── object_instances_1.ply      # Detected objects in video 1
    └── object_instances_2.ply      # Detected objects in video 2
```

## Viewing Point Clouds

Open `.ply` files with any of these tools:

### Option 1: MeshLab (Recommended)
```bash
# Download from: https://www.meshlab.net/
meshlab demo_output/results/cost_map_merged.ply
```

### Option 2: CloudCompare
```bash
# Download from: https://www.cloudcompare.org/
CloudCompare demo_output/results/cost_map_merged.ply
```

### Option 3: Python + Open3D
```python
import open3d as o3d

# Load and visualize
pcd = o3d.io.read_point_cloud("demo_output/results/cost_map_merged.ply")
o3d.visualization.draw_geometries([pcd])
```

## Advanced Usage

### Custom Configuration

```bash
python demo.py \
    --video1 video1.mp4 \
    --video2 video2.mp4 \
    --config my_custom_config.yml \
    --output my_results
```

### Process More Frames (Higher Quality)

```bash
# Default: 1 frame every 30 frames
python demo.py --video1 video1.mp4 --video2 video2.mp4 --resample_rate 15

# Process every frame (slow but most accurate)
python demo.py --video1 video1.mp4 --video2 video2.mp4 --resample_rate 1
```

### Different Output Directory

```bash
python demo.py \
    --video1 video1.mp4 \
    --video2 video2.mp4 \
    --output experiment_001
```

## Command-Line Options

```
Required:
  --video1 PATH          Path to first video (before scene change)
  --video2 PATH          Path to second video (after scene change)

Optional:
  --config PATH          Configuration file (default: scenediff_config.yml)
  --output DIR           Output directory (default: demo_output)
  --resample_rate N      Sample 1 frame every N frames (default: 30)
                         Lower = more frames = slower but more accurate
```

## Understanding the Output

### Cost Maps (cost_map_*.ply)
- **Red/Hot colors**: High change probability
- **Blue/Cool colors**: Low change probability
- Use these to identify changed regions

### Object Instances (object_instances_*.ply)
- Each detected object has a unique color
- Objects present in both videos have the same color
- Use these to see what objects changed

### RGB Point Clouds (rgb_pc_*.ply)
- Original scene appearance
- Use as reference to understand the scene geometry

## Tips

### 1. Adjust Resample Rate Based on Your Needs

```bash
# Fast preview (every 30 frames)
python demo.py --video1 v1.mp4 --video2 v2.mp4 --resample_rate 30

# Balanced (every 15 frames)
python demo.py --video1 v1.mp4 --video2 v2.mp4 --resample_rate 15

# High quality (every 5 frames)
python demo.py --video1 v1.mp4 --video2 v2.mp4 --resample_rate 5
```

### 2. Check Point Clouds Immediately

```bash
# Run demo
python demo.py --video1 v1.mp4 --video2 v2.mp4

# Open results
meshlab demo_output/results/cost_map_merged.ply
```

### 3. Compare Different Settings

```bash
# Run with different configs
python demo.py --video1 v1.mp4 --video2 v2.mp4 --config config_geo_heavy.yml --output geo_results
python demo.py --video1 v1.mp4 --video2 v2.mp4 --config config_semantic_heavy.yml --output sem_results

# Compare point clouds
meshlab geo_results/results/cost_map_merged.ply
meshlab sem_results/results/cost_map_merged.ply
```

## Example Workflow

```bash
# 1. Quick preview with default settings
python demo.py --video1 before.mp4 --video2 after.mp4

# 2. Check results
meshlab demo_output/results/cost_map_merged.ply

# 3. If needed, run with higher quality
python demo.py \
    --video1 before.mp4 \
    --video2 after.mp4 \
    --resample_rate 15 \
    --output demo_output_hq

# 4. Compare results
meshlab demo_output_hq/results/cost_map_merged.ply
```

## Troubleshooting

### Out of Memory
```bash
# Reduce number of frames
python demo.py --video1 v1.mp4 --video2 v2.mp4 --resample_rate 60
```

### Too Slow
```bash
# Increase resample rate (fewer frames)
python demo.py --video1 v1.mp4 --video2 v2.mp4 --resample_rate 45
```

### Missing Dependencies
```bash
# Make sure all packages are installed
pip install -r requirements.txt

# Initialize submodules (Pi3, DINOv3, segment_ls are included as submodules)
git submodule update --init --recursive

# Set environment variable for SAM checkpoint
export SAM_CHECKPOINT=/path/to/sam_checkpoint.pth
```

### Video Format Issues
```bash
# Convert video to compatible format
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# Then run demo
python demo.py --video1 output.mp4 --video2 output2.mp4
```

## Next Steps

After running the demo:

1. **Inspect point clouds** to verify detection quality
2. **Load object masks** for downstream applications:
   ```python
   import pickle
   with open('demo_output/results/object_masks.pkl', 'rb') as f:
       results = pickle.load(f)
   ```
3. **Adjust configuration** if needed and rerun
4. **Use full pipeline** (`predict_multiview_refactored.py`) for batch processing

## Performance Notes

Typical processing times on GPU:
- **30 fps video, 10 seconds each, resample_rate=30**: ~2-3 minutes
- **30 fps video, 30 seconds each, resample_rate=30**: ~5-7 minutes
- **30 fps video, 60 seconds each, resample_rate=15**: ~15-20 minutes

Memory requirements:
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB GPU memory
- **For long videos**: 24GB+ GPU memory

## Questions?

- Check the main documentation: `README_REFACTORED.md`
- See configuration details: `scenediff_config.yml`
- View examples: `example_usage.py`
