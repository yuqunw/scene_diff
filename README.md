<br />
<p align="center">

  <h1 align="center">SceneDiff: A Benchmark and Method for Multiview Object Change Detection</h1>

  <p align="center">
    <br />
   <a href="http://yuqunw.github.io"><strong>Yuqun Wu</strong></a>
    ·
    <a href="https://chih-hao-lin.github.io"><strong>Chih-hao Lin</strong></a>
    ·
    <a href="https://hungdche.github.io"><strong>Henry Che</strong></a>
    ·
    <a href="https://adititiwari19.github.io"><strong>Aditi Tiwari</strong></a>
    ·
    <a href="https://zouchuhang.github.io"><strong>Chuhang Zou</strong></a>
    ·
    <a href="https://shenlong.web.illinois.edu"><strong>Shenlong Wang</strong></a>
    ·
    <a href="http://dhoiem.cs.illinois.edu"><strong>Derek Hoiem</strong></a>
  </p>
</p>


  <p align="center">
    <a href='http://yuqunw.github.io/SceneDiff' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a>
    <a href='https://arxiv.org/abs/2409.18964'><img src='https://img.shields.io/badge/arXiv-2409.18964-b31b1b.svg'  alt='Arxiv'></a>
    <a href='https://huggingface.co/datasets/yuqunw/SceneDiff' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow?style=flat' alt='Dataset'></a>
    <a href='https://github.com/yuqunw/scenediff_annotator' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/GitHub-Data%20Annotator-black?style=flat&logo=github&logoColor=white' alt='Data Annotator'></a>
  </p>

</p>
<p align="center"> 
<img src="assets/teaser.png"/>
</p>

<br />

This repository contains the code for the paper [SceneDiff: A Benchmark and Method for Multiview Object Change Detection](http://yuqunw.github.io/SceneDiff). We investigate the problem of identifying objects that have been changed between a pair of captures of the same scene at different times, introducing the first object-level multiview change detection benchmark and a new training-free method.

## SceneDiff Benchmark

Download the SceneDiff benchmark dataset from [🤗 Hugging Face](https://huggingface.co/datasets/yuqunw/SceneDiff).

### Dataset Structure

```
scenediff_benchmark/
├── sequence_pair_1/
│   ├── original_video1.mp4   # Captured video before change (can have different extension)
│   ├── original_video2.mp4   # Captured video after change
│   ├── video1.mp4             # Video before change with annotation
│   ├── video2.mp4             # Video after change with annotation
│   └── segments.pkl           # Dense segmentations of changed objects (for evaluation)
├── sequence_pair_2/
└── ...
```

**About `segments.pkl`:** See the [detailed description here](https://github.com/yuqunw/scenediff_annotator?tab=readme-ov-file#segments-structure).

**Visualization:** For better visualization, place the downloaded dataset as the `results` directory of the [SceneDiff Annotator](https://github.com/yuqunw/scenediff_annotator) and open the review interface.

### Dataset Splits

The dataset is divided into validation and test splits, stored in the `splits/` directory:
- `splits/val_split.json` - Validation set with subsets: `varied`, `kitchen`
- `splits/test_split.json` - Test set with subsets: `varied`, `kitchen`

### Evaluation
We expect the method predictions have following structures:
```
output_dir/
├── sequence_pair_1/
│   └── object_masks.pkl           # Dense segmentations of changed objects (for evaluation)
├── sequence_pair_2/
└── ...
```
with `object_masks.pkl` following this structure:
```python
object_masks = {
    'H': int,                           # Image height
    'W': int,                           # Image width
    'video_1': {                        # Objects existing in video_1
        'object_id_1': {                # Integer ID for each detected object
            'frame_id_1': {             # Integer frame number
                'mask': RLE_Mask,       # Run-length encoded mask
                'cost': float           # Confidence score of the prediction
            },
            ...
        },
        ...
    },
    'video_2': {                        # Objects existing in video_2
        'object_id_1': {                # Integer ID for each detected object
            'frame_id_1': {             # Integer frame number
                'mask': RLE_Mask,       # Run-length encoded mask
                'cost': float           # Confidence score of the prediction
            },
            ...
        },
        ...
    }
}
```
Then the evaluation script can be run with:

```bash
python scripts/evaluate_multiview.py \
    --pred_dir ${OUTPUT_DIR} \
    --duplicate_match_threshold 2 \
    --per_frame_duplicate_match_threshold 2 \
    --splits val \
    --sets varied \
    --output_path ${OUTPUT_FILE_PATH} \
    --visualize False
```

**Arguments:**
- `--duplicate_match_threshold`: Tolerance for duplicate objects across frames (default: 2)
- `--per_frame_duplicate_match_threshold`: Tolerance for duplicate regions per frame (default: 2)
- `--splits`: Choose from `val`, `test`, or `all`
- `--sets`: Choose from `varied`, `kitchen`, or `All`
- `--visualize`: Set to `True` to save visualization outputs

**Output:** The evaluation results will be saved to `${OUTPUT_FILE_PATH}`
## Getting Started

### Installation

1. **Clone this repository with submodules:**
    ```bash
    git clone --recursive https://github.com/yuqunw/scenediff.git
    cd scenediff
    ```

2. **Create conda environment and install dependencies:**
    ```bash
    conda create -n scenediff python=3.10 -y
    conda activate scenediff
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # Install the pytorch fitting your nvcc version 
    pip install -r requirements.txt
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html # install torch_scatter
    ```

3. **Install submodules:**
    ```bash
    # Install segment-anything submodule
    cd submodules/segment-anything-langsplat-modified
    pip install -e .
    cd ../..
    ```

### Download Checkpoints

Download the Segment-Anything checkpoint:
```bash
bash checkpoints/download_sam_checkpoint.sh
```

Other checkpoints will be automatically downloaded in the first run.

## Quick Demo 

Run change detection on any two videos:

```bash
python scripts/demo.py \
    --config configs/scenediff_config.yml \
    --video1 path/to/video1.mp4 \
    --video2 path/to/video2.mp4 \
    --output output/demo
```

**Output:** The script generates point cloud visualizations including score maps and object segmentations for both videos in the specified output directory.

**Parameters:** You can modify parameters in `configs/scenediff_config.yml`. If the automatic threshold for change detection doesn't work well (score maps look correct but too many or few detections), you can manually set `detection.object_threshold` in the config file.   



## Predict on SceneDiff Benchmark

Run inference on all sequences in the benchmark:

```bash
python scripts/predict_multiview.py \
    --config configs/scenediff_config.yml \
    --splits val \
    --sets varied \
    --output_dir output/scenediff_benchmark
```

**Arguments:**
- `--splits`: Choose from `val`, `test`, or `all`
- `--sets`: Choose from `varied`, `kitchen`, or `All`
- `--output_dir`: Directory to save predictions
- Modify more arguments in the config file







<!-- ## Citation

If you find our work useful in your research, please cite:

```bibtex
@inproceedings{wu2024scenediff,
  title={SceneDiff: A Benchmark and Method for Multiview Object Change Detection},
  author={Wu, Yuqun and Lin, Chih-hao and Che, Henry and Tiwari, Aditi and Zou, Chuhang and Wang, Shenlong and Hoiem, Derek},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
``` -->

## Acknowledgement

We thank the great work from these repositories:
* [Segment-Anything](https://github.com/facebookresearch/segment-anything) and [LangSplat](https://github.com/minghanqin/LangSplat) for region segmentation
* [Pi3](https://github.com/yyfz/Pi3) for geometry estimation 
* [DINOv3](https://github.com/facebookresearch/dinov3) for appearance feature extraction

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
