<div align=center>
  <h1>
    Splatfinity
  </h1>
</div>

![Neural Rendering: 3D Gaussian Splatting Renderer](https://img.shields.io/badge/Neural%20Rendering-3D%20Gaussian%20Splatting%20Renderer-brown.svg)

Splatfinity is a Python/CUDA research renderer built around a Gaussian Splat scene. The current repository ships with a sample scene, `nubzuki_only_v2`, and renders a scripted 260-frame camera path with mirror and repeated “inception” duplication effects.

> The project includes two related workflows:
> - **Rendering** an existing Gaussian Splat `.ply` file using the custom PyTorch rasterizer in `src/renderer.py`.
> - **Preprocessing** raw camera images with Nerfstudio/COLMAP/Splatfacto to produce a Gaussian Splat `.ply` file that can be rendered by this project.

## Authors

[Chloe Xin DAI<sup>*</sup>](https://github.com/PhDinTimeManagement) <br>
[Pablo von Baum Garcia<sup>*</sup>](https://github.com/Pab-G) <br>
[Etienne GILLE<sup>*</sup>](https://github.com/etiennegille) <br>

<sup>*</sup> These authors contributed equally to this work.

## Project Demo
Note: The demo is high-resolution and may take a few moments to load.
<p align="center">
  <img src="media/video_demo.gif" width="480" alt="demo animation">
</p>

## Features

- Loads Nerfstudio-style Gaussian Splat `.ply` files containing positions, opacity, scale, rotation, and spherical-harmonic color coefficients.
- Renders a scripted camera path from `data/nerf_synthetic/nubzuki_only_v2/nubzuki_only_v2.json`.
- Exports individual PNG frames and an MP4 video through `imageio`/FFmpeg.
- Applies frame-dependent scene edits:
  - mirror reflection,
  - object duplication,
  - infinite-inception style repeated copies,
  - scene filtering and alignment tuned for the bundled scene.
- Includes an optional preprocessing pipeline using Nerfstudio commands:
  - HEIC/JPEG input conversion,
  - COLMAP pose estimation through `ns-process-data`,
  - Splatfacto training through `ns-train`,
  - Gaussian Splat export through `ns-export`.
- Includes RGB metric helpers for LPIPS, PSNR, and SSIM comparison between image directories.

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.10, C++/CUDA for the optional `simple-knn` extension |
| Data | Nerfstudio-style Gaussian Splat `.ply`, JSON camera path metadata |
| Video/image output | imageio, FFmpeg, torchvision image utilities |
| Preprocessing | Nerfstudio, COLMAP, Splatfacto, gsplat |
| Rendering | PyTorch, torchvision, custom tiled Gaussian Splat rasterizer |
| Metrics | torchmetrics, LPIPS, PSNR, SSIM |
| CLI | tyro for `render.py`, argparse for `preprocess.py` |

## Repository Structure
Note: Local editor metadata and generated build artifacts are not required to run the project and should be excluded from a clean source distribution.
```text
Splatfinity
├── README.md
├── LICENSE
├── render.py                          # Main renderer CLI
├── preprocess.py                      # Optional Nerfstudio preprocessing pipeline
├── convertor.py                       # Image conversion helper used by preprocess.py
├── path_creator.py                    # Regenerates the hard-coded 260-frame camera path
├── data/
│   ├── nubzuki_only_v2.ply            # Bundled sample Gaussian Splat scene
│   └── nerf_synthetic/
│       └── nubzuki_only_v2/
│           └── nubzuki_only_v2.json   # Camera intrinsics and scripted path
├── media/                             # Demo images and GIF
├── rendering_outputs/
│   └── nubzuki_only_v2/               # Bundled generated render outputs
├── simple-knn/                        # Optional CUDA KNN extension source
└── src/
    ├── camera.py                      # Camera dataclass
    ├── constants.py                   # Rendering constants, including USE_HALF
    ├── renderer.py                    # PyTorch Gaussian Splat rasterizer and scene effects
    ├── rgb_metrics.py                 # LPIPS/PSNR/SSIM helpers
    ├── scene.py                       # Scene dataclass
    └── sh.py                          # Spherical harmonics utilities
```

## Prerequisites

### Preprocessing

The preprocessing workflow additionally requires:

- Nerfstudio CLI tools: `ns-process-data`, `ns-train`, and `ns-export`.
- COLMAP.
- A compatible CUDA/PyTorch/Nerfstudio environment.
- Raw input images in a local directory such as `camera_input_pics/`.

### Rendering

Rendering is intended for a CUDA-capable NVIDIA GPU. Although `render.py` exposes `--device-type cpu`, the current renderer allocates several tensors directly on CUDA in `src/renderer.py`, so the CPU path is not fully supported.

Recommended baseline:

- Linux or another CUDA-supported development environment.
- NVIDIA GPU with a working CUDA runtime.
- Python 3.10.
- FFmpeg support for MP4 output through `imageio[ffmpeg]`.

## Installation

### 1. Create a Rendering Environment

```bash
conda create --name splatfinity python=3.10 -y
conda activate splatfinity
```

Install PyTorch and torchvision. Choose the PyTorch command that matches your CUDA runtime. The original project notes used PyTorch `2.5.1` and torchvision `0.20.1`:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining Python dependencies used by the renderer and utility scripts:

```bash
pip install numpy tqdm imageio[ffmpeg] plyfile tyro==0.6.0 jaxtyping==0.2.36 typeguard==2.13.3
pip install pillow pillow-heif torchmetrics[image]
```

`pillow-heif` is required by `preprocess.py` because it imports `convertor.py` at startup, even when conversion is not the selected step.

### 2. Create a Preprocessing Environment

Use a separate environment if Nerfstudio dependency resolution conflicts with the rendering environment.

```bash
conda create -n splatfinity-preprocess -c conda-forge python=3.10 -y
conda activate splatfinity-preprocess

pip install nerfstudio pillow-heif tqdm
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0

conda install -c conda-forge colmap ffmpeg -y
conda install \
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  pytorch-cuda=11.8 \
  -c pytorch -c nvidia \
  -y
```

### 3. Build the bundled `simple-knn` extension

The current `render.py` path does not import `simple_knn` directly. The source is included for CUDA KNN functionality and can be built manually when needed:

```bash
pip install ./simple-knn
```

Build requirements include PyTorch, a CUDA compiler, and a compatible host compiler. Rebuild this extension locally rather than committing platform-specific `.so`, `.o`, `build/`, or `egg-info/` artifacts.

## Preprocessing Workflow

Preprocessing is optional if you only want to render the bundled scene. Use it when creating a new Gaussian Splat from raw camera images.

### Supported GPU Presets

`preprocess.py` sets CUDA build variables based on `--GPU`:

| `--GPU` value | `TORCH_CUDA_ARCH_LIST` | Extra behavior |
| --- | --- | --- |
| `A100` | `8.0` | Default. |
| `RTX4090` | `8.9` | For RTX 4090. |
| `A6000` | `8.6` | Also prepends `/usr/local/cuda-12.2/bin` and `/usr/local/cuda-12.2/lib64`. |

### Automated Preprocessing (Step 1-6)

```bash
python preprocess.py \
  --GPU RTX4090 \
  --convert \
  --colmap \
  --train \
  --ply \
  --scene_name nubzuki_only_v2 \
  --input_dir camera_input_pics
```

- Scene_name for our scene is: nubzuki_only_v2
- input_dir: you can download the input pictures converted or unconverted: https://drive.google.com/drive/folders/1zehi2jmguVz13y1qFWzGgW9K9I2LFjAj

This command performs the following requested stages:

1. Convert Camera Input Images
    ```bash 
    python convertor.py
    ```
   - The camera input pictures are gitignored: `camera_input_pics/`, `camera_input_pics_converted/`.
   - Link to Download: https://drive.google.com/drive/folders/1uLroHJXeJLAx3mO67CzmIuwsV-WOKOWP?usp=sharing


2. Generate Camera Poses with COLMAP

   ```bash
   ns-process-data images --data ./camera_input_pics_converted --output-dir ./processed_images_colmap
   ```

   - The COLMAP–processed images are gitignored: `processed_images_colmap/`
   - Link to Download: https://drive.google.com/drive/folders/15lzamNo2JjFHmjq44iJfDnQnInIL363u?usp=sharing


3. GPU Build Configuration
   - Option 1: for RTX 4090
    ```bash 
    export MAX_JOBS=1
    export TORCH_CUDA_ARCH_LIST="8.9"
    ```

   - Option 2: for A100
    ```bash 
    export MAX_JOBS=1
    export TORCH_CUDA_ARCH_LIST="8.0"
    ```

   - Option 3: for A6000
    ```bash 
    export MAX_JOBS=1
    export TORCH_CUDA_ARCH_LIST="8.6"
    export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```


4. Train with Splatfacto

   ```bash
   ns-train splatfacto --data ./processed_images_colmap
   ```

   - The Splatfacto trained model outputs are gitignored: `outputs/`
   - Link to Download: https://drive.google.com/drive/folders/1AniBSBACpUI5WL0LCbVa4b_IqVVTidcM?usp=sharing 


5. Dump out the Gaussian Splat
   ```bash
   ns-export gaussian-splat \
     --load-config ./outputs/processed_images_colmap/splatfacto/{timestamp}/config.yml \
     --output-dir ./data
   ```


6. Rename ply File
    ```bash
    mv export/splat/splat.ply export/splat/{rename}.ply
    ```
   - The Gaussian Splat is gitignored: `export/splat/`
   - Link to Download: https://drive.google.com/drive/folders/1U4meVGaYqIFF0W6BxDdylooCOpICI9cx?usp=sharing
   - Copy ply file to `data/` directory
    ```bash
    cp export/splat/{rename}.ply data/
    ```


Add `--remove_all` only when you intentionally want to delete intermediate folders after export:

```bash
python preprocess.py \
  --GPU RTX4090 \
  --convert \
  --colmap \
  --train \
  --ply \
  --scene_name nubzuki_only_v2 \
  --input_dir camera_input_pics \
  --remove_all
```

Warning: `--remove_all` deletes `camera_input_pics_converted/`, `processed_images_colmap/`, `outputs/`, and the directory passed through `--input_dir` if it exists. Back up original camera inputs before using it.

## Mirror Rendering

### Rendering Pipeline

At a high level, the renderer performs the following steps:

```text
Gaussian Splat PLY + camera JSON
        │
        ├── load_scene() / load_ply()
        ├── load_camera_params()
        ├── scene filtering, alignment, mirroring, and duplication
        ├── projection to NDC and screen coordinates
        ├── 3D covariance → 2D covariance projection
        ├── tiled rasterization and alpha blending
        └── PNG frames + MP4 video
```

The frame-dependent effect schedule is defined in `idx_manager()` inside `src/renderer.py`:

| Frame range | Effect path |
| --- | --- |
| `0–65` | `mirror(...)` |
| `66–94` | `infiniteinception_back(..., 1)` |
| `95–117` | `infiniteinception_front(..., 12)` |
| `118–200` | `infiniteinception(..., 12)` |
| `201–259` | `infiniteinception_front(..., 12)` |

### Regenerate the Camera Path

`path_creator.py` overwrites the bundled camera-path JSON while preserving the existing metadata keys other than `frames`:

```bash
python path_creator.py
```

This script is hard-coded to read and write:

```text
data/nerf_synthetic/nubzuki_only_v2/nubzuki_only_v2.json
```

It creates 260 scripted camera frames named like `images/circle_00000.jpg`. The renderer uses the transform matrices and metadata; it does not load those image paths during rendering.

### Render the bundled scene

The repository already includes the sample scene and camera path:

- `data/nubzuki_only_v2.ply`
- `data/nerf_synthetic/nubzuki_only_v2/nubzuki_only_v2.json`

Run the renderer from the repository root:

```bash
python render.py
```

By default, this writes outputs to:

```text
rendering_outputs/nubzuki_only_v2/
├── r_0.png
├── r_1.png
├── ...
├── r_259.png
└── video.mp4
```

The camera JSON contains 260 frames. Because `src/constants.py` sets `USE_HALF = True`, the 1200×1200 camera metadata is rendered at 600×600.

- The mirror rendering output file is gitignored: `mirror_rendering_outputs/`
- Link to Download: https://drive.google.com/drive/folders/1ZRAbBIHspBpg4I_Ix_4AJKGpS3aoricH?usp=sharing

## Data Expectations

### Gaussian Splat PLY

`render.py` expects a `.ply` file at:

```text
data/{scene_type}.ply
```

For the bundled scene this is:

```text
data/nubzuki_only_v2.ply
```

The loader reads the following Nerfstudio-style fields from the first PLY element:

- `x`, `y`, `z`
- `opacity`
- `f_dc_*`
- `f_rest_*`
- `scale_*`
- `rot_*`

The bundled PLY contains 262,140 vertices and was generated by Nerfstudio according to its PLY header.

### Camera JSON

`render.py` expects camera metadata under:

```text
data/nerf_synthetic/{scene_type}/nubzuki_only_v2.json
```

For the current scene this resolves to:

```text
data/nerf_synthetic/nubzuki_only_v2/nubzuki_only_v2.json
```

The JSON includes image dimensions, focal/camera parameters, and a `frames` array of transform matrices.

## Metrics Helpers

`src/rgb_metrics.py` provides helper functions for comparing rendered images with target images:

- `compute_lpips_between_directories(pred_dir, target_dir)`
- `compute_psnr_between_directories(pred_dir, target_dir)`
- `compute_ssim_between_directories(pred_dir, target_dir)`

These helpers assume matching filenames between directories and currently run on CUDA through `.cuda()` calls. There is no command-line wrapper for metrics in the current repository.

## Acknowledgements

- `src/renderer.py` notes that the rasterizer implementation is based on `torch-splatting`.
- `src/sh.py` includes spherical-harmonics utility code credited to the PlenOctree authors.
- `simple-knn/` includes source files with GraphDECO/Inria copyright headers.
- The preprocessing workflow relies on Nerfstudio, COLMAP, Splatfacto, and gsplat.
