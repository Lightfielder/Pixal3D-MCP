# Pixal3D-MCP

A standalone MCP (Model Context Protocol) server for running Pixal3D image-to-3D inference. This makes it possible to generate high-fidelity 3D assets from a single image, or a series of 3D assets from an image sequence.

## Overview

Pixal3D generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

**Key Features:**

- Single image to 3D mesh generation  
- Detailed geometry and PBR textures  
- Memory-efficient inference modes  
- MCP protocol support for desktop IDEs (Zed, etc.)  

## Repository Structure

```
Pixal3D-MCP/
├── src/pixal3d_mcp/       # Python package (MCP server implementation)
│   ├── __init__.py
│   └── server.py           # MCP server with 8 tools
├── assets/                 # Sample images for testing
│   └── images/             # Test images (0_img.png, etc.)
├── tests/                  # Unit Tests
│   └── __init__.py         # Init File
│   └── test_pixal3d_mcp.py # Test Script
├── pixal3d/                # Core Pixal3D library (from TencentARC/Pixal3D)
├── mcp.json                # MCP configuration for uvx
├── Rules.json              # Rules configuration
├── pyproject.toml          # Python package config
├── environment.yaml        # Conda environment definition
├── requirements.txt        # All dependencies (PyTorch, custom wheels, etc.)
├── inference.py            # Reference CLI script
└── README.md               # This document
└── LICENSE                 # License files
```

## Software Requirements

- **OS:** Linux (Rocky Linux 9.7, Ubuntu 22.04+, etc.)
- **GPU:** NVIDIA RTX series with CUDA 12.4 support
- **Python:** 3.10
- **CUDA:** 12.4
- **Compiler:** GCC 11

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Lightfielder/Pixal3D-MCP.git
cd Pixal3D-MCP
```

### 2. Install Dependencies

#### Create Conda Environment

Create an `environment.yaml` file with these contents:

```yaml
name: Pixal3D
channels:
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - uv
  - cuda=12.4.0
  - cuda-toolkit=12.4
  - gxx_linux-64=11
  - gcc_linux-64=11
  - cmake
  - ninja
  - cupy>=13.0.0
  - pybind11
  - setuptools
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu124
    - torch==2.6.0
    - torchvision==0.21.0
    - triton==3.2.0
    - transformers==4.57.3
    - timm==1.0.22
    - diffusers==0.37.1
    - accelerate==1.13.0
    - gradio
    - kornia==0.8.2
    - pillow==12.0.0
    - imageio==2.37.2
    - opencv-python-headless==4.12.0.88
    - trimesh==4.10.1
    - plyfile==1.1.3
    - tqdm==4.67.1
    - git+https://github.com/microsoft/MoGe.git
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl
    - https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl
```

#### Build and Install

```bash
# Linux dnf repos
sudo dnf update -y
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled crb
sudo dnf makecache

# Add git
sudo dnf install -y git

# Set up the Conda env
cd $HOME

# Add Miniconda virtual environment support
wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod -v +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Accept the Conda license terms for the repos
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Update Conda
conda update -n base -c defaults conda

# Create conda environment
cd $HOME/Pixal3D-MCP/
conda env create -f environment.yaml
conda activate Pixal3D

# Install Natten
cd $HOME
git clone --recursive https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
git checkout v0.17.3
export NATTEN_CUDA_ARCH="8.6"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8
export CUDA_HOME=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python setup.py build_ext --inplace
pip install --editable . --no-build-isolation

# Install the Pixal3D-MCP dependencies
cd $HOME/Pixal3D-MCP/
uv pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

### 3. Hugging Face Token Setup

Pixal3D uses the Bria AI "[RMGB v2.0](https://huggingface.co/briaai/RMBG-2.0)" model. You need to agree to the RMGB model's non-commercial license terms to be able to use it. If you have not done so then Pixal3D will report the following error in the terminal whent the inference.py script is launched:

```bash
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/briaai/RMBG-2.0.
403 Client Error. (Request ID: Root=1-6a050af0-56492db66f4b2c0c06cf7809;6ef33523-2961-4d57-8801-b28b5d8f92fe)

Cannot access gated repo for url https://huggingface.co/briaai/RMBG-2.0/resolve/main/config.json.
Access to model briaai/RMBG-2.0 is restricted and you are not in the authorized list. Visit https://huggingface.co/briaai/RMBG-2.0 to ask for access.
```

Use the Hugging Face website to generate a personal access token:

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) page to Generate a token
2. Click the "New token" button. Name the token something memorable like "Pixal3D". Click on the "Fine-grained" option near the top of the webpage. Do not select the "Read" option.
3. Under the "User permissions > Repositories" section enable the checkbox next to the "Read access to contents of all public gated repos you can access" option.
4. Click the "Generate token" button. Copy the token string that starts with "hf_".

Then run the following Hugging Face command from a Linux terminal session, to save the personal access token locally on the Linux system:

```bash
hf auth login
```

The output should lool like:

```bash
$ hf auth login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `hf auth whoami` to get more information or `hf auth logout` if you want to log out.
    Setting a new token will erase the existing one.
    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) y
Token is valid (permission: fineGrained).
The token `Pixal3D` has been saved to /home/vfx/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/vfx/.cache/huggingface/token
Login successful.
The current active token is: `Pixal3D`
```

### 4. Configure MCP Client

Add this to your MCP client settings (e.g., Zed IDE):

```json
{
  "mcpServers": {
    "pixal3d": {
      "command": "uvx",
      "args": [
        "git+https://github.com/Lightfielder/Pixal3D-MCP.git"
      ]
    }
  }
}
```

## How to activate the Conda Virtual Environment

```bash
conda activate Pixal3D
cd $HOME/Pixal3D/
```

## How to Remove the Pixal3D Repo Files / Conda Virtual Environment

```bash
cd $HOME

conda deactivate
conda env remove -n Pixal3D
sudo rm -rf $HOME/Pixal3D/
cd $HOME
```

## Direct CLI Usage

The MCP server wraps the underlying Pixal3D CLI:

```bash
conda activate Pixal3D
cd Pixal3D-MCP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inference.py --image assets/images/0_img.png --output ./output.glb --max_num_tokens 12288
```

The terminal should show a result like:

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb --max_num_tokens 12288
[SPARSE] Conv backend: flex_gemm; Attention backend: flash_attn_3
 _______  ___   __   __  _______  ___      _______  ______          __   __  _______  _______
|       ||   | |  |_|  ||   _   ||   |    |       ||      |        |  |_|  ||       ||       |
|    _  ||   | |       ||  |_|  ||   |    |___    ||  _    | ____  |       ||       ||    _  |
|   |_| ||   | |       ||       ||   |     ___|   || | |   ||____| |       ||       ||   |_| |
|    ___||   |  |     | |       ||   |___ |___    || |_|   |       |       ||      _||    ___|
|   |    |   | |   _   ||   _   ||       | ___|   ||       |       | ||_|| ||     |_ |   |
|___|    |___| |__| |__||__| |__||_______||_______||______|        |_|   |_||_______||___|    
     Pixal3D generates high-fidelity 3D assets from a single image
======================================================================

GPU Memory: 0.00GB allocated
[Pipeline] Loading from TencentARC/Pixal3D...
[ATTENTION] Using backend: flash_attn_3
/home/vfx/miniconda3/envs/Pixal3D/lib/python3.10/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
/home/vfx/miniconda3/envs/Pixal3D/lib/python3.10/site-packages/timm/models/registry.py:4: FutureWarning: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
[Pipeline] Loaded successfully
[Pipeline] low_vram = True
[Pipeline] Moving to GPU...
[Pipeline] On GPU
GPU Memory: 0.00GB allocated
[ImageCond] Loading SS model...
  SS model loaded
GPU Memory: 1.13GB allocated
[Inference] Processing image: assets/images/0_img.png
  Using default camera parameters
[ImageCond] Loading Shape 512 model...
  Loading NAF upsampler for shape 512...
Using cache found in /home/vfx/.cache/torch/hub/valeoai_NAF_main
GPU Memory: 2.27GB allocated
[ImageCond] Loading Shape 1024 model...
  Loading NAF upsampler for shape 1024...
Using cache found in /home/vfx/.cache/torch/hub/valeoai_NAF_main
GPU Memory: 3.40GB allocated
[ImageCond] Loading Texture model...
  Loading NAF upsampler for texture...
Using cache found in /home/vfx/.cache/torch/hub/valeoai_NAF_main
GPU Memory: 4.54GB allocated
[Inference] Running 3D generation pipeline...
Sampling sparse structure (proj): 100%|██████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.92it/s]
Sampling shape SLat (proj): 100%|████████████████████████████████████████████████████████████████████████████████| 12/12 [00:07<00:00,  1.58it/s]
Sampling HR shape SLat (proj, 1024): 100%|███████████████████████████████████████████████████████████████████████| 12/12 [00:44<00:00,  3.70s/it]
Sampling texture SLat (proj): 100%|██████████████████████████████████████████████████████████████████████████████| 12/12 [00:26<00:00,  2.17s/it]
[Memory] Unloading models...
GPU Memory: 0.39GB allocated
[Inference] Extracting GLB...
Cleaning mesh:  33%|███████████████████████████████▋                                                               | 2/6 [00:04<00:09,  2.49s/it]/home/vfx/miniconda3/envs/Pixal3D/lib/python3.10/site-packages/cumesh/remeshing.py:220: UserWarning: Using torch.cross without specifying the dim arg is deprecated.
Please either pass the dim explicitly or simply use torch.linalg.cross.
The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /pytorch/aten/src/ATen/native/Cross.cpp:62.)
  normals0 = torch.cross(mesh_vertices[atempt_triangles_0[:, 1]] - mesh_vertices[atempt_triangles_0[:, 0]], mesh_vertices[atempt_triangles_0[:, 2]] - mesh_vertices[atempt_triangles_0[:, 0]])
Finalizing mesh: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:12<00:00,  2.00s/it]
[Done] GLB saved to: ./output.glb
```

On NVIDIA RTX GPUs with limited memory you might get a terminal result from Pixal3D with a CUDA out of memory error like this:

```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 265.62 MiB is free. Process 3256 has 264.75 MiB memory in use. Including non-PyTorch memory, this process has 22.52 GiB memory in use. Of the allocated memory 22.13 GiB is allocated by PyTorch, and 39.62 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

## Available Tools

### Conda Environment Tools

| Tool | Description |
|------|-------------|
| `list_conda_envs` | List all available conda environments |
| `load_conda_env` | Load env vars from a conda environment |
| `get_conda_env_status` | Get current conda environment status |

### Pixal3D Inference Tools

| Tool | Description |
|------|-------------|
| `load_pipeline` | Load Pixal3D models into GPU memory |
| `unload_pipeline` | Free GPU memory |
| `run_inference` | Convert single image to 3D GLB |
| `run_inference_sequence` | Convert image sequence to 3D meshes |
| `preview_images` | Generate image thumbnail previews (fast) |
| `preview_model` | Render GLB models to preview PNGs |
| `preview_model_sequence` | Render GLB sequence to preview PNGs |
| `get_memory_status` | Check GPU memory usage |
| `cleanup_memory` | Run garbage collection |

## Typical Workflow

```json
// 1. Discover available conda environments
{"method": "tools/call", "params": {"name": "list_conda_envs", "arguments": {}}}

// 2. Load your conda environment
{"method": "tools/call", "params": {"name": "load_conda_env", "arguments": {"env_name": "Pixal3D"}}}

// 3. Load the Pixal3D pipeline
{"method": "tools/call", "params": {"name": "load_pipeline", "arguments": {}}}

// 4. Run inference
{"method": "tools/call", "params": {"name": "run_inference", "arguments": {"image": "assets/images/0_img.png", "output": "result.glb", "max_num_tokens": 12288}}}

// 5. Cleanup when done
{"method": "tools/call", "params": {"name": "unload_pipeline", "arguments": {}}}
```

## `run_inference` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | **yes** | - | Path to input image |
| `output` | string | no | `./output.glb` | Output GLB path |
| `seed` | integer | no | `42` | Random seed |
| `max_num_tokens` | integer | no | `4096` | Memory limit |

## `run_inference_sequence` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_path` | string | **yes** | - | Path to an image in the sequence |
| `output_template` | string | no | auto | Output mesh template (e.g., `mesh_####.glb`) |
| `frame_start` | integer | no | auto | Starting frame number |
| `frame_end` | integer | no | auto | Ending frame number |
| `frame_step` | integer | no | `1` | Process every N frames |
| `seed` | integer | no | `42` | Random seed |
| `max_num_tokens` | integer | no | `4096` | Memory limit |
| `auto_detect` | boolean | no | `true` | Auto-detect sequence pattern |

**Supported Sequence Patterns:**

| Style | Example | Padding |
|-------|---------|--------|
| Local files | `render_0001.png` | 4 digits |
| Maya | `render_0001.png` | 4 digits |
| Maya hash | `render.####.png` | variable |
| After Effects | `sequence_0001.png` | 4 digits |
| Fusion dot | `image.0001.png` | variable |
| printf | `image%04d.png` | 4 digits |
| Nuke/Foundry | `image.%04d.png` | 4 digits |
| Houdini/SideFX | `image.$F4.png` | 1-9 digits |
| IFL (Image File List) | `sequence.ifl` | per-line filenames |

**Network/Archive Paths:**

| Style | Example |
|-------|---------|
| HTTP/HTTPS | `https://example.com/image_0001.png` |
| FTP | `ftp://server.com/images/` |
| ZIP Archive | `/path/archive.zip/image.png` |
| TAR Archive | `/path/archive.tar.gz/frame_001.png` |

## `preview_images` Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|---------------|
| `image_paths` | array | no | - | Array of image paths to preview |
| `input_path` | string | no | - | Single image, URL, archive member, or pattern |
| `max_size` | integer | no | `256` | Max thumbnail dimension in pixels |

**Supported Input Types:**

- Local files: `image.png`, `frames_####.png`
- URLs: `https://example.com/image.png`
- Archives: `/path/archive.zip/member.png`
- IFL files: `sequence.ifl`

**Example:**

```json
// Preview a local sequence
{"method": "tools/call", "params": {"name": "preview_images", "arguments": {"input_path": "frames/image_0001.png"}}}

// Preview from URL
{"method": "tools/call", "params": {"name": "preview_images", "arguments": {"input_path": "https://example.com/image.png"}}}

// Preview from archive
{"method": "tools/call", "params": {"name": "preview_images", "arguments": {"input_path": "/data/archive.zip/frame_001.png"}}}

// Preview specific images
{"method": "tools/call", "params": {"name": "preview_images", "arguments": {"image_paths": ["a.png", "b.png", "c.png"]}}}
```

## `preview_model` Parameters

Render GLB 3D models to preview PNG images using the `3d-to-image` package.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `glb_paths` | array | no | - | Array of GLB file paths (supports URLs, archives) |
| `input_path` | string | no | - | Single GLB, URL, archive member, or pattern |
| `max_size` | integer | no | `512` | Max preview dimension in pixels |

**Supported Input Types:**

- Local files: `model.glb`, `meshes_####.glb`
- URLs: `https://example.com/model.glb`
- Archives: `/path/archive.zip/model.glb`

**Example:**

```json
// Preview a GLB sequence
{"method": "tools/call", "params": {"name": "preview_model", "arguments": {"input_path": "meshes_####.glb"}}}
// Preview from URL
{"method": "tools/call", "params": {"name": "preview_model", "arguments": {"input_path": "https://example.com/model.glb"}}}
// Preview from archive
{"method": "tools/call", "params": {"name": "preview_model", "arguments": {"input_path": "/data/archive.zip/model.glb"}}}
// Preview specific GLB files
{"method": "tools/call", "params": {"name": "preview_model", "arguments": {"glb_paths": ["a.glb", "b.glb"]}}}
```

## `preview_model_sequence` Parameters

Render a GLB sequence with optional frame range filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_path` | string | **yes** | - | Sequence pattern or IFL file |
| `frame_start` | integer | no | - | Starting frame number |
| `frame_end` | integer | no | - | Ending frame number |
| `max_size` | integer | no | `512` | Max preview dimension |

**Example:**
```json
{"method": "tools/call", "params": {"name": "preview_model_sequence", "arguments": {"input_path": "output_####.glb", "frame_start": 1, "frame_end": 10}}}
```

## Memory Management

If you encounter out of memory errors, try adjusting `max_num_tokens`:

| Value | Quality | Memory Usage |
|-------|---------|--------------|
| 1024 | Minimal | Lowest |
| 2048 | Lower | Low |
| 4096 | Default | Medium |
| 12288 | Higher | High |

Example with higher memory:
```json
{"method": "tools/call", "params": {"name": "run_inference", "arguments": {"image": "photo.png", "max_num_tokens": 12288}}}
```

## Sample Images

Sample images are included in the `assets/images/` folder:

- `0_img.png` - Classic test image  
- Various other test images in `.png`, `.webp`, `.jpg` formats. 

## Web Links

**GitHub Repos:**
- https://github.com/TencentARC/Pixal3D (Original)  
- https://github.com/Lightfielder/Pixal3D-MCP (This Repo)  
- https://github.com/microsoft/MoGe  
- https://github.com/microsoft/TRELLIS.2  

**Documentation:**
- https://ldyang694.github.io/projects/pixal3d/  
- https://huggingface.co/spaces/TencentARC/Pixal3D  
- https://wangrc.site/MoGePage/  

## Tools & Credits

**ASCII Art Banner:** The CLI banner in `inference.py` was generated using [TAAG](http://www.patorjk.com/software/taag/) - Text to ASCII Art Generator by Patrick Gillespie.

- Tool: http://www.patorjk.com/software/taag/
- Font: [Modular](http://www.patorjk.com/software/taag/#p=display&f=Modular&t=Pixal3D-MCP&x=none&v=4&h=4&w=80&we=false)
- Settings: Width=80, Spacing=4

## License

- **Pixal3D (upstream):** Non-commercial license - see TencentARC/Pixal3D for terms

