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
├── src/pixal3d_mcp/     # Python package (MCP server implementation)
│   ├── __init__.py
│   └── server.py          # MCP server with 8 tools
├── assets/               # Sample images for testing
│   └── images/           # Test images (0_img.png, etc.)
├── pixal3d/              # Core Pixal3D library (from TencentARC/Pixal3D)
├── mcp.json              # MCP configuration for uvx
├── pyproject.toml        # Python package config
├── environment.yaml      # Conda environment definition
├── requirements.txt      # All dependencies (PyTorch, custom wheels, etc.)
├── inference.py          # Reference CLI script
└── LICENSE               # License files
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
# Create conda environment
conda env create -f environment.yaml
conda activate Pixal3D

# Install Natten (required for attention)
git clone --recursive https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
git checkout v0.17.3
export NATTEN_CUDA_ARCH="8.6"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8
export CUDA_HOME=$CONDA_PREFIX
python setup.py build_ext --inplace
pip install --editable . --no-build-isolation
```

### 3. Hugging Face Token Setup

Pixal3D uses the Bria AI "RMGB v2.0" model. You need a Hugging Face token:

1. Generate a token at https://huggingface.co/settings/tokens
2. Run authentication:

```bash
hf auth login
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

## Direct CLI Usage

The MCP server wraps the underlying Pixal3D CLI:

```bash
conda activate Pixal3D
cd Pixal3D-MCP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python inference.py --image assets/images/0_img.png --output ./output.glb --max_num_tokens 12288
```

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

