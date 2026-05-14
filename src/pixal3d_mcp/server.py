"""
Pixal3D MCP Server

An MCP server for running Pixal3D image-to-3D inference.
"""

import gc
import json
import os
import re
import sys
from pathlib import Path

import torch

# Add the parent directory to sys.path to import pixal3d
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Global state
PIPELINE = None
PIPELINE_LOADED = False
CONDA_ENV_NAME = "Pixal3D"  # Default conda environment name


def load_conda_env(env_name: str = "Pixal3D") -> dict:
    """Load environment variables from a conda environment.

    Reads the conda environment's variables and applies them to os.environ
    for the current process. This allows the MCP server to access packages
    installed in a specific conda environment when launched from a desktop app
    like Zed that doesn't have the environment pre-activated.

    Args:
        env_name: Name of the conda environment (default: "Pixal3D")

    Returns:
        dict with status and environment path, or error message
    """
    global CONDA_ENV_NAME
    import subprocess

    try:
        # Get environment info as JSON
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        env_data = json.loads(result.stdout)

        # Find the environment path
        env_path = None
        for env_info in env_data.get("envs", []):
            # env_info can be a string path or a dict depending on conda version
            if isinstance(env_info, str):
                if env_info.endswith(f"/{env_name}") or env_info.endswith(
                    f"\\{env_name}"
                ):
                    env_path = env_info
                    break
            elif isinstance(env_info, dict):
                if env_info.get("name") == env_name:
                    env_path = env_info.get("path")
                    break

        if not env_path:
            # Try direct path lookup
            result = subprocess.run(
                ["conda", "env", "list"], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if env_name in line and not line.strip().startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = parts[-1]
                        break

        if not env_path:
            return {
                "status": "error",
                "message": f"Conda environment '{env_name}' not found. Run 'conda env list' to see available environments.",
            }

        # Export environment variables as JSON
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "printenv", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )

        env_vars = json.loads(result.stdout) if result.stdout.strip() else {}

        # Apply environment variables to os.environ
        for key, value in env_vars.items():
            os.environ[key] = value

        # Also set CONDA-specific variables
        os.environ["CONDA_PREFIX"] = env_path
        os.environ["CONDA_DEFAULT_ENV"] = env_name
        os.environ["CONDA_ENV_NAME"] = env_name

        # Update sys.path to include the conda env's site-packages
        env_python = Path(env_path) / "bin" / "python"
        if env_python.exists():
            result = subprocess.run(
                [
                    str(env_python),
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                site_packages = result.stdout.strip()
                if site_packages and site_packages not in sys.path:
                    sys.path.insert(0, site_packages)

        CONDA_ENV_NAME = env_name

        return {
            "status": "success",
            "message": f"Loaded conda environment: {env_name}",
            "env_path": env_path,
            "vars_loaded": len(env_vars),
            "env_name": env_name,
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Conda command failed: {e.stderr if e.stderr else str(e)}",
        }
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Failed to parse conda output: {e}"}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load conda environment: {str(e)}",
        }


def get_conda_env_status() -> dict:
    """Get current conda environment status."""
    return {
        "status": "success",
        "current_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "not set"),
        "env_name": CONDA_ENV_NAME,
        "python_path": sys.executable,
    }


def list_conda_envs() -> dict:
    """List all available conda environments.

    Runs 'conda env list' and parses the output to return
    a list of all available conda environment names and paths.

    Returns:
        dict with status and list of environments
    """
    import subprocess

    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )

        envs = []
        for line in result.stdout.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse: "env_name" or "env_name /full/path/to/env"
            parts = line.split()
            if len(parts) >= 1:
                env_name = parts[0]
                env_path = parts[-1] if len(parts) > 1 else ""

                # Skip the 'base' environment entry if it's just the root
                if env_name == "base" and r"\conda" in env_path:
                    continue

                envs.append({"name": env_name, "path": env_path})

        return {"status": "success", "environments": envs, "count": len(envs)}

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Conda command failed: {e.stderr if e.stderr else str(e)}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list conda environments: {str(e)}",
        }


def clean_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")


def load_pipeline():
    """Load the Pixal3D pipeline."""
    global PIPELINE, PIPELINE_LOADED

    if PIPELINE_LOADED:
        return PIPELINE

    print("[Pipeline] Loading from TencentARC/Pixal3D...")

    # Set environment variables for memory efficiency
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.7"
    )
    os.environ["ATTN_BACKEND"] = "flash_attn_3"

    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    PIPELINE = Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")

    if PIPELINE is None:
        raise RuntimeError("Failed to load pipeline")

    print("[Pipeline] Loaded successfully")
    PIPELINE.low_vram = True
    PIPELINE.to("cuda")

    # Load SS model
    print("[ImageCond] Loading SS model...")
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    ss_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    }
    PIPELINE.image_cond_model_ss = DinoV3ProjFeatureExtractor(**ss_config)
    PIPELINE.image_cond_model_ss.to("cuda")
    PIPELINE.image_cond_model_ss.eval()
    print("  SS model loaded")
    clean_memory()

    # Load shape model 512
    print("[ImageCond] Loading Shape 512 model...")
    shape_512_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    }
    PIPELINE.image_cond_model_shape_512 = DinoV3ProjFeatureExtractor(**shape_512_config)
    PIPELINE.image_cond_model_shape_512.to("cuda")
    PIPELINE.image_cond_model_shape_512.eval()

    if (
        hasattr(PIPELINE.image_cond_model_shape_512, "use_naf_upsample")
        and PIPELINE.image_cond_model_shape_512.use_naf_upsample
    ):
        print("  Loading NAF upsampler for shape 512...")
        PIPELINE.image_cond_model_shape_512._load_naf()
    clean_memory()

    # Load shape model 1024
    print("[ImageCond] Loading Shape 1024 model...")
    shape_1024_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    }
    PIPELINE.image_cond_model_shape_1024 = DinoV3ProjFeatureExtractor(
        **shape_1024_config
    )
    PIPELINE.image_cond_model_shape_1024.to("cuda")
    PIPELINE.image_cond_model_shape_1024.eval()

    if (
        hasattr(PIPELINE.image_cond_model_shape_1024, "use_naf_upsample")
        and PIPELINE.image_cond_model_shape_1024.use_naf_upsample
    ):
        print("  Loading NAF upsampler for shape 1024...")
        PIPELINE.image_cond_model_shape_1024._load_naf()
    clean_memory()

    # Load texture model
    print("[ImageCond] Loading Texture model...")
    tex_config = {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    }
    PIPELINE.image_cond_model_tex_1024 = DinoV3ProjFeatureExtractor(**tex_config)
    PIPELINE.image_cond_model_tex_1024.to("cuda")
    PIPELINE.image_cond_model_tex_1024.eval()

    if (
        hasattr(PIPELINE.image_cond_model_tex_1024, "use_naf_upsample")
        and PIPELINE.image_cond_model_tex_1024.use_naf_upsample
    ):
        print("  Loading NAF upsampler for texture...")
        PIPELINE.image_cond_model_tex_1024._load_naf()
    clean_memory()

    PIPELINE_LOADED = True
    return PIPELINE


def unload_pipeline():
    """Unload the pipeline and free memory."""
    global PIPELINE, PIPELINE_LOADED

    if PIPELINE is not None:
        print("[Memory] Unloading models...")
        for attr in [
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ]:
            if hasattr(PIPELINE, attr) and getattr(PIPELINE, attr) is not None:
                model = getattr(PIPELINE, attr)
                model.to("cpu")
                del model
                setattr(PIPELINE, attr, None)

        PIPELINE = None
        PIPELINE_LOADED = False
        clean_memory()


def generate_thumbnail(image_path: str, max_size: int = 256) -> dict:
    """Generate a base64-encoded thumbnail of an image.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension size for the thumbnail

    Returns:
        dict with thumbnail info including:
        - thumbnail: Base64-encoded JPEG thumbnail
        - width: Original image width
        - height: Original image height
        - thumbnail_width: Thumbnail width
        - thumbnail_height: Thumbnail height
        - format: Image format
    """
    import base64
    import io

    from PIL import Image

    try:
        # Open and get original dimensions
        img = Image.open(image_path)
        original_width, original_height = img.size

        # Convert to RGB if necessary
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Calculate thumbnail dimensions maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if original_width > original_height:
            thumb_width = max_size
            thumb_height = int(max_size / aspect_ratio)
        else:
            thumb_height = max_size
            thumb_width = int(max_size * aspect_ratio)

        # Create thumbnail
        thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        # Encode to JPEG
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "thumbnail": thumbnail_b64,
            "width": original_width,
            "height": original_height,
            "thumbnail_width": thumb_width,
            "thumbnail_height": thumb_height,
            "format": img.format or "JPEG",
            "size_bytes": len(thumbnail_b64),
        }

    except Exception as e:
        return {
            "error": str(e),
            "image_path": image_path,
        }


def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    max_num_tokens: int = 4096,
    return_thumbnail: bool = False,
):
    """Run Pixal3D inference on an image.

    Args:
        image_path: Path to input image
        output_path: Path to output GLB
        seed: Random seed
        max_num_tokens: Memory limit
        return_thumbnail: Whether to return a base64-encoded thumbnail

    Returns:
        dict with output path and optionally thumbnail data
    """
    import numpy as np
    import o_voxel
    from PIL import Image

    clean_memory()

    # Generate thumbnail if requested
    thumbnail_data = None
    if return_thumbnail:
        print(f"[Thumbnail] Generating preview for: {image_path}")
        thumbnail_data = generate_thumbnail(image_path)

    # Load pipeline
    pipeline = load_pipeline()

    # Preprocess image
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    image_preprocessed = pipeline.preprocess_image(img)

    # Default camera parameters
    camera_params = {"camera_angle_x": 0.8575, "distance": 1.8, "mesh_scale": 1.0}
    print(f"  Using default camera parameters")

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    shape_sampler_override = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    tex_sampler_override = {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }

    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=200000,
        texture_size=1024,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    # Apply rotation
    rot = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")

    result = {"output_path": output_path}

    # Include thumbnail if requested
    if return_thumbnail and thumbnail_data:
        if "error" not in thumbnail_data:
            result["thumbnail"] = thumbnail_data
        else:
            result["thumbnail_error"] = thumbnail_data.get("error")

    return result


def detect_ifl_pattern(ifl_path: str) -> dict:
    """Detect pattern from an IFL (Image File List).

    An IFL file is a plain text file with one image filename per line.

    Args:
        ifl_path: Path to the .ifl file

    Returns:
        dict with pattern info including:
        - is_ifl: True
        - files: List of image paths from the IFL
        - directory: Directory of the IFL file
        - extension: File extension of images
        - is_sequence: True (IFL implies a sequence)
    """
    directory = os.path.dirname(ifl_path)
    files = []
    extensions = set()

    try:
        with open(ifl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Handle relative paths in IFL
                if os.path.isabs(line):
                    files.append(line)
                else:
                    files.append(os.path.join(directory, line))
                # Track extension
                _, ext = os.path.splitext(line)
                if ext:
                    extensions.add(ext.lower())
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read IFL file: {e}",
            "is_ifl": True,
        }

    if not files:
        return {
            "status": "error",
            "message": "IFL file is empty or contains no valid image paths",
            "is_ifl": True,
        }

    return {
        "status": "success",
        "is_ifl": True,
        "ifl_path": ifl_path,
        "files": files,
        "directory": directory,
        "extension": list(extensions)[0] if len(extensions) == 1 else extensions,
        "total_frames": len(files),
        "is_sequence": True,
        "pattern_type": "ifl",
    }


def detect_sequence_pattern(input_path: str) -> dict:
    """Detect image sequence pattern from a filename or path.

    Supports patterns inspired by:
    - Maya: render_0001.png, render.####.png, render.####.ext
    - After Effects: sequence_0001.png, sequence_####.png
    - Fusion: image_0001.png, image.0001.png
    - Nuke/Foundry: image.%04d.png (printf-style)
    - Houdini/SideFX: image.$F4.png (variable padding $F1-$F9)
    - IFL (Image File List): .ifl text file with one filename per line

    Args:
        input_path: Path to an image file or sequence pattern

    Returns:
        dict with pattern info including:
        - base_filename: Filename without frame number and extension
        - padding: Frame number padding (e.g., 4 for "0001")
        - frame_start: Starting frame number (if detectable)
        - frame_end: Ending frame number (if detectable)
        - extension: File extension (e.g., ".png")
        - sequence_pattern: Formatted pattern string
        - is_sequence: Whether this looks like a sequence
        - is_ifl: Whether this is an IFL file
    """
    import re

    # Check if input is an IFL file
    if input_path.lower().endswith(".ifl"):
        return detect_ifl_pattern(input_path)

    filename = os.path.basename(input_path)
    directory = os.path.dirname(input_path)

    # Common frame number patterns
    # Pattern 1: underscore padding (e.g., image_0001.png)
    # Pattern 2: dot padding (e.g., image.0001.png)
    # Pattern 3: hash padding (e.g., image####.png or image.####.png)
    # Pattern 4: %04d printf style (e.g., image%04d.png) - Nuke/Foundry
    # Pattern 5: $F4 Houdini style (e.g., image.$F4.png)

    # Try to match various patterns
    patterns = [
        # Underscore + padding: image_0001.png, image_001.jpg
        (r"^(.+?)_(\d+)(\.\w+)$", "underscore"),
        # Dot + padding: image.0001.png, render.0001.ext
        (r"^(.+?)\.(\d+)(\.\w+)$", "dot"),
        # Hash padding: image####.png, image.####.png
        (r"^(.+?)#+(?:\.(\w+))?$", "hash"),
        # %04d printf style: image%04d.png - Nuke/Foundry
        (r"^(.+?)%0(\d+)d(\.\w+)$", "printf"),
        # $F4 Houdini style: image.$F4.png, render_$F3.png
        (r"^(.+?)\.\$(F\d+)(\.\w+)$", "houdini"),
        (r"^(.+?)_\$(F\d+)(\.\w+)$", "houdini_underscore"),
    ]

    result = {
        "base_filename": filename,
        "directory": directory,
        "padding": 0,
        "frame_start": None,
        "frame_end": None,
        "extension": "",
        "sequence_pattern": input_path,
        "is_sequence": False,
        "matches": [],
    }

    for pattern, pattern_type in patterns:
        match = re.match(pattern, filename)
        if match:
            if pattern_type == "underscore":
                base, frame_num, ext = match.groups()
                result["base_filename"] = base + "_XXXX" + ext
                result["padding"] = len(frame_num)
                result["extension"] = ext
                result["frame_start"] = int(frame_num)
                result["is_sequence"] = True
                result["sequence_pattern"] = os.path.join(
                    directory, base + "_%0" + str(len(frame_num)) + "d" + ext
                )

            elif pattern_type == "dot":
                base, frame_num, ext = match.groups()
                result["base_filename"] = base + ".XXXX" + ext
                result["padding"] = len(frame_num)
                result["extension"] = ext
                result["frame_start"] = int(frame_num)
                result["is_sequence"] = True
                result["sequence_pattern"] = os.path.join(
                    directory, base + ".%0" + str(len(frame_num)) + "d" + ext
                )

            elif pattern_type == "hash":
                base, ext = match.groups()
                if ext is None:
                    ext = ""
                # Count hashes
                hash_match = re.search(r"#+", filename)
                if hash_match:
                    hash_count = len(hash_match.group())
                    result["base_filename"] = re.sub(r"#+", "XXXX", filename)
                    result["padding"] = hash_count
                    result["extension"] = "." + ext if ext else ""
                    result["is_sequence"] = True
                    # Try to detect frame from filename
                    hash_pos = filename.find(hash_match.group())
                    frame_match = re.search(r"\d+", base[hash_pos:])
                    if frame_match:
                        result["frame_start"] = int(frame_match.group())
                    # Build pattern
                    result["sequence_pattern"] = os.path.join(
                        directory, re.sub(r"#+", "%0" + str(hash_count) + "d", filename)
                    )
                    if ext:
                        result["sequence_pattern"] = (
                            result["sequence_pattern"][: -len(ext) - 1] + "." + ext
                        )

            elif pattern_type == "printf":
                base, ext = match.groups()
                result["base_filename"] = re.sub(r"%0\d+d", "XXXX", filename)
                result["extension"] = ext
                result["is_sequence"] = True
                # Detect padding from format spec
                padding_match = re.search(r"%0(\d)d", filename)
                if padding_match:
                    result["padding"] = int(padding_match.group(1))
                result["sequence_pattern"] = input_path

            elif pattern_type == "houdini":
                # Houdini style: image.$F4.png
                base, frame_var, ext = match.groups()
                # Extract padding from $F4 -> 4
                padding = int(frame_var[2:])  # $F4 -> 4
                result["base_filename"] = base + ".$F" + str(padding) + ext
                result["padding"] = padding
                result["extension"] = ext
                result["is_sequence"] = True
                result["pattern_type"] = "houdini"
                # Convert to printf format for processing
                result["sequence_pattern"] = os.path.join(
                    directory, base + ".%0" + str(padding) + "d" + ext
                )

            elif pattern_type == "houdini_underscore":
                # Houdini style with underscore: render_$F3.png
                base, frame_var, ext = match.groups()
                padding = int(frame_var[2:])  # $F3 -> 3
                result["base_filename"] = base + "_$F" + str(padding) + ext
                result["padding"] = padding
                result["extension"] = ext
                result["is_sequence"] = True
                result["pattern_type"] = "houdini"
                # Convert to printf format for processing
                result["sequence_pattern"] = os.path.join(
                    directory, base + "_%0" + str(padding) + "d" + ext
                )

            result["matches"].append(
                {
                    "pattern_type": pattern_type,
                    "base": match.group(1) if match.lastindex else filename,
                }
            )
            break

    # If no pattern matched, check if it's a single file
    if not result["is_sequence"]:
        # Check if there are other numbered files in the directory
        if directory and os.path.isdir(directory):
            try:
                files = os.listdir(directory)
                numbered_files = []
                for f in files:
                    # Try to find frame numbers
                    frame_match = re.search(r"_(\d+)", f)
                    if frame_match:
                        frame_num = int(frame_match.group(1))
                        ext_match = re.search(r"\.(\w+)$", f)
                        if ext_match:
                            numbered_files.append((f, frame_num, ext_match.group(1)))

                if numbered_files:
                    # Found numbered files - this is likely a sequence
                    numbered_files.sort(key=lambda x: x[1])
                    min_frame = min(f[1] for f in numbered_files)
                    max_frame = max(f[1] for f in numbered_files)

                    # Try to determine padding from first file
                    first_file = numbered_files[0][0]
                    frame_match = re.search(r"_(\d+)", first_file)
                    if frame_match:
                        padding = len(frame_match.group(1))
                        ext = numbered_files[0][2]
                        base = first_file[: frame_match.start()] + "_XXXX"

                        result["base_filename"] = base + "." + ext
                        result["padding"] = padding
                        result["frame_start"] = min_frame
                        result["frame_end"] = max_frame
                        result["extension"] = "." + ext
                        result["is_sequence"] = True
                        result["sequence_pattern"] = os.path.join(
                            directory,
                            base.replace("_XXXX", "%0" + str(padding) + "d." + ext),
                        )
                        result["total_frames"] = len(numbered_files)
            except Exception:
                pass

    return result


def find_sequence_frames(
    directory: str,
    pattern: str = None,
    base: str = None,
    padding: int = None,
    extension: str = None,
) -> list:
    """Find all frames in a sequence.

    Args:
        directory: Directory containing sequence files
        pattern: Optional glob pattern (e.g., "image_*.png")
        base: Base filename without frame number
        padding: Frame number padding
        extension: File extension

    Returns:
        List of tuples (frame_number, full_path)
    """
    import re

    frames = []

    if pattern:
        # Use glob pattern
        import glob

        search_pattern = os.path.join(directory, pattern.replace("XXXX", "*"))
        files = glob.glob(search_pattern)

        for f in files:
            # Extract frame number
            frame_match = re.search(r"(\d+)\.\w+$", os.path.basename(f))
            if frame_match:
                frames.append((int(frame_match.group(1)), f))
    else:
        # List directory and find matching files
        try:
            all_files = os.listdir(directory)
            for f in all_files:
                if f.startswith("."):
                    continue

                # Match against base pattern
                if base:
                    # Check if filename starts with base and has frame number before extension
                    ext_match = re.search(r"\.(\w+)$", f)
                    if not ext_match:
                        continue
                    ext = ext_match.group(1)
                    if ext != extension.replace(".", ""):
                        continue

                    # Pattern: base + frame_number + extension
                    frame_match = re.search(r"_(\d+)\.", f)
                    if frame_match:
                        frame_num = int(frame_match.group(1))
                        frames.append((frame_num, os.path.join(directory, f)))
        except Exception:
            pass

    frames.sort(key=lambda x: x[0])
    return frames


def process_sequence_frames(
    frame_files: list,
    output_template: str,
    padding: int,
    directory: str,
    seed: int,
    max_num_tokens: int,
    frame_step: int = 1,
) -> dict:
    """Process a sequence of frames.

    Args:
        frame_files: List of (frame_number, file_path) tuples
        output_template: Output template for meshes
        padding: Frame number padding
        directory: Output directory
        seed: Random seed
        max_num_tokens: Memory limit
        frame_step: Frame step

    Returns:
        dict with processing results
    """
    if not frame_files:
        return {
            "status": "error",
            "message": "No frames to process",
        }

    results = []
    errors = []

    # Get padding from output template if specified
    output_padding = padding
    padding_match = re.search(r"XXXX|%0\d+d", output_template)
    if padding_match:
        if "XXXX" in padding_match.group():
            output_padding = len(padding_match.group())
        else:
            output_padding = int(re.search(r"%0(\d)d", output_template).group(1))

    for frame_num, frame_path in frame_files:
        # Generate output path
        output_path = output_template.replace(
            "XXXX", str(frame_num).zfill(output_padding)
        )
        output_path = re.sub(r"%0\d+d", str(frame_num), output_path)

        try:
            print(f"[Sequence] Processing frame {frame_num}: {frame_path}")

            # Run inference on this frame
            result = run_inference(
                image_path=frame_path,
                output_path=output_path,
                seed=seed + frame_num if seed else seed,
                max_num_tokens=max_num_tokens,
            )

            results.append(
                {
                    "frame": frame_num,
                    "input": frame_path,
                    "output": output_path,
                    "status": "success",
                }
            )

            # Clean memory between frames
            clean_memory()

        except Exception as e:
            error_msg = str(e)
            print(f"[Sequence] Error on frame {frame_num}: {error_msg}")
            errors.append(
                {
                    "frame": frame_num,
                    "input": frame_path,
                    "error": error_msg,
                }
            )

    # Build summary
    return {
        "status": "completed" if results else "failed",
        "total_frames": len(frame_files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
        "pattern_info": {
            "directory": directory,
            "padding": padding,
            "frame_start": frame_files[0][0] if frame_files else None,
            "frame_end": frame_files[-1][0] if frame_files else None,
        },
    }


def run_inference_sequence(
    input_path: str,
    output_template: str = None,
    frame_start: int = None,
    frame_end: int = None,
    frame_step: int = 1,
    seed: int = 42,
    max_num_tokens: int = 4096,
    auto_detect: bool = True,
    return_thumbnails: bool = False,
) -> dict:
    """Run Pixal3D inference on an image sequence.

    Supports various sequence naming conventions:
    - Maya style: render_0001.png, render.####.png
    - After Effects style: sequence_0001.png
    - Fusion style: image_0001.png, image.0001.png
    - Nuke/Foundry: image.%04d.png (printf-style)
    - Houdini/SideFX: image.$F4.png (variable padding $F1-$F9)
    - IFL (Image File List): .ifl text file with one filename per line

    Args:
        input_path: Path to an image in the sequence, sequence pattern, or IFL file
        output_template: Output template for meshes (e.g., "output_####.glb")
        frame_start: Starting frame (auto-detected if not specified)
        frame_end: Ending frame (auto-detected if not specified)
        frame_step: Frame step for processing every N frames
        seed: Random seed for reproducibility
        max_num_tokens: Memory limit
        auto_detect: Whether to auto-detect sequence pattern

    Returns:
        dict with processing results
    """
    import glob
    import re

    # Detect sequence pattern
    if auto_detect:
        pattern_info = detect_sequence_pattern(input_path)
    else:
        pattern_info = {
            "is_sequence": False,
            "directory": os.path.dirname(input_path),
            "extension": os.path.splitext(input_path)[1],
        }

    # Handle IFL files
    if pattern_info.get("is_ifl"):
        if pattern_info.get("status") == "error":
            return pattern_info

        # Use files directly from IFL
        ifl_files = pattern_info.get("files", [])
        if not ifl_files:
            return {"status": "error", "message": "IFL file contains no valid paths"}

        directory = pattern_info.get("directory", ".")

        # Build frame list from IFL - use index as frame number
        frame_files = []
        for idx, file_path in enumerate(ifl_files):
            if not os.path.exists(file_path):
                print(f"[IFL] File not found: {file_path}, skipping")
                continue
            frame_files.append((idx + 1, file_path))  # 1-indexed frames

        if not frame_files:
            return {"status": "error", "message": "No valid files found in IFL"}

        # Determine output template
        if not output_template:
            output_template = os.path.join(directory, "output_XXXX.glb")

        return process_sequence_frames(
            frame_files=frame_files,
            output_template=output_template,
            padding=4,
            directory=directory,
            seed=seed,
            max_num_tokens=max_num_tokens,
            frame_step=frame_step,
        )

    if not pattern_info.get("is_sequence", False) and auto_detect:
        # Check if it's a single file
        if os.path.isfile(input_path):
            return {
                "status": "error",
                "message": f"Input does not appear to be a sequence: {input_path}. Use auto_detect=False for single files or specify a sequence pattern.",
            }
        else:
            return {"status": "error", "message": f"File not found: {input_path}"}

    directory = pattern_info.get("directory", ".")
    padding = pattern_info.get("padding", 4)
    extension = pattern_info.get("extension", ".png")
    base_pattern = pattern_info.get("sequence_pattern", "")

    # Find all frames in the sequence
    if "*" in input_path or "%" in input_path or "$" in input_path:
        # User provided a pattern - handle various formats
        search_pattern = input_path
        # Handle printf style: %04d
        search_pattern = re.sub(r"%0\d+d", "*", search_pattern)
        # Handle Houdini style: $F4
        search_pattern = re.sub(r"\$F\d+", "*", search_pattern)
        glob_pattern = search_pattern
    else:
        # Build glob pattern from detected info
        glob_pattern = os.path.join(directory, "*" + extension)

    # Find matching files
    all_files = glob.glob(glob_pattern)
    frame_files = []

    for f in all_files:
        # Extract frame number from various patterns
        frame_num = None
        basename = os.path.basename(f)

        # Try underscore pattern: image_0001.png
        frame_match = re.search(r"_(\d+)\.", basename)
        if frame_match:
            frame_num = int(frame_match.group(1))

        # Try dot pattern: image.0001.png
        if frame_num is None:
            frame_match = re.search(r"\.(\d+)\.", basename)
            if frame_match:
                frame_num = int(frame_match.group(1))

        # Try printf pattern: already converted to *
        if frame_num is None:
            frame_match = re.search(r"(\d+)\.", basename)
            if frame_match:
                frame_num = int(frame_match.group(1))

        if frame_num is not None:
            if frame_start and frame_num < frame_start:
                continue
            if frame_end and frame_num > frame_end:
                continue
            frame_files.append((frame_num, f))

    frame_files.sort(key=lambda x: x[0])

    # Apply frame step
    if frame_step > 1:
        frame_files = frame_files[::frame_step]

    if not frame_files:
        return {
            "status": "error",
            "message": f"No frames found in sequence: {input_path}",
        }

    # Determine output template
    if not output_template:
        # Create default output template
        first_frame = frame_files[0][1]
        frame_match = re.search(r"^(.+?)[_-]?(\d+)", os.path.basename(first_frame))
        if frame_match:
            base_out = frame_match.group(1)
        else:
            base_out = "output"
        output_template = os.path.join(directory, base_out + "_XXXX.glb")

    # Process each frame
    results = []
    errors = []

    # Get padding from output template if specified
    output_padding = padding
    padding_match = re.search(r"XXXX|%0\d+d", output_template)
    if padding_match:
        if "XXXX" in padding_match.group():
            output_padding = len(padding_match.group())
        else:
            output_padding = int(re.search(r"%0(\d)d", output_template).group(1))

    for frame_num, frame_path in frame_files:
        # Generate output path
        output_path = output_template.replace(
            "XXXX", str(frame_num).zfill(output_padding)
        )
        output_path = re.sub(r"%0\d+d", str(frame_num), output_path)

        try:
            print(f"[Sequence] Processing frame {frame_num}: {frame_path}")

            # Run inference on this frame
            result = run_inference(
                image_path=frame_path,
                output_path=output_path,
                seed=seed + frame_num if seed else seed,
                max_num_tokens=max_num_tokens,
            )

            results.append(
                {
                    "frame": frame_num,
                    "input": frame_path,
                    "output": output_path,
                    "status": "success",
                }
            )

            # Clean memory between frames
            clean_memory()

        except Exception as e:
            error_msg = str(e)
            print(f"[Sequence] Error on frame {frame_num}: {error_msg}")
            errors.append(
                {
                    "frame": frame_num,
                    "input": frame_path,
                    "error": error_msg,
                }
            )

    # Build summary
    summary = {
        "status": "completed" if results else "failed",
        "total_frames": len(frame_files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
        "pattern_info": {
            "directory": directory,
            "padding": padding,
            "extension": extension,
            "frame_start": frame_files[0][0] if frame_files else None,
            "frame_end": frame_files[-1][0] if frame_files else None,
        },
    }

    return summary


# Tool handlers
def handle_load_pipeline(args):
    """Load the Pixal3D pipeline."""
    try:
        load_pipeline()
        return {"status": "success", "message": "Pipeline loaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_unload_pipeline(args):
    """Unload the Pixal3D pipeline and free memory."""
    try:
        unload_pipeline()
        return {"status": "success", "message": "Pipeline unloaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_run_inference(args):
    """Run Pixal3D inference on an image.

    Args matching CLI conventions:
        --image: Path to input image (required)
        --output: Output GLB path (default: ./output.glb)
        --seed: Random seed (default: 42)
        --max_num_tokens: Memory limit (default: 4096)
        --return_thumbnail: Return base64 thumbnail preview (default: false)
    """
    try:
        image = args.get("image")
        output = args.get("output", "./output.glb")
        seed = args.get("seed", 42)
        max_num_tokens = args.get("max_num_tokens", 4096)
        return_thumbnail = args.get("return_thumbnail", False)

        if not image:
            return {"status": "error", "message": "image is required (use --image)"}

        result = run_inference(
            image_path=image,
            output_path=output,
            seed=seed,
            max_num_tokens=max_num_tokens,
            return_thumbnail=return_thumbnail,
        )
        return {"status": "success", **result}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                "status": "error",
                "message": f"Out of memory. Try reducing max_num_tokens. Error: {str(e)}",
            }
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_get_memory_status(args):
    """Get current GPU memory status."""
    try:
        if torch.cuda.is_available():
            return {
                "status": "success",
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "pipeline_loaded": PIPELINE_LOADED,
            }
        else:
            return {
                "status": "success",
                "gpu_available": False,
                "pipeline_loaded": PIPELINE_LOADED,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_cleanup_memory(args):
    """Clean up GPU memory."""
    try:
        clean_memory()
        return {"status": "success", "message": "Memory cleaned successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def render_glb_preview(glb_path: str, max_size: int = 512) -> dict:
    """Render a GLB file to a base64-encoded preview image.

    Uses the 3d-to-image package for auto-fit rendering.

    Args:
        glb_path: Path to the GLB file
        max_size: Maximum dimension for the preview image

    Returns:
        dict with preview image data or error
    """
    import base64
    import io

    try:
        # Read GLB file as bytes
        with open(glb_path, "rb") as f:
            glb_bytes = f.read()

        # Import the rendering function
        from model_to_image import glb_to_image

        # Render to image
        result = glb_to_image(glb_bytes)

        if isinstance(result, io.BytesIO):
            # Encode to base64
            png_data = base64.b64encode(result.getvalue()).decode("utf-8")

            # Get image dimensions if possible
            result.seek(0)
            from PIL import Image

            img = Image.open(result)
            width, height = img.size

            return {
                "preview": png_data,
                "width": width,
                "height": height,
                "max_size": max_size,
                "format": "PNG",
                "size_bytes": len(png_data),
            }
        else:
            # Rendering failed, result contains error message
            return {
                "error": str(result),
                "glb_path": glb_path,
            }

    except FileNotFoundError:
        return {
            "error": f"GLB file not found: {glb_path}",
            "glb_path": glb_path,
        }
    except ImportError as e:
        return {
            "error": f"3d-to-image package not installed. Run: pip install 3d-to-image. Details: {e}",
            "glb_path": glb_path,
        }
    except Exception as e:
        return {
            "error": str(e),
            "glb_path": glb_path,
        }


def is_url(path: str) -> bool:
    """Check if a path is a URL (http, https, ftp, sftp)."""
    return path.lower().startswith(("http://", "https://", "ftp://", "sftp://"))


def download_url(url: str, cache_dir: str = None) -> str:
    """Download a URL to a local temporary file.
    Args:
        url: URL to download
        cache_dir: Optional directory to cache downloaded files
    Returns:
        Path to local cached file
    """
    import tempfile
    import urllib.parse

    try:
        # Parse URL to get filename
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path) or "downloaded_file"

        # Use cache_dir if provided, otherwise use temp
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            local_path = os.path.join(cache_dir, filename)
        else:
            local_path = os.path.join(tempfile.gettempdir(), filename)

        # Download the file
        if url.lower().startswith(("http://", "https://")):
            import urllib.request

            urllib.request.urlretrieve(url, local_path)
        elif url.lower().startswith(("ftp://", "sftp://")):
            # Use subprocess for FTP/SFTP
            import subprocess

            subprocess.run(["curl", "-o", local_path, url], check=True)

        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def is_archive(path: str) -> bool:
    """Check if a path points to a file inside an archive.
    Examples:
        /path/to/archive.zip/image.png
        /path/to/archive.tar.gz/frame_001.jpg
    """
    archive_extensions = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
    path_lower = path.lower()
    return any(
        path_lower.endswith(ext) or ext in path_lower for ext in archive_extensions
    )


def extract_from_archive(
    archive_path: str, member_name: str = None, cache_dir: str = None
) -> list:
    """Extract files from an archive.
    Args:
        archive_path: Path to the archive file
        member_name: Optional specific member to extract (supports wildcards)
        cache_dir: Directory to extract to (uses temp if None)
    Returns:
        List of extracted file paths
    """
    import fnmatch
    import glob as glob_module
    import tarfile
    import tempfile
    import zipfile

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        extract_to = cache_dir
    else:
        extract_to = tempfile.mkdtemp(prefix="pixal3d_")

    extracted = []
    archive_lower = archive_path.lower()

    try:
        if archive_lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                if member_name:
                    # Extract specific member(s)
                    if "*" in member_name:
                        members = [
                            m for m in zf.namelist() if fnmatch.fnmatch(m, member_name)
                        ]
                    else:
                        members = [member_name] if member_name in zf.namelist() else []
                else:
                    members = zf.namelist()

                for member in members:
                    if member.endswith("/"):  # Skip directories
                        continue
                    try:
                        zf.extract(member, extract_to)
                        extracted.append(os.path.join(extract_to, member))
                    except Exception:
                        pass

        elif archive_lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
            mode = (
                "r:gz"
                if archive_lower.endswith((".gz", ".tgz"))
                else "r:bz2"
                if archive_lower.endswith(".bz2")
                else "r"
            )
            with tarfile.open(archive_path, mode=mode) as tf:
                if member_name:
                    members = [
                        m
                        for m in tf.getnames()
                        if fnmatch.fnmatch(os.path.basename(m), member_name)
                    ]
                else:
                    members = tf.getnames()

                for member in members:
                    if member.endswith("/"):
                        continue
                    try:
                        tf.extract(member, extract_to)
                        extracted.append(os.path.join(extract_to, member))
                    except Exception:
                        pass
    except Exception as e:
        raise RuntimeError(f"Failed to extract from archive {archive_path}: {e}")

    return extracted


def resolve_path(path: str, cache_dir: str = None) -> list:
    """Resolve a path that could be:
    - Local file
    - URL (http, https, ftp, sftp)
    - Archive with member path (archive.zip/member.png)

    Args:
        path: The path to resolve
        cache_dir: Directory for cached downloads/extractions

    Returns:
        List of resolved local file paths
    """
    # Check for URL
    if is_url(path):
        local_file = download_url(path, cache_dir)
        return [local_file]

    # Check for archive
    if is_archive(path):
        # Parse archive path and member
        # Format: /path/to/archive.zip/member/name.png
        parts = path.split("/")
        for i, part in enumerate(parts):
            if any(
                part.lower().endswith(ext)
                for ext in (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
            ):
                archive_path = "/".join(parts[: i + 1])
                member_name = "/".join(parts[i + 1 :]) if i + 1 < len(parts) else None
                if archive_path != path and member_name:  # There's a member specified
                    return extract_from_archive(archive_path, member_name, cache_dir)
                else:
                    # No member specified, extract all
                    return extract_from_archive(archive_path, None, cache_dir)
        return []

    # Regular local file
    return [path]


def preview_images(
    image_paths: list = None,
    input_path: str = None,
    max_size: int = 256,
) -> dict:
    """Generate thumbnails for images without running inference.

    This is a fast, non-blocking operation that returns instant previews.
    Can accept either a list of image paths or a sequence pattern.

    Args:
        image_paths: List of image paths (alternative to input_path)
        input_path: Single image path or sequence pattern (alternative to image_paths)
        max_size: Maximum thumbnail dimension (default: 256)

    Returns:
        dict with thumbnails for all images
    """
    import glob
    import re

    files_to_preview = []
    errors = []

    # Collect all files to preview
    if image_paths:
        for p in image_paths:
            # Resolve each path (handles URLs, archives)
            try:
                resolved = resolve_path(p)
                files_to_preview.extend(resolved)
            except Exception as e:
                errors.append({"path": p, "error": str(e)})

    if input_path:
        # Check if it's a URL or archive first
        if is_url(input_path) or is_archive(input_path):
            try:
                resolved = resolve_path(input_path)
                files_to_preview.extend(resolved)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to resolve input_path: {e}",
                }
        elif input_path.endswith(".ifl"):
            # IFL file - read image list
            try:
                with open(input_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            directory = os.path.dirname(input_path)
                            if os.path.isabs(line):
                                files_to_preview.append(line)
                            else:
                                files_to_preview.append(os.path.join(directory, line))
            except Exception as e:
                return {"status": "error", "message": f"Failed to read IFL: {e}"}
        elif "*" in input_path or "%" in input_path or "$F" in input_path:
            # Glob pattern - expand it
            pattern = input_path
            pattern = re.sub(r"%0\d+d", "*", pattern)
            pattern = re.sub(r"\$F\d+", "*", pattern)
            files = glob.glob(pattern)
            files_to_preview.extend(files)
        elif os.path.isfile(input_path):
            files_to_preview.append(input_path)
        else:
            # Try to find matching files in directory
            directory = os.path.dirname(input_path)
            ext = os.path.splitext(input_path)[1]
            if not ext:
                ext = ".*"
            search = os.path.join(directory if directory else ".", "*" + ext)
            files = glob.glob(search)
            files_to_preview.extend(files)

    if not files_to_preview:
        return {"status": "error", "message": "No images found to preview"}

    # Generate thumbnails for all files
    thumbnails = []
    errors = []

    for file_path in files_to_preview:
        if not os.path.exists(file_path):
            errors.append({"path": file_path, "error": "File not found"})
            continue

        thumb = generate_thumbnail(file_path, max_size=max_size)
        if "error" in thumb:
            errors.append({"path": file_path, "error": thumb["error"]})
        else:
            thumbnails.append(
                {
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "frame": len(thumbnails),  # 0-indexed
                    "thumbnail": thumb["thumbnail"],
                    "original_width": thumb["width"],
                    "original_height": thumb["height"],
                    "thumbnail_width": thumb["thumbnail_width"],
                    "thumbnail_height": thumb["thumbnail_height"],
                }
            )

    return {
        "status": "success",
        "count": len(thumbnails),
        "thumbnails": thumbnails,
        "errors": errors if errors else None,
        "max_size": max_size,
    }


def handle_preview_images(args):
    """Generate thumbnail previews for images without running inference.

    This fast, non-blocking tool returns instant visual feedback.
    Use before run_inference or run_inference_sequence to verify inputs.

    Args:
        image_paths: Array of image paths (optional)
        input_path: Single image, sequence pattern, or IFL file (optional)
        max_size: Maximum thumbnail dimension (default: 256)
    """
    try:
        image_paths = args.get("image_paths")
        input_path = args.get("input_path")
        max_size = args.get("max_size", 256)

        if not image_paths and not input_path:
            return {
                "status": "error",
                "message": "Either image_paths or input_path is required",
            }

        result = preview_images(
            image_paths=image_paths,
            input_path=input_path,
            max_size=max_size,
        )
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


def preview_model(
    glb_paths: list = None,
    input_path: str = None,
    max_size: int = 512,
) -> dict:
    """Generate preview images for GLB model files.

    Uses 3d-to-image package for auto-fit rendering.

    Args:
        glb_paths: List of GLB file paths (alternative to input_path)
        input_path: Single GLB path, sequence pattern, or IFL file (alternative)
        max_size: Maximum dimension for preview images (default: 512)

    Returns:
        dict with preview images for all models
    """
    import glob
    import re

    files_to_preview = []

    # Collect all GLB files to preview
    if glb_paths:
        files_to_preview.extend(glb_paths)

    if input_path:
        # Check if it's a sequence pattern or IFL
        if input_path.endswith(".ifl"):
            # IFL file - read file list
            try:
                with open(input_path, "r") as f:
                    directory = os.path.dirname(input_path)
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if os.path.isabs(line):
                                files_to_preview.append(line)
                            else:
                                files_to_preview.append(os.path.join(directory, line))
            except Exception as e:
                return {"status": "error", "message": f"Failed to read IFL: {e}"}
        elif "*" in input_path or "%" in input_path or "$" in input_path:
            # Glob pattern - expand it
            pattern = input_path
            pattern = re.sub(r"%0\d+d", "*", pattern)
            pattern = re.sub(r"\$F\d+", "*", pattern)
            files = glob.glob(pattern)
            files_to_preview.extend(files)
        elif os.path.isfile(input_path):
            files_to_preview.append(input_path)
        else:
            # Try to find matching files in directory
            directory = os.path.dirname(input_path)
            search = os.path.join(directory if directory else ".", "*.glb")
            files = glob.glob(search)
            files_to_preview.extend(files)

    if not files_to_preview:
        return {
            "status": "error",
            "message": "No GLB files found to preview",
        }

    # Filter to only GLB files
    glb_files = [f for f in files_to_preview if f.lower().endswith(".glb")]

    if not glb_files:
        return {
            "status": "error",
            "message": "No GLB files found matching the criteria",
        }

    # Generate previews for all GLB files
    previews = []
    errors = []

    for glb_path in glb_files:
        print(f"[Preview] Rendering: {glb_path}")
        preview_data = render_glb_preview(glb_path, max_size=max_size)

        if "error" in preview_data:
            errors.append(
                {
                    "path": glb_path,
                    "error": preview_data["error"],
                }
            )
        else:
            previews.append(
                {
                    "path": glb_path,
                    "filename": os.path.basename(glb_path),
                    "preview": preview_data["preview"],
                    "width": preview_data["width"],
                    "height": preview_data["height"],
                }
            )

    return {
        "status": "success",
        "count": len(previews),
        "previews": previews,
        "errors": errors if errors else None,
        "max_size": max_size,
    }


def handle_preview_model(args):
    """Generate preview images for GLB models without running inference.

    This renders GLB files to PNG previews using 3d-to-image.
    Supports single files, sequences, and IFL files.

    Args:
        glb_paths: Array of GLB file paths (optional)
        input_path: Single GLB, sequence pattern, or IFL file (optional)
        max_size: Maximum preview dimension (default: 512)
    """
    try:
        glb_paths = args.get("glb_paths")
        input_path = args.get("input_path")
        max_size = args.get("max_size", 512)

        if not glb_paths and not input_path:
            return {
                "status": "error",
                "message": "Either glb_paths or input_path is required",
            }

        result = preview_model(
            glb_paths=glb_paths,
            input_path=input_path,
            max_size=max_size,
        )
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_preview_model_sequence(args):
    """Generate preview images for a GLB model sequence.

    Alias for preview_model with sequence-focused parameters.

    Args:
        input_path: Sequence pattern or IFL file for GLB files
        output_template: Optional output template for naming
        frame_start: Starting frame (optional)
        frame_end: Ending frame (optional)
        max_size: Maximum preview dimension (default: 512)
    """
    try:
        input_path = args.get("input_path")
        frame_start = args.get("frame_start")
        frame_end = args.get("frame_end")
        max_size = args.get("max_size", 512)

        if not input_path:
            return {
                "status": "error",
                "message": "input_path is required",
            }

        # Use preview_model with the sequence
        result = preview_model(
            input_path=input_path,
            max_size=max_size,
        )

        # Filter by frame range if specified
        if frame_start is not None or frame_end is not None:
            filtered_previews = []
            for p in result.get("previews", []):
                # Extract frame number from filename if possible
                import re

                frame_match = re.search(r"_(\d+)", os.path.basename(p["filename"]))
                if frame_match:
                    frame_num = int(frame_match.group(1))
                    if frame_start is not None and frame_num < frame_start:
                        continue
                    if frame_end is not None and frame_num > frame_end:
                        continue
                filtered_previews.append(p)
            result["previews"] = filtered_previews
            result["count"] = len(filtered_previews)

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_run_inference_sequence(args):
    """Run Pixal3D inference on an image sequence.

    Args:
        input_path: Path to an image in the sequence (auto-detects pattern)
        output_template: Output mesh template (e.g., "mesh_####.glb")
        frame_start: Starting frame number (auto-detected)
        frame_end: Ending frame number (auto-detected)
        frame_step: Process every N frames
        seed: Random seed
        max_num_tokens: Memory limit
        auto_detect: Auto-detect sequence pattern
        return_thumbnails: Return base64 thumbnail for each frame (default: false)
    """
    try:
        input_path = args.get("input_path")
        output_template = args.get("output_template")
        frame_start = args.get("frame_start")
        frame_end = args.get("frame_end")
        frame_step = args.get("frame_step", 1)
        seed = args.get("seed", 42)
        max_num_tokens = args.get("max_num_tokens", 4096)
        auto_detect = args.get("auto_detect", True)
        return_thumbnails = args.get("return_thumbnails", False)

        if not input_path:
            return {"status": "error", "message": "input_path is required"}

        result = run_inference_sequence(
            input_path=input_path,
            output_template=output_template,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_step=frame_step,
            seed=seed,
            max_num_tokens=max_num_tokens,
            auto_detect=auto_detect,
            return_thumbnails=return_thumbnails,
        )
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_load_conda_env(args):
    """Load conda environment variables.

    Args:
        env_name: Name of the conda environment (default: Pixal3D)
    """
    try:
        env_name = args.get("env_name", "Pixal3D")
        result = load_conda_env(env_name)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_get_conda_env_status(args):
    """Get conda environment status."""
    try:
        return get_conda_env_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_list_conda_envs(args):
    """List all available conda environments."""
    try:
        return list_conda_envs()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Tool definitions
TOOL_DEFINITIONS = [
    {
        "name": "load_pipeline",
        "description": "Load the Pixal3D pipeline and all required models into GPU memory. This must be called before running inference.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "unload_pipeline",
        "description": "Unload the Pixal3D pipeline and free GPU memory.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_inference",
        "description": "Run Pixal3D image-to-3D inference. CLI-style args: --image --output --seed --max_num_tokens",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Path to input image (--image)",
                },
                "output": {
                    "type": "string",
                    "description": "Output GLB path (--output, default: ./output.glb)",
                    "default": "./output.glb",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (--seed, default: 42)",
                    "default": 42,
                },
                "max_num_tokens": {
                    "type": "integer",
                    "description": "Max tokens for memory management (--max_num_tokens, lower = less memory)",
                    "default": 4096,
                },
            },
            "required": ["image"],
        },
    },
    {
        "name": "get_memory_status",
        "description": "Get current GPU memory usage and pipeline status.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "cleanup_memory",
        "description": "Clean up GPU memory by running garbage collection and clearing CUDA cache.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "load_conda_env",
        "description": "Load environment variables from a conda environment. Use this when the MCP server is launched from a desktop app (like Zed) without the conda environment pre-activated.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "env_name": {
                    "type": "string",
                    "description": "Name of the conda environment to load (default: Pixal3D)",
                    "default": "Pixal3D",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_conda_env_status",
        "description": "Get the current conda environment status and configuration.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_conda_envs",
        "description": "List all available conda environments. Use this to discover what conda environments you can load with load_conda_env.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_inference_sequence",
        "description": "Run Pixal3D inference on an image sequence. Supports Maya, After Effects, and Fusion filename conventions. Auto-detects frame padding, sequence pattern, and frame range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to an image in the sequence or sequence pattern (e.g., 'frames/image_0001.png')",
                },
                "output_template": {
                    "type": "string",
                    "description": "Output mesh template (e.g., 'output_####.glb', 'mesh.%04d.glb')",
                },
                "frame_start": {
                    "type": "integer",
                    "description": "Starting frame number (auto-detected if not specified)",
                },
                "frame_end": {
                    "type": "integer",
                    "description": "Ending frame number (auto-detected if not specified)",
                },
                "frame_step": {
                    "type": "integer",
                    "description": "Process every N frames (default: 1)",
                    "default": 1,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (default: 42)",
                    "default": 42,
                },
                "max_num_tokens": {
                    "type": "integer",
                    "description": "Memory limit (default: 4096)",
                    "default": 4096,
                },
                "auto_detect": {
                    "type": "boolean",
                    "description": "Auto-detect sequence pattern from filename (default: true)",
                    "default": True,
                },
            },
            "required": ["input_path"],
        },
    },
    {
        "name": "preview_images",
        "description": "Generate thumbnail previews for images without running inference. Fast, non-blocking operation that returns instant visual feedback. Supports sequences, IFL files, and glob patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of image paths to preview (alternative to input_path)",
                },
                "input_path": {
                    "type": "string",
                    "description": "Single image, sequence pattern, or IFL file (alternative to image_paths)",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum thumbnail dimension in pixels (default: 256)",
                    "default": 256,
                },
            },
        },
    },
    {
        "name": "preview_model",
        "description": "Render GLB models to preview PNG images using 3d-to-image. Fast, non-blocking. Supports single files, sequences, and IFL files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "glb_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of GLB file paths to preview (alternative to input_path)",
                },
                "input_path": {
                    "type": "string",
                    "description": "Single GLB, sequence pattern, or IFL file (alternative to glb_paths)",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum preview dimension in pixels (default: 512)",
                    "default": 512,
                },
            },
        },
    },
    {
        "name": "preview_model_sequence",
        "description": "Render a GLB model sequence to preview PNG images with optional frame range filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Sequence pattern or IFL file for GLB files",
                },
                "frame_start": {
                    "type": "integer",
                    "description": "Starting frame number (optional)",
                },
                "frame_end": {
                    "type": "integer",
                    "description": "Ending frame number (optional)",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum preview dimension in pixels (default: 512)",
                    "default": 512,
                },
            },
        },
    },
]

# Tool handlers mapping
TOOL_HANDLERS = {
    "load_pipeline": handle_load_pipeline,
    "unload_pipeline": handle_unload_pipeline,
    "run_inference": handle_run_inference,
    "run_inference_sequence": handle_run_inference_sequence,
    "preview_images": handle_preview_images,
    "preview_model": handle_preview_model,
    "preview_model_sequence": handle_preview_model_sequence,
    "get_memory_status": handle_get_memory_status,
    "cleanup_memory": handle_cleanup_memory,
    "load_conda_env": handle_load_conda_env,
    "get_conda_env_status": handle_get_conda_env_status,
    "list_conda_envs": handle_list_conda_envs,
}


def handle_request(request):
    """Handle an MCP request and return a response."""
    request_id = request.get("id", "unknown")
    method = request.get("method")
    params = request.get("params", {})

    jsonrpc_response = {"jsonrpc": "2.0"}

    if method == "initialize":
        resp = {
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "Pixal3D Image-to-3D MCP Server",
                    "version": "1.0.0",
                },
                "capabilities": {"tools": {}},
                "instructions": "Pixal3D MCP Server for image-to-3D generation. Use load_pipeline first, then run_inference.",
            },
        }
        resp.update(jsonrpc_response)
        return resp

    elif method == "tools/list":
        resp = {"id": request_id, "result": {"tools": TOOL_DEFINITIONS}}
        resp.update(jsonrpc_response)
        return resp

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if tool_name in TOOL_HANDLERS:
            result = TOOL_HANDLERS[tool_name](tool_args)
            resp = {
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }
            resp.update(jsonrpc_response)
            return resp
        else:
            resp = {
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
            resp.update(jsonrpc_response)
            return resp

    elif method == "shutdown":
        unload_pipeline()
        resp = {"id": request_id, "result": {"success": True}}
        resp.update(jsonrpc_response)
        return resp

    else:
        resp = {
            "id": request_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }
        resp.update(jsonrpc_response)
        return resp


def main():
    """Main entry point for the MCP server."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = handle_request(request)

            if response:
                print(json.dumps(response), flush=True)

        except KeyboardInterrupt:
            unload_pipeline()
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
