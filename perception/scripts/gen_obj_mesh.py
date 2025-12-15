#!/usr/bin/env python
"""
Standalone script to generate PLY Gaussian splat from a single RGB image with mask using SAM3D.

This is a minimal, self-contained script that takes an image and a mask, runs SAM3D inference,
and outputs a PLY file (Gaussian splat point cloud).

Usage:
    python generate_ply_from_image.py --image path/to/image.png --mask path/to/mask.png --output model.ply
    
    # Or use mask from directory with index
    python generate_ply_from_image.py --image path/to/image.png --mask_dir path/to/masks --mask_index 0 --output model.ply
    
Examples:
    # Basic usage with mask file
    python perception/examples/sam3d_obj_ply_from_image.py \
        --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
        --mask ./test_case/images/shutterstock_stylish_kidsroom_1640806567/mask_14.png \
        --output toy_bear.ply
    
    # Using mask directory and index
    python perception/examples/sam3d_obj_ply_from_image.py \
        --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \
        --mask_dir ./test_case/images/shutterstock_stylish_kidsroom_1640806567 \
        --mask_index 14 \
        --output toy_bear.ply
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment variables before importing sam3d_objects
os.environ["LIDRA_SKIP_INIT"] = "true"
if "CONDA_PREFIX" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]

# Add notebook to path for inference utilities
notebook_path = Path(__file__).parent.parent.parent / "external" / "sam-3d-objects" / "notebook"
sys.path.insert(0, str(notebook_path))

from inference import Inference, load_image, load_single_mask
from PIL import Image
import numpy as np


def load_mask_from_file(mask_path):
    """
    Load mask from an image file.
    Supports PNG, JPG, and other common formats.
    Returns boolean mask where True = foreground object.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask_img = Image.open(mask_path)
    
    # Convert to grayscale if needed
    if mask_img.mode == 'RGBA':
        # Use alpha channel as mask
        mask = np.array(mask_img)[:, :, 3] > 127
    elif mask_img.mode == 'L':
        mask = np.array(mask_img) > 127
    else:
        # Convert to grayscale and threshold
        mask_img = mask_img.convert('L')
        mask = np.array(mask_img) > 127
    
    return mask.astype(bool)


def main():
    parser = argparse.ArgumentParser(
        description='Generate PLY Gaussian splat from single RGB image with mask using SAM3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PLY from image and mask file:
  python gen_obj_mesh.py \\
      --image my_photo.jpg \\
      --mask my_mask.png \\
      --output model.ply

  # Generate PLY using mask from directory:
  python gen_obj_mesh.py \\
      --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \\
      --mask_dir notebook/images/shutterstock_stylish_kidsroom_1640806567 \\
      --mask_index 14 \\
      --output toy_bear.ply

Notes:
  - The mask should be a binary image (white=object, black=background)
  - Supports PNG, JPG, and other common image formats
  - Output is a PLY file containing a Gaussian splat point cloud
  - Use --seed for reproducible results
"""
    )
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input RGB image')
    
    # Mask input (one of these is required)
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument('--mask', type=str,
                           help='Path to mask image file (binary image)')
    mask_group.add_argument('--mask_dir', type=str,
                           help='Directory containing masks (use with --mask_index)')
    
    parser.add_argument('--mask_index', type=int, default=0,
                        help='Mask index when using --mask_dir (default: 0)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PLY file path')
    
    # Optional parameters
    parser.add_argument('--config', type=str, default='external/sam-3d-objects/checkpoints/hf/pipeline.yaml',
                        help='Path to SAM3D config file (default: external/sam-3d-objects/checkpoints/hf/pipeline.yaml)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--compile', action='store_true',
                        help='Enable model compilation for faster inference (takes longer first time)')
    
    args = parser.parse_args()
    
    # ========================================
    # Validation
    # ========================================
    
    print("\n" + "="*70)
    print(" SAM3D - Generate PLY from Image + Mask")
    print("="*70 + "\n")
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"   Make sure you have downloaded the SAM3D checkpoints.")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}\n")
    
    # ========================================
    # Load Model
    # ========================================
    
    print(f"📦 Loading SAM3D model...")
    print(f"   Config: {args.config}")
    if args.compile:
        print(f"   Compilation: enabled (first run will be slow)")
    
    try:
        inference = Inference(args.config, compile=args.compile)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # ========================================
    # Load Image
    # ========================================
    
    print(f"🖼️  Loading image...")
    print(f"   Path: {args.image}")
    
    try:
        image = load_image(args.image)
        print(f"✓ Image loaded: {image.shape} (H×W×C)\n")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # ========================================
    # Load Mask
    # ========================================
    
    print(f"🎭 Loading mask...")
    
    try:
        if args.mask:
            print(f"   Path: {args.mask}")
            mask = load_mask_from_file(args.mask)
        else:
            print(f"   Directory: {args.mask_dir}")
            print(f"   Index: {args.mask_index}")
            mask = load_single_mask(args.mask_dir, index=args.mask_index)
        
        print(f"✓ Mask loaded: {mask.shape} (H×W)")
        coverage = mask.sum() / mask.size * 100
        print(f"   Coverage: {coverage:.1f}% of image\n")
        
        if coverage < 1.0:
            print(f"⚠️  Warning: Mask coverage is very low ({coverage:.1f}%)")
            print(f"   Make sure the mask correctly identifies the object.\n")
        
    except Exception as e:
        print(f"❌ Error loading mask: {e}")
        sys.exit(1)
    
    # ========================================
    # Run Inference
    # ========================================
    
    print(f"🚀 Running SAM3D inference...")
    print(f"   Seed: {args.seed}")
    print(f"   This may take a minute...\n")
    
    try:
        output = inference(image, mask, seed=args.seed)
        print("✓ Inference complete\n")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        sys.exit(1)
    
    # ========================================
    # Export PLY
    # ========================================
    
    print(f"💾 Exporting PLY Gaussian splat...")
    print(f"   Output: {args.output}")
    
    # Check if Gaussian splat was generated
    if 'gs' not in output or output['gs'] is None:
        print("❌ Error: No Gaussian splat generated in output")
        print("   This may indicate an issue with the inference process.")
        sys.exit(1)
    
    try:
        # Save the Gaussian splat as PLY
        output['gs'].save_ply(args.output)
        
        # Get file info
        file_size = os.path.getsize(args.output)
        file_size_mb = file_size / (1024 * 1024)
        point_count = output['gs'].get_xyz.shape[0]
        
        print(f"✓ PLY file saved successfully")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Point count: {point_count:,} points\n")
        
    except Exception as e:
        print(f"❌ Error saving PLY: {e}")
        sys.exit(1)
    
    # ========================================
    # Summary
    # ========================================
    
    print("="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"\nGenerated PLY file: {args.output}")
    print(f"\nYou can view this file in:")
    print("  • CloudCompare")
    print("  • MeshLab")
    print("  • Blender")
    print("  • Online viewers (e.g., https://3dviewer.net/)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
