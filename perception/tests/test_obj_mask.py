"""Simple test script for obj_mask.py with text + bbox combined input.

This script tests the obj_mask.py script with a sample image from COCO dataset
using both text description and bounding box for segmentation.

Usage:
    python test_obj_mask.py
"""

import subprocess
import tempfile
import os
from PIL import Image
import requests


def test_obj_mask_combined():
    """Test obj_mask.py with text + bbox combined input."""
    print("=" * 60)
    print("Testing obj_mask.py with text + bbox combined input")
    print("=" * 60)
    
    # Download sample image
    image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    print(f"\nDownloading sample image from COCO dataset...")
    print(f"URL: {image_url}")
    
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input image
        input_path = os.path.join(tmpdir, "input.jpg")
        image.save(input_path)
        print(f"Saved input to: {input_path}")
        
        # Define output paths
        mask_path = os.path.join(tmpdir, "output_mask.png")
        overlay_path = os.path.join(tmpdir, "output_overlay.png")
        
        # Define test parameters
        # Approximate bounding box for a zebra in the image (in xywh format)
        bbox_xywh = "100,150,400,300"
        object_name = "cat"
        
        print(f"\nTest parameters:")
        print(f"  Object name: {object_name}")
        print(f"  Bbox (xywh): {bbox_xywh}")
        
        # Build command
        cmd = [
            "python",
            "./perception/scripts/obj_mask.py",
            "--frame_path", input_path,
            "--object_name", object_name,
            "--bbox_xywh", bbox_xywh,
            "--output_mask_path", mask_path,
            "--output_overlay_path", overlay_path,
            "--mask_id", "1"
        ]
        
        print(f"\nRunning command:")
        print(" ".join(cmd))
        print("\n" + "-" * 60)
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("-" * 60)
        
        # Check results
        if result.returncode != 0:
            print(f"\n❌ Test FAILED with exit code {result.returncode}")
            return False
        
        # Verify output files exist
        if not os.path.exists(mask_path):
            print(f"\n❌ Test FAILED: Mask file not created at {mask_path}")
            return False
        
        if not os.path.exists(overlay_path):
            print(f"\n❌ Test FAILED: Overlay file not created at {overlay_path}")
            return False
        
        # Load and check mask
        mask_img = Image.open(mask_path)
        print(f"\n✓ Mask created successfully")
        print(f"  Path: {mask_path}")
        print(f"  Size: {mask_img.size}")
        print(f"  Mode: {mask_img.mode}")
        
        # Load and check overlay
        overlay_img = Image.open(overlay_path)
        print(f"\n✓ Overlay created successfully")
        print(f"  Path: {overlay_path}")
        print(f"  Size: {overlay_img.size}")
        print(f"  Mode: {overlay_img.mode}")
        
        # Copy results to current directory for inspection
        mask_img.save("test_output_mask.png")
        overlay_img.save("test_output_overlay.png")
        print(f"\n✓ Test outputs saved to current directory:")
        print(f"  - test_output_mask.png")
        print(f"  - test_output_overlay.png")
        
        print("\n" + "=" * 60)
        print("✓ TEST PASSED")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = test_obj_mask_combined()
    exit(0 if success else 1)
