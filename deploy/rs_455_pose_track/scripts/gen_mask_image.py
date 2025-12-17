#!/usr/bin/env python3
"""
Generate mask image from RealSense camera using mesh API.

This script:
1. Captures a single RGB frame from RealSense camera
2. Gets object bbox (interactive or from text prompt)
3. Calls mesh API to generate mask
4. Saves mask image

Usage Examples:
    # Interactive bbox selection
    python gen_mask_image.py --output mask.png

    # With text prompt
    python gen_mask_image.py --prompt "toy bear" --output mask.png

    # Specify API server
    python gen_mask_image.py --api-url http://10.150.240.101:5001 --prompt "black mug" --output mask.png

    # Custom camera resolution
    python gen_mask_image.py --width 640 --height 480 --prompt "cup" --output mask.png

    # Select specific mask if multiple detected
    python gen_mask_image.py --prompt "object" --mask-id 1 --output mask.png
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import sys
import os
import base64
import tempfile
import io
import requests
from pathlib import Path
from PIL import Image
from typing import Optional, List


def check_api_health(api_url: str):
    """Check if the API server is available"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Connected to mesh generation API: {api_url}")
            print(f"  Device: {health_data.get('device', 'unknown')}")
            print(f"  Models loaded: {health_data.get('models_loaded', False)}")
            return True
    except Exception as e:
        print(f"✗ Cannot connect to API server: {api_url}")
        print(f"  Error: {e}")
        print(f"\n  Make sure the mesh generation API is running:")
        print(f"  cd perception/webapi && ./start_mesh_api.sh")
        sys.exit(1)


def capture_realsense_frame(width: int = 640, height: int = 480, fps: int = 30):
    """Capture single RGB frame from RealSense camera"""
    print(f"\n📷 Starting RealSense camera ({width}x{height} @ {fps}fps)...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    try:
        pipeline.start(config)
        print(f"✓ Camera started")
        
        # Warm up camera
        for _ in range(30):
            pipeline.wait_for_frames()
        
        # Capture frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            raise RuntimeError("Failed to capture frame")
        
        # Convert to numpy array
        rgb = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        print("✓ Frame captured")
        return rgb
        
    finally:
        pipeline.stop()
        print("✓ Camera stopped\n")


def interactive_bbox_selection(rgb: np.ndarray) -> Optional[List[int]]:
    """Interactive bounding box selection using mouse drag"""
    print("\n🖱️  Interactive bounding box selection:")
    print("  - Click and drag to draw bounding box around object")
    print("  - Press 'Enter' to accept bbox")
    print("  - Press 'r' to reset")
    print("  - Press 'q' to cancel\n")
    
    bbox = []
    drawing = False
    temp_bbox = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal bbox, drawing, temp_bbox
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            temp_bbox = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_bbox[2] = x
                temp_bbox[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            temp_bbox[2] = x
            temp_bbox[3] = y
            bbox = temp_bbox.copy()
    
    # Create window and set callback
    window_name = 'Select Object Bounding Box'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Convert RGB to BGR for OpenCV display
    display_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    
    while True:
        # Create display copy
        img_display = display_img.copy()
        
        # Draw current bbox
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, "Press 'Enter' to accept", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw temporary bbox while dragging
        if drawing and len(temp_bbox) == 4:
            x1, y1, x2, y2 = temp_bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        cv2.imshow(window_name, img_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('\r') or key == ord('\n'):  # Enter
            if len(bbox) == 4:
                cv2.destroyAllWindows()
                # Normalize bbox coordinates
                x1, y1, x2, y2 = bbox
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                return [x1, y1, x2, y2]
        
        elif key == ord('r'):  # Reset
            bbox = []
            temp_bbox = []
        
        elif key == ord('q'):  # Cancel
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None


def generate_mask_via_api(
    api_url: str,
    image: np.ndarray,
    text: Optional[str] = None,
    bbox_xyxy: Optional[List[int]] = None,
    mask_id: int = 0,
    seed: int = 42
) -> np.ndarray:
    """Generate mask by calling mesh generation API"""
    
    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img_path = tmp.name
        Image.fromarray(image).save(img_path)
    
    try:
        # Encode image
        print(f"📤 Encoding image...")
        with open(img_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Convert bbox format if provided
        bbox_xywh = None
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            bbox_xywh = [x1, y1, x2-x1, y2-y1]
        
        # Build request payload
        payload = {
            'image': image_b64,
            'output_format': 'ply',
            'mask_id': mask_id,
            'seed': seed,
            'return_mask': True,
            'return_file': False  # Don't need the mesh, just the mask
        }
        
        if text:
            payload['text'] = text
        
        if bbox_xywh:
            payload['bbox_xywh'] = bbox_xywh
        
        # Make API request
        print(f"\n🌐 Calling API: {api_url}/api/generate_mesh")
        print(f"   Text: {text}")
        print(f"   BBox: {bbox_xyxy}")
        print(f"   Mask ID: {mask_id}")
        print("\nGenerating mask...\n")
        
        response = requests.post(
            f"{api_url}/api/generate_mesh",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}\n{response.text}")
        
        result = response.json()
        
        if result['status'] != 'success':
            raise RuntimeError(f"API error: {result.get('error', 'Unknown error')}")
        
        # Decode mask
        if 'mask_base64' not in result:
            raise RuntimeError("No mask returned from API")
        
        mask_b64 = result['mask_base64']
        mask_bytes = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_img)
        
        print(f"✓ Mask generated successfully")
        print(f"  Mask score: {result['mask_score']:.3f}")
        print(f"  Shape: {mask_array.shape}")
        
        return mask_array
        
    finally:
        # Cleanup temp file
        if os.path.exists(img_path):
            os.unlink(img_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate mask image from RealSense camera using mesh API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Output
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output mask file path (PNG)')
    
    # Prompting options
    parser.add_argument('--prompt', type=str,
                        help='Text description of object (optional, enables interactive bbox selection)')
    
    # API options
    parser.add_argument('--api-url', type=str, default='http://localhost:5001',
                        help='Mesh generation API URL (default: http://localhost:5001)')
    
    # Camera options
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS (default: 30)')
    
    # Mask options
    parser.add_argument('--mask-id', type=int, default=0,
                        help='Which mask to use if multiple detected (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check API availability
    check_api_health(args.api_url)
    
    # Capture frame from RealSense
    rgb = capture_realsense_frame(args.width, args.height, args.fps)
    
    # Get bbox if no prompt provided
    bbox_xyxy = None
    if not args.prompt:
        print("⚠️  No prompt provided. Using interactive bbox selection.")
        bbox_xyxy = interactive_bbox_selection(rgb)
        if bbox_xyxy is None:
            print("❌ Bbox selection cancelled")
            return
        print(f"✓ Selected bbox: {bbox_xyxy}")
    
    # Generate mask via API
    mask = generate_mask_via_api(
        api_url=args.api_url,
        image=rgb,
        text=args.prompt,
        bbox_xyxy=bbox_xyxy,
        mask_id=args.mask_id,
        seed=args.seed
    )
    
    # Save mask
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(mask).save(output_path)
    print(f"\n✓ Mask saved to: {output_path}")
    
    # Show mask preview
    print("\n👁️  Displaying mask preview (press any key to close)...")
    cv2.imshow('Generated Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
