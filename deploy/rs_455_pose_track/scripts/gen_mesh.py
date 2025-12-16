#!/usr/bin/env python3
"""
Client script to generate 3D mesh from RealSense camera using webapi.

This script:
1. Captures RGB-D frames from RealSense camera
2. Gets object bbox and optional text prompt (or loads existing mask)
3. Sends data to mesh generation webapi (API handles SAM3/SAM3D)
4. Downloads generated mesh file

Usage Examples:
    # Generate mesh with interactive bbox and text prompt
    python gen_mesh.py --prompt "red mug" --output toy_bear.ply

    # Interactive bbox only (no text prompt)
    python gen_mesh.py --output model.ply

    # Save generated mask from API response
    python gen_mesh.py --prompt "toy bear" --output model.ply --save-mask mask.png

    # Use existing mask file
    python gen_mesh.py --mask path/to/mask.png --output model.ply

    # Specify API server and object description (mesh API runs on port 5001)
    python gen_mesh.py --api-url http://10.150.240.101:5001 --prompt "black mug" --output model.ply --save-mask mask.png

    # Custom camera resolution with prompt
    python gen_mesh.py --width 1280 --height 720 --prompt "cup" --output model.ply

    # Save captured images for debugging
    python gen_mesh.py --save-images ./debug_images --prompt "bottle" --output model.ply
    
    # Complete example with all options
    python gen_mesh.py --prompt "object" --output model.ply --save-mask mask.png --save-images ./debug
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
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image

# Add perception/webapi to path to import mesh client
project_root = Path(__file__).parent.parent.parent.parent
webapi_path = project_root / "perception" / "webapi"
if str(webapi_path) not in sys.path:
    sys.path.insert(0, str(webapi_path))

# Import the mesh generation client functions
from mesh_client_example import generate_mesh, encode_image_file


class MeshGenerationClient:
    """Wrapper for mesh generation client"""
    
    def __init__(self, api_url: str = "http://localhost:5001"):
        self.api_url = api_url
        self._check_health()
    
    def _check_health(self):
        """Check if the API server is available"""
        import requests
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ Connected to mesh generation API: {self.api_url}")
                print(f"  Device: {health_data.get('device', 'unknown')}")
                print(f"  Models loaded: {health_data.get('models_loaded', False)}")
                return True
        except Exception as e:
            print(f"✗ Cannot connect to API server: {self.api_url}")
            print(f"  Error: {e}")
            print(f"\n  Make sure the mesh generation API is running:")
            print(f"  cd perception/webapi && ./start_mesh_api.sh")
            sys.exit(1)
    
    def generate_from_array(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
        bbox_xyxy: Optional[List[int]] = None,
        mask: Optional[np.ndarray] = None,
        output_format: str = 'ply',
        mask_id: int = 0,
        seed: int = 42,
        return_mask: bool = True
    ) -> Tuple[bytes, Optional[np.ndarray]]:
        """
        Generate mesh from numpy array image
        
        Args:
            image: RGB image (H, W, 3)
            text: Text description of object
            bbox_xyxy: Bounding box [x1, y1, x2, y2]
            mask: Binary mask (H, W)
            output_format: 'ply' or 'stl'
            mask_id: Which mask to use if multiple detected
            seed: Random seed
            return_mask: Whether to return the generated mask
        
        Returns:
            Tuple of (mesh_bytes, mask_array)
        """
        # Convert bbox format if provided
        bbox_xywh = None
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            bbox_xywh = [x1, y1, x2-x1, y2-y1]
        
        # Save image to temp file (required by mesh_client_example)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img_path = tmp.name
            Image.fromarray(image).save(img_path)
        
        try:
            # Call the mesh generation API using the client function
            result = generate_mesh(
                api_url=self.api_url,
                image_path=img_path,
                text=text,
                bbox_xywh=bbox_xywh,
                output_format=output_format,
                mask_id=mask_id,
                seed=seed,
                return_mask=return_mask
            )
            
            # Check result
            if result['status'] != 'success':
                raise RuntimeError(f"API error: {result.get('error', 'Unknown error')}")
            
            # Decode mesh file from base64
            mesh_bytes = base64.b64decode(result['file_base64'])
            
            # Decode mask if available
            mask_array = None
            if return_mask and 'mask_base64' in result:
                mask_b64 = result['mask_base64']
                mask_bytes = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_bytes))
                mask_array = np.array(mask_img)
            
            # Print stats
            mesh_info = result['mesh_info']
            print(f"\n✓ Mesh generated successfully")
            if 'point_count' in mesh_info:
                print(f"  Points: {mesh_info['point_count']:,}")
            print(f"  Size: {mesh_info['file_size'] / 1024:.2f} KB")
            print(f"  Mask score: {result['mask_score']:.3f}")
            
            return mesh_bytes, mask_array
            
        finally:
            # Cleanup temp file
            if os.path.exists(img_path):
                os.unlink(img_path)


class RealSenseCapture:
    """Capture RGB-D frames from RealSense camera"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
    
    def start(self):
        """Start camera pipeline"""
        print(f"\n📷 Starting RealSense camera ({self.width}x{self.height} @ {self.fps}fps)...")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        profile = self.pipeline.start(config)
        
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        print(f"✓ Camera started")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  Focal length: ({intrinsics.fx:.1f}, {intrinsics.fy:.1f})")
        
        # Warm up camera
        for _ in range(30):
            self.pipeline.wait_for_frames()
    
    def capture_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture aligned RGB-D frame
        
        Returns:
            rgb: RGB image (H, W, 3)
            depth: Depth map (H, W) in millimeters
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture frame")
        
        # Convert to numpy arrays
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        return rgb, depth
    
    def stop(self):
        """Stop camera pipeline"""
        if self.pipeline:
            self.pipeline.stop()
            print("\n✓ Camera stopped")


def interactive_bbox_selection(rgb: np.ndarray) -> Optional[List[int]]:
    """
    Interactive bounding box selection using mouse drag
    
    Args:
        rgb: RGB image
    
    Returns:
        bbox_xyxy: [x1, y1, x2, y2] or None if cancelled
    """
    print("\n🖱️  Interactive bounding box selection:")
    print("  - Click and drag to draw bounding box around object")
    print("  - Press 'Enter' to accept bbox")
    print("  - Press 'r' to reset")
    print("  - Press 'q' to quit")
    
    bbox_start = None
    bbox_end = None
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal bbox_start, bbox_end, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_start = (x, y)
            bbox_end = (x, y)
            drawing = True
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            bbox_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            bbox_end = (x, y)
            drawing = False
            if bbox_start:
                x1, y1 = bbox_start
                x2, y2 = bbox_end
                print(f"  Bbox: [{min(x1,x2)}, {min(y1,y2)}, {max(x1,x2)}, {max(y1,y2)}]")
    
    cv2.namedWindow('Bbox Selection')
    cv2.setMouseCallback('Bbox Selection', mouse_callback)
    
    while True:
        display = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
        
        # Draw current bbox
        if bbox_start and bbox_end:
            x1, y1 = bbox_start
            x2, y2 = bbox_end
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show bbox dimensions
            w, h = abs(x2 - x1), abs(y2 - y1)
            cv2.putText(display, f"{w}x{h}", (min(x1, x2), min(y1, y2) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show instructions
        cv2.putText(display, "Draw bbox | Enter to accept | 'r' to reset | 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Bbox Selection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 and bbox_start and bbox_end:  # Enter key
            x1, y1 = bbox_start
            x2, y2 = bbox_end
            bbox_xyxy = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            print(f"\n✓ Bbox selected: {bbox_xyxy}")
            cv2.destroyAllWindows()
            return bbox_xyxy
        
        elif key == ord('r'):
            bbox_start = None
            bbox_end = None
            print("  Bbox reset")
        
        elif key == ord('q'):
            print("  Cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None


def load_mask_from_file(mask_path: str) -> np.ndarray:
    """Load mask from file"""
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    return mask_img


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D mesh from RealSense camera using webapi',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output mesh file path (e.g., model.ply)')
    
    parser.add_argument('--api-url', type=str, default='http://localhost:5001',
                       help='Mesh generation API URL (default: http://localhost:5001)')
    
    parser.add_argument('--mask', type=str,
                       help='Path to existing mask file (skip interactive selection)')
    
    parser.add_argument('--prompt', type=str,
                       help='Text prompt describing the object (e.g., "red mug", "toy bear")')
    
    parser.add_argument('--bbox', type=str,
                       help='Bounding box in format "x1,y1,x2,y2" (skip interactive selection)')
    
    parser.add_argument('--mask-id', type=int, default=0,
                       help='Which mask to use if multiple detected (default: 0 = highest score)')
    
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    parser.add_argument('--no-depth', action='store_true',
                       help='Do not send depth data (RGB + mask only)')
    
    parser.add_argument('--save-mask', type=str,
                       help='Path to save generated mask image from API')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" RealSense Mesh Generation Client")
    print("="*70)
    
    try:
        # Initialize API client
        client = MeshGenerationClient(args.api_url)
        
        # Initialize camera
        camera = RealSenseCapture(args.width, args.height, args.fps)
        camera.start()
        
        # Capture frame
        print("\n📸 Capturing frame...")
        rgb, depth = camera.capture_frame()
        print(f"✓ Frame captured: {rgb.shape}")
        
        # Get mask or bbox
        mask = None
        bbox_xyxy = None
        
        if args.mask:
            # Use existing mask file
            print(f"\n🎭 Loading mask from: {args.mask}")
            mask = load_mask_from_file(args.mask)
        else:
            # Get bounding box for API
            if args.bbox:
                # Parse bbox from command line
                try:
                    bbox_xyxy = [int(x) for x in args.bbox.split(',')]
                    if len(bbox_xyxy) != 4:
                        raise ValueError("Bbox must have 4 values")
                    print(f"\n📦 Using bbox: {bbox_xyxy}")
                    if args.prompt:
                        print(f"   Text prompt: '{args.prompt}'")
                except Exception as e:
                    print(f"\n✗ Invalid bbox format: {e}")
                    print("   Expected format: x1,y1,x2,y2")
                    camera.stop()
                    return 1
            else:
                # Interactive bbox selection
                bbox_xyxy = interactive_bbox_selection(rgb)
                if bbox_xyxy is None:
                    print("\n✗ No bbox selected. Exiting.")
                    camera.stop()
                    return 1
                if args.prompt:
                    print(f"   Text prompt: '{args.prompt}'")
        
        # # Save images if requested
        # if args.save_images:
        #     save_dir = Path(args.save_images)
        #     save_dir.mkdir(parents=True, exist_ok=True)
        #     cv2.imwrite(str(save_dir / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        #     if mask is not None:
        #         cv2.imwrite(str(save_dir / 'mask.png'), mask)
        #     if bbox_xyxy is not None:
        #         # Draw bbox on image for debugging
        #         debug_img = rgb.copy()
        #         cv2.rectangle(debug_img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 2)
        #         cv2.imwrite(str(save_dir / 'rgb_with_bbox.png'), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        #     if not args.no_depth:
        #         cv2.imwrite(str(save_dir / 'depth.png'), (depth / 16).astype(np.uint16))
        #     print(f"\n💾 Saved images to: {args.save_images}")
        
        # Generate mesh (API will handle SAM3/SAM3D internally)
        mesh_bytes, generated_mask = client.generate_from_array(
            image=rgb,
            text=args.prompt,
            bbox_xyxy=bbox_xyxy,
            mask=mask,
            output_format='ply',
            mask_id=args.mask_id,
            seed=42,
            return_mask=bool(args.save_mask)
        )
        
        # Save mesh file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(mesh_bytes)
        
        print(f"\n💾 Mesh saved to: {args.output}")
        print(f"   Size: {len(mesh_bytes) / 1024:.2f} KB")
        
        # Save generated mask if requested and available
        # Save generated mask if requested and available
        if args.save_mask and generated_mask is not None:
            mask_path = Path(args.save_mask)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), generated_mask)
            print(f"💾 Mask saved to: {args.save_mask}")
        
        # Stop camera
        camera.stop()
        
        print("\n" + "="*70)
        print("="*70)
        print(f"\nYou can view the mesh in:")
        print("  • CloudCompare")
        print("  • MeshLab")
        print("  • Blender")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'camera' in locals():
            try:
                camera.stop()
            except:
                pass


if __name__ == '__main__':
    sys.exit(main())
