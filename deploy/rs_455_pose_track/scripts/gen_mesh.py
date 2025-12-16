#!/usr/bin/env python3
"""
Client script to generate 3D mesh from RealSense camera using webapi.

This script:
1. Captures RGB-D frames from RealSense camera
2. Gets object mask (from SAM or manual input)
3. Sends data to mesh generation webapi
4. Downloads generated mesh file

Usage Examples:
    # Generate mesh with interactive bbox and text prompt
    python gen_mesh.py --prompt "red mug" --output toy_bear.ply

    # Interactive bbox only (no text prompt)
    python gen_mesh.py --output model.ply

    # Use existing mask file
    python gen_mesh.py --mask path/to/mask.png --output model.ply

    # Specify API server and object description
    python gen_mesh.py --api-url http://10.150.240.101:5000 --prompt "toy bear" --output model.ply

    # Custom camera resolution with prompt
    python gen_mesh.py --width 1280 --height 720 --prompt "cup" --output model.ply

    # Save captured images for debugging
    python gen_mesh.py --save-images ./debug_images --prompt "bottle" --output model.ply
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import base64
import json
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import io
from PIL import Image
import torch


class MeshGenerationClient:
    """Client for mesh generation webapi"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self._check_health()
    
    def _check_health(self):
        """Check if the API server is available"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Connected to API server: {self.api_url}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to API server: {self.api_url}")
            print(f"  Error: {e}")
            print(f"\n  Make sure the webapi server is running:")
            print(f"  cd perception/webapi && ./start_api.sh")
            sys.exit(1)
    
    def encode_image(self, image: np.ndarray, format: str = 'PNG') -> str:
        """Encode numpy array to base64 string"""
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def generate_mesh(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> bytes:
        """
        Generate mesh from RGB image and mask
        
        Args:
            rgb: RGB image (H, W, 3)
            mask: Binary mask (H, W) or (H, W, 1)
            depth: Optional depth map (H, W)
        
        Returns:
            PLY file content as bytes
        """
        print("\n📤 Sending data to mesh generation API...")
        
        # Prepare mask
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        payload = {
            'rgb': self.encode_image(rgb),
            'mask': self.encode_image(mask_uint8)
        }
        
        if depth is not None:
            payload['depth'] = self.encode_image(depth)
        
        try:
            response = requests.post(
                f"{self.api_url}/api/mesh/generate",
                json=payload,
                timeout=300  # 5 minutes timeout for mesh generation
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Decode mesh file from base64
                mesh_b64 = result.get('mesh_file')
                if not mesh_b64:
                    raise ValueError("No mesh file in response")
                
                mesh_bytes = base64.b64decode(mesh_b64)
                
                print(f"✓ Mesh generated successfully")
                print(f"  Points: {result.get('point_count', 'N/A')}")
                print(f"  Size: {len(mesh_bytes) / 1024:.2f} KB")
                
                return mesh_bytes
            else:
                error_msg = response.json().get('error', 'Unknown error')
                raise RuntimeError(f"API error: {error_msg}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timeout - mesh generation took too long")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")


class SAM3MaskGenerator:
    """Generate mask using SAM3 with bbox and optional text prompt"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        """Load SAM3 model (lazy loading)"""
        if self.model is not None:
            return
        
        print("\n🔧 Loading SAM3 model...")
        from transformers import Sam3Processor, Sam3Model
        
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"✓ SAM3 model loaded on {self.device}")
    
    def generate_mask(
        self,
        image: np.ndarray,
        bbox_xyxy: List[int],
        text_prompt: Optional[str] = None,
        mask_id: int = 0
    ) -> np.ndarray:
        """
        Generate mask using bbox and optional text prompt
        
        Args:
            image: RGB image (H, W, 3)
            bbox_xyxy: Bounding box [x1, y1, x2, y2]
            text_prompt: Optional text description
            mask_id: Which mask to return (0 = highest score)
        
        Returns:
            mask: Binary mask (H, W)
        """
        self.load_model()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        print(f"\n🎭 Generating mask with SAM3...")
        print(f"   Bbox: {bbox_xyxy}")
        if text_prompt:
            print(f"   Prompt: '{text_prompt}'")
        
        # Prepare inputs
        if text_prompt:
            # Use both bbox and text
            inputs = self.processor(
                images=pil_image,
                input_boxes=[[bbox_xyxy]],
                input_boxes_labels=[[1]],
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Use bbox only
            inputs = self.processor(
                images=pil_image,
                input_boxes=[[bbox_xyxy]],
                input_boxes_labels=[[1]],
                return_tensors="pt"
            ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        # Sort by scores
        if "scores" in results and len(results["scores"]) > 0:
            scores = results["scores"]
            sorted_indices = torch.argsort(scores, descending=True)
            masks = results["masks"][sorted_indices]
            scores_sorted = results["scores"][sorted_indices]
            
            print(f"✓ Found {len(masks)} masks with scores: {scores_sorted.tolist()}")
            
            if mask_id >= len(masks):
                print(f"⚠️  Warning: mask_id {mask_id} >= {len(masks)}, using mask 0")
                mask_id = 0
            
            # Convert to numpy
            mask = masks[mask_id].cpu().numpy().astype(np.uint8) * 255
            print(f"   Using mask {mask_id} (score: {scores_sorted[mask_id]:.3f})")
            
            return mask
        else:
            raise RuntimeError("No masks generated by SAM3")


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
    
    parser.add_argument('--api-url', type=str, default='http://localhost:5000',
                       help='Mesh generation API URL (default: http://localhost:5000)')
    
    parser.add_argument('--mask', type=str,
                       help='Path to existing mask file (skip interactive selection)')
    
    parser.add_argument('--prompt', type=str,
                       help='Text prompt describing the object (e.g., "red mug", "toy bear")')
    
    parser.add_argument('--bbox', type=str,
                       help='Bounding box in format "x1,y1,x2,y2" (skip interactive selection)')
    
    parser.add_argument('--mask-id', type=int, default=0,
                       help='Which mask to use if multiple detected (default: 0 = highest score)')
    
    parser.add_argument('--width', type=int, default=640,
                       help='Camera width (default: 640)')
    
    parser.add_argument('--height', type=int, default=480,
                       help='Camera height (default: 480)')
    
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    
    parser.add_argument('--save-images', type=str,
                       help='Directory to save captured images for debugging')
    
    parser.add_argument('--no-depth', action='store_true',
                       help='Do not send depth data (RGB + mask only)')
    
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
        
        # Get or create mask
        if args.mask:
            print(f"\n🎭 Loading mask from: {args.mask}")
            mask = load_mask_from_file(args.mask)
        else:
            # Get bounding box
            if args.bbox:
                # Parse bbox from command line
                try:
                    bbox_xyxy = [int(x) for x in args.bbox.split(',')]
                    if len(bbox_xyxy) != 4:
                        raise ValueError("Bbox must have 4 values")
                    print(f"\n📦 Using bbox: {bbox_xyxy}")
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
            
            # Generate mask using SAM3
            sam3 = SAM3MaskGenerator()
            try:
                mask = sam3.generate_mask(
                    rgb,
                    bbox_xyxy,
                    text_prompt=args.prompt,
                    mask_id=args.mask_id
                )
            except Exception as e:
                print(f"\n✗ Failed to generate mask: {e}")
                camera.stop()
                return 1
        
        # Save images if requested
        if args.save_images:
            save_dir = Path(args.save_images)
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_dir / 'mask.png'), mask)
            if not args.no_depth:
                cv2.imwrite(str(save_dir / 'depth.png'), (depth / 16).astype(np.uint16))
            print(f"\n💾 Saved images to: {args.save_images}")
        
        # Stop camera
        camera.stop()
        
        # Generate mesh
        mesh_bytes = client.generate_mesh(
            rgb, 
            mask,
            depth if not args.no_depth else None
        )
        
        # Save mesh file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(mesh_bytes)
        
        print(f"\n💾 Mesh saved to: {args.output}")
        print(f"   Size: {len(mesh_bytes) / 1024:.2f} KB")
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
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
