"""
Client example for the 3D Mesh Generation API

This script demonstrates how to use the mesh generation API to create
3D meshes (PLY/STL) from RGB images with text and/or bounding box prompts.

Usage:
    python client_example.py --image path/to/image.jpg --text "toy bear" --output mesh.ply
    
Examples:
    # Text prompt only
    python perception/webapi/mesh_client_example.py \\
        --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \\
        --text "toy bear" \\
        --output toy_bear.ply
    
    # Text + bounding box (recommended for best results)
    python perception/webapi/mesh_client_example.py \\
        --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \\
        --text "toy bear" \\
        --bbox_xywh 100,150,200,300 \\
        --output toy_bear.ply \\
        --format stl
    
    # Bounding box only
    python perception/webapi/mesh_client_example.py \\
        --image ./test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png \\
        --bbox_xywh 100,150,200,300 \\
        --output toy_bear.ply
"""

import os
import sys
import argparse
import base64
import requests
import json
from pathlib import Path
from PIL import Image


def encode_image_file(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def decode_file(file_b64: str, output_path: str):
    """Decode base64 file and save"""
    file_bytes = base64.b64decode(file_b64)
    print(f'Saving generated mesh file to {output_path}...')
    with open(output_path, 'wb') as f:
        f.write(file_bytes)
    print(f'Saved generated mesh file to {output_path}')


def generate_mesh(api_url: str, image_path: str, text: str = None, 
                  bbox_xywh: list = None, output_format: str = 'ply',
                  mask_id: int = 0, seed: int = 42,
                  return_mask: bool = True) -> dict:
    """
    Call the mesh generation API.
    
    Args:
        api_url: Base URL of the API (e.g., "http://localhost:5001")
        image_path: Path to input image
        text: Optional text description of object
        bbox_xywh: Optional bounding box [x, y, width, height]
        output_format: 'ply' or 'stl'
        mask_id: Which mask to use if multiple generated
        seed: Random seed for reproducibility
        return_mask: Whether to return the generated mask
        
    Returns:
        API response dictionary
    """
    # Encode image
    print(f"📤 Encoding image: {image_path}")
    image_b64 = encode_image_file(image_path)
    
    # Build request payload
    payload = {
        'image': image_b64,
        'output_format': output_format,
        'mask_id': mask_id,
        'seed': seed,
        'return_mask': return_mask,
        'return_file': True
    }
    
    if text:
        payload['text'] = text
    
    if bbox_xywh:
        payload['bbox_xywh'] = bbox_xywh
    
    # Make API request
    print(f"\n🌐 Calling API: {api_url}/api/generate_mesh")
    print(f"   Text: {text}")
    print(f"   BBox: {bbox_xywh}")
    print(f"   Format: {output_format}")
    print(f"   Mask ID: {mask_id}")
    print("\nWaiting for response (this may take 30-60 seconds)...\n")
    
    response = requests.post(
        f"{api_url}/api/generate_mesh",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}\n{response.text}")
    
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description='Client for 3D Mesh Generation API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input RGB image')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path (PLY or STL)')
    
    # Prompting options (at least one required)
    parser.add_argument('--text', type=str,
                        help='Text description of object')
    parser.add_argument('--bbox_xywh', type=str,
                        help='Bounding box in format "x,y,width,height"')
    parser.add_argument('--bbox_xyxy', type=str,
                        help='Bounding box in format "x1,y1,x2,y2"')
    
    # Optional parameters
    parser.add_argument('--format', type=str, default='ply', choices=['ply', 'stl'],
                        help='Output format (default: ply)')
    parser.add_argument('--mask_id', type=int, default=0,
                        help='Mask index to use if multiple generated (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--api_url', type=str, default='http://localhost:5001',
                        help='API base URL (default: http://localhost:5001)')
    parser.add_argument('--save_mask', type=str,
                        help='Optional: Save generated mask to this path')
    
    args = parser.parse_args()
    
    # Validation
    if not args.text and not args.bbox_xywh and not args.bbox_xyxy:
        parser.error("Must provide at least --text, --bbox_xywh, or --bbox_xyxy")
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Parse bounding box
    bbox_xywh = None
    if args.bbox_xywh:
        try:
            bbox_xywh = [int(x) for x in args.bbox_xywh.split(',')]
            if len(bbox_xywh) != 4:
                raise ValueError()
        except:
            print(f"❌ Error: Invalid bbox_xywh format. Expected 'x,y,w,h'")
            sys.exit(1)
    elif args.bbox_xyxy:
        try:
            x1, y1, x2, y2 = [int(x) for x in args.bbox_xyxy.split(',')]
            bbox_xywh = [x1, y1, x2-x1, y2-y1]
        except:
            print(f"❌ Error: Invalid bbox_xyxy format. Expected 'x1,y1,x2,y2'")
            sys.exit(1)
    
    print("\n" + "="*70)
    print(" 3D Mesh Generation Client")
    print("="*70 + "\n")
    
    try:
        # Check API health
        print("🏥 Checking API health...")
        health_response = requests.get(f"{args.api_url}/health")
        
        if health_response.status_code != 200:
            print(f"❌ API not responding at {args.api_url}")
            print(f"   Make sure the API server is running:")
            print(f"   python perception/webapi/mesh_generation_api.py")
            sys.exit(1)
        
        health_data = health_response.json()
        print(f"✓ API is healthy")
        print(f"  Device: {health_data.get('device')}")
        print(f"  Models loaded: {health_data.get('models_loaded')}")
        
        # Generate mesh
        result = generate_mesh(
            api_url=args.api_url,
            image_path=args.image,
            text=args.text,
            bbox_xywh=bbox_xywh,
            output_format=args.format,
            mask_id=args.mask_id,
            seed=args.seed,
            return_mask=args.save_mask is not None
        )
        
        if result['status'] != 'success':
            print(f"❌ API returned error: {result.get('error')}")
            sys.exit(1)
        
        # Save mesh file
        print(f"\n💾 Saving mesh file...")
        decode_file(result['file_base64'], args.output)
        
        mesh_info = result['mesh_info']
        print(f"✓ Mesh saved: {args.output}")
        print(f"  Format: {mesh_info['format']}")
        print(f"  File size: {mesh_info['file_size'] / 1024 / 1024:.2f} MB")
        
        if 'point_count' in mesh_info:
            print(f"  Point count: {mesh_info['point_count']:,}")
        if 'vertex_count' in mesh_info:
            print(f"  Vertex count: {mesh_info['vertex_count']:,}")
            print(f"  Face count: {mesh_info['face_count']:,}")
        
        print(f"\n  Mask score: {result['mask_score']:.3f}")
        
        # Save mask if requested
        if args.save_mask and 'mask_base64' in result:
            print(f"\n🎭 Saving mask...")
            decode_file(result['mask_base64'], args.save_mask)
            print(f"✓ Mask saved: {args.save_mask}")
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\nGenerated mesh: {args.output}")
        print(f"Session ID: {result['session_id']}")
        print("\nYou can view the mesh file in:")
        print("  • CloudCompare")
        print("  • MeshLab")
        print("  • Blender")
        print("  • Online viewers (e.g., https://3dviewer.net/)")
        print("\n" + "="*70 + "\n")
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Cannot connect to API at {args.api_url}")
        print(f"   Make sure the API server is running:")
        print(f"   python perception/webapi/mesh_generation_api.py")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
