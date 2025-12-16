#!/usr/bin/env python3
"""
Test script for WebAPI-based 3D mesh generation

This script tests the mesh generation API by:
1. Sending a test image with text/bbox prompts
2. Generating 3D mesh (PLY/STL format)
3. Verifying the response and output files

Prerequisites:
- API server must be running: ./perception/webapi/start_mesh_api.sh
- Test image must be available in test_case directory
- conda environment sam3d-objects must be activated

Usage:
    python perception/tests/test_webapi_mesh.py
    python perception/tests/test_webapi_mesh.py --api-url http://localhost:5001
    python perception/tests/test_webapi_mesh.py --image path/to/image.jpg --text "toy bear"
"""

import sys
import os
import argparse
import base64
import requests
import json
from pathlib import Path
import tempfile


def encode_image_file(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def decode_file(file_b64: str, output_path: str):
    """Decode base64 file and save"""
    file_bytes = base64.b64decode(file_b64)
    with open(output_path, 'wb') as f:
        f.write(file_bytes)
    print(f"✓ Saved file to: {output_path} ({len(file_bytes)} bytes)")


def test_health_check(api_url: str):
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        print(f"\nMake sure the API server is running:")
        print(f"  ./perception/webapi/start_mesh_api.sh")
        return False


def test_process_mesh(api_url: str, ply_path: str, 
                     reference_size_meters: float = 0.5,
                     reference_axis: str = 'width',
                     output_dir: str = None):
    """Test mesh processing endpoint (PLY to STL with scaling)"""
    print("\n" + "="*70)
    print("TEST 3: Process Mesh (PLY to STL with Scaling)")
    print("="*70)
    
    # Verify PLY file exists
    if not os.path.exists(ply_path):
        print(f"✗ PLY file not found: {ply_path}")
        print("  Skipping process_mesh test (requires PLY from previous test)")
        return True  # Don't fail if PLY doesn't exist
    
    print(f"PLY file: {ply_path}")
    print(f"Reference: {reference_axis} = {reference_size_meters} meters")
    
    # Encode PLY file
    print("\n📤 Encoding PLY file...")
    with open(ply_path, 'rb') as f:
        ply_b64 = base64.b64encode(f.read()).decode('utf-8')
    print(f"✓ PLY encoded ({len(ply_b64)} chars)")
    
    # Build request payload
    payload = {
        'mesh_file': ply_b64,
        'reference_size_meters': reference_size_meters,
        'reference_axis': reference_axis,
        'return_file': True
    }
    
    # Make API request
    print(f"\n🌐 Calling API: {api_url}/api/process_mesh")
    print("⏳ Processing mesh...")
    
    try:
        response = requests.post(
            f"{api_url}/api/process_mesh",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Mesh processing successful!")
            
            mesh_info = result.get('mesh_info', {})
            print(f"  Output format: {mesh_info.get('format', 'unknown')}")
            print(f"  File size: {mesh_info.get('file_size', 0):,} bytes")
            print(f"  Triangles: {mesh_info.get('triangle_count', 0):,}")
            print(f"  Scale factor: {mesh_info.get('scale_factor', 0):.6f}")
            
            if 'scaled_dimensions_meters' in mesh_info:
                dims = mesh_info['scaled_dimensions_meters']
                print(f"  Scaled dimensions:")
                print(f"    Width:  {dims['width']:.4f} m ({dims['width']*100:.2f} cm)")
                print(f"    Height: {dims['height']:.4f} m ({dims['height']*100:.2f} cm)")
                print(f"    Depth:  {dims['depth']:.4f} m ({dims['depth']*100:.2f} cm)")
            
            # Save output if directory provided
            if output_dir and result.get('mesh_file_base64'):
                os.makedirs(output_dir, exist_ok=True)
                stl_path = os.path.join(output_dir, "output_mesh_scaled.stl")
                decode_file(result['mesh_file_base64'], stl_path)
                print(f"\n✓ Scaled STL saved to: {output_dir}")
            
            return True
        else:
            error_data = response.json()
            print(f"✗ Mesh processing failed:")
            print(f"  Error: {error_data.get('error', 'Unknown error')}")
            if 'details' in error_data:
                print(f"  Details: {error_data['details']}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (>2 minutes)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_mesh_generation(api_url: str, image_path: str, text: str = None, 
                        bbox_xywh: list = None, output_format: str = 'ply',
                        output_dir: str = None):
    """Test mesh generation endpoint"""
    print("\n" + "="*70)
    print("TEST 2: Mesh Generation")
    print("="*70)
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    print(f"Image: {image_path}")
    print(f"Text prompt: {text}")
    print(f"BBox: {bbox_xywh}")
    print(f"Output format: {output_format}")
    
    # Encode image
    print("\n📤 Encoding image...")
    image_b64 = encode_image_file(image_path)
    print(f"✓ Image encoded ({len(image_b64)} chars)")
    
    # Build request payload
    payload = {
        'image': image_b64,
        'output_format': output_format,
        'mask_id': 0,
        'seed': 42,
        'return_mask': True,
        'return_file': True
    }
    
    if text:
        payload['text'] = text
    
    if bbox_xywh:
        payload['bbox_xywh'] = bbox_xywh
    
    # Make API request
    print(f"\n🌐 Calling API: {api_url}/api/generate_mesh")
    print("⏳ This may take 1-2 minutes for model inference...")
    
    try:
        response = requests.post(
            f"{api_url}/api/generate_mesh",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Mesh generation successful!")
            mesh_info = result.get('mesh_info', {})
            print(f"  Mesh format: {mesh_info.get('format', 'unknown')}")
            print(f"  Mask returned: {result.get('mask_base64') is not None}")
            print(f"  File size: {len(result.get('file_base64', ''))} chars (base64)")
            
            # Save outputs if directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save mesh file
                if result.get('file_base64'):
                    mesh_path = os.path.join(output_dir, f"output_mesh.{output_format}")
                    decode_file(result['file_base64'], mesh_path)
                
                # Save mask if available
                if result.get('mask_base64'):
                    mask_path = os.path.join(output_dir, "output_mask.png")
                    decode_file(result['mask_base64'], mask_path)
                
                print(f"\n✓ Outputs saved to: {output_dir}")
            
            return True
        else:
            error_data = response.json()
            print(f"✗ Mesh generation failed:")
            print(f"  Error: {error_data.get('error', 'Unknown error')}")
            if 'details' in error_data:
                print(f"  Details: {error_data['details']}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (>5 minutes)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test the mesh generation API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5001",
        help="Base URL of the API server (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--image",
        default="test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--text",
        default="toy bear",
        help="Text description of object to segment"
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as 'x,y,w,h' (e.g., '100,150,200,300')"
    )
    parser.add_argument(
        "--format",
        choices=['ply', 'stl'],
        default='ply',
        help="Output mesh format (default: ply)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files (default: temp directory)"
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check test"
    )
    parser.add_argument(
        "--test-process-mesh",
        action="store_true",
        help="Test only the process_mesh endpoint (requires --ply-file)"
    )
    parser.add_argument(
        "--ply-file",
        help="Path to input PLY file for process_mesh test"
    )
    parser.add_argument(
        "--reference-size",
        type=float,
        default=0.5,
        help="Reference size in meters for process_mesh test (default: 0.5)"
    )
    parser.add_argument(
        "--reference-axis",
        choices=['width', 'height', 'depth'],
        default='width',
        help="Reference axis for process_mesh test (default: width)"
    )
    
    args = parser.parse_args()
    
    # Parse bbox if provided
    bbox_xywh = None
    if args.bbox:
        try:
            bbox_xywh = [int(x) for x in args.bbox.split(',')]
            if len(bbox_xywh) != 4:
                raise ValueError("BBox must have 4 values")
        except ValueError as e:
            print(f"Error: Invalid bbox format: {e}")
            print("Expected format: 'x,y,width,height' (e.g., '100,150,200,300')")
            return 1
    
    # Use temp directory if no output directory specified
    output_dir = args.output_dir
    temp_dir_obj = None
    if not output_dir:
        temp_dir_obj = tempfile.TemporaryDirectory()
        output_dir = temp_dir_obj.name
        print(f"Using temporary output directory: {output_dir}")
    
    print("="*70)
    print("3D Mesh Generation API Test")
    print("="*70)
    print(f"API URL: {args.api_url}")
    print(f"Image: {args.image}")
    print(f"Text: {args.text}")
    print(f"BBox: {bbox_xywh}")
    print(f"Format: {args.format}")
    print(f"Output: {output_dir}")
    
    # Run tests
    success = True
    
    # Determine test mode
    if args.test_process_mesh:
        # Standalone process_mesh test
        if not args.ply_file:
            print(f"Error: --test-process-mesh requires --ply-file")
            return 1
        
        # Test 1: Health check (optional)
        if not args.skip_health:
            if not test_health_check(args.api_url):
                success = False
                return 1
        
        # Test 2: Process mesh only
        if not test_process_mesh(
            args.api_url,
            args.ply_file,
            reference_size_meters=args.reference_size,
            reference_axis=args.reference_axis,
            output_dir=output_dir
        ):
            success = False
    else:
        # Standard mesh generation test
        # Test 1: Health check
        if not args.skip_health:
            if not test_health_check(args.api_url):
                success = False
                # Don't continue if health check fails
                return 1
        
        # Test 2: Mesh generation
        if not test_mesh_generation(
            args.api_url,
            args.image,
            text=args.text,
            bbox_xywh=bbox_xywh,
            output_format=args.format,
            output_dir=output_dir
        ):
            success = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    if success:
        print("✓ All tests passed!")
        return_code = 0
    else:
        print("✗ Some tests failed")
        return_code = 1
    
    # Cleanup temp directory
    if temp_dir_obj:
        temp_dir_obj.cleanup()
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
