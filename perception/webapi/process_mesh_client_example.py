"""
Client example for the Process Mesh API

This script demonstrates how to use the process_mesh API to scale PLY meshes
to real-world dimensions and convert them to STL format.

Usage:
    python process_mesh_client_example.py --mesh input.ply --reference-size 0.5 --reference-axis width --output scaled.stl
    
Examples:
    # Scale a toy bear PLY where the width is 0.5 meters
    python perception/webapi/process_mesh_client_example.py \\
        --mesh toy_bear.ply \\
        --reference-size 0.5 \\
        --reference-axis width \\
        --output toy_bear_scaled.stl
    
    # Scale based on height (Y-axis) being 0.3 meters
    python perception/webapi/process_mesh_client_example.py \\
        --mesh object.ply \\
        --reference-size 0.3 \\
        --reference-axis height \\
        --output object_scaled.stl
"""

import os
import sys
import argparse
import base64
import requests
import json
from pathlib import Path


def encode_file(file_path: str) -> str:
    """Encode file to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def decode_file(file_b64: str, output_path: str):
    """Decode base64 file and save"""
    file_bytes = base64.b64decode(file_b64)
    with open(output_path, 'wb') as f:
        f.write(file_bytes)


def process_mesh(api_url: str, mesh_path: str, 
                reference_size_meters: float,
                reference_axis: str = 'width',
                return_file: bool = True) -> dict:
    """
    Call the process mesh API.
    
    Args:
        api_url: Base URL of the API (e.g., "http://localhost:5001")
        mesh_path: Path to input PLY mesh file
        reference_size_meters: Known real-world size in meters
        reference_axis: 'width', 'height', or 'depth'
        return_file: Whether to return the STL file
        
    Returns:
        API response dictionary
    """
    # Encode mesh file
    print(f"📤 Encoding mesh file: {mesh_path}")
    mesh_b64 = encode_file(mesh_path)
    
    # Build request payload
    payload = {
        'mesh_file': mesh_b64,
        'reference_size_meters': reference_size_meters,
        'reference_axis': reference_axis,
        'return_file': return_file
    }
    
    # Make API request
    print(f"\n🌐 Calling API: {api_url}/api/process_mesh")
    print(f"   Reference: {reference_axis} = {reference_size_meters} meters")
    print(f"⏳ Processing mesh...")
    
    response = requests.post(
        f"{api_url}/api/process_mesh",
        json=payload,
        timeout=120  # 2 minutes timeout
    )
    
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Process PLY mesh: scale to real-world dimensions and convert to STL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5001",
        help="Base URL of the API server (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to input PLY mesh file"
    )
    parser.add_argument(
        "--reference-size",
        type=float,
        required=True,
        help="Known real-world size in meters for the reference axis"
    )
    parser.add_argument(
        "--reference-axis",
        choices=['width', 'height', 'depth'],
        default='width',
        help="Which axis to use as reference (default: width)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output STL file path"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.mesh):
        print(f"❌ Error: Input file not found: {args.mesh}")
        return 1
    
    # Verify file is PLY
    if not args.mesh.lower().endswith('.ply'):
        print(f"⚠️  Warning: Input file should be .ply format")
    
    print("="*70)
    print("Process Mesh API Client")
    print("="*70)
    print(f"API URL: {args.api_url}")
    print(f"Input: {args.mesh}")
    print(f"Reference: {args.reference_axis} = {args.reference_size} meters")
    print(f"Output: {args.output}")
    print("="*70)
    
    try:
        # Call API
        response = process_mesh(
            args.api_url,
            args.mesh,
            args.reference_size,
            args.reference_axis
        )
        
        print(f"\n📥 Response received")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Mesh processing successful!")
            
            # Display mesh info
            mesh_info = result.get('mesh_info', {})
            print(f"\n📊 Mesh Information:")
            print(f"  Format: {mesh_info.get('format', 'unknown')}")
            print(f"  File size: {mesh_info.get('file_size', 0):,} bytes")
            print(f"  Vertices: {mesh_info.get('vertex_count', 0):,}")
            print(f"  Triangles: {mesh_info.get('triangle_count', 0):,}")
            print(f"  Scale factor: {mesh_info.get('scale_factor', 0):.6f}")
            
            # Display dimensions
            if 'original_dimensions' in mesh_info:
                orig = mesh_info['original_dimensions']
                print(f"\n📏 Original Dimensions (arbitrary units):")
                print(f"  Width:  {orig['width']:.6f}")
                print(f"  Height: {orig['height']:.6f}")
                print(f"  Depth:  {orig['depth']:.6f}")
            
            if 'scaled_dimensions_meters' in mesh_info:
                scaled = mesh_info['scaled_dimensions_meters']
                print(f"\n📏 Scaled Dimensions (meters):")
                print(f"  Width:  {scaled['width']:.4f} m ({scaled['width']*100:.2f} cm)")
                print(f"  Height: {scaled['height']:.4f} m ({scaled['height']*100:.2f} cm)")
                print(f"  Depth:  {scaled['depth']:.4f} m ({scaled['depth']*100:.2f} cm)")
            
            # Save output file
            if result.get('mesh_file_base64'):
                print(f"\n💾 Saving output file...")
                decode_file(result['mesh_file_base64'], args.output)
                file_size = os.path.getsize(args.output)
                print(f"✓ Saved to: {args.output} ({file_size:,} bytes)")
            
            print(f"\n{'='*70}")
            print("✅ SUCCESS!")
            print(f"{'='*70}\n")
            return 0
            
        else:
            error_data = response.json()
            print(f"\n❌ Error: {error_data.get('error', 'Unknown error')}")
            if 'details' in error_data:
                print(f"\nDetails:\n{error_data['details']}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        print(f"\nMake sure the API server is running:")
        print(f"  ./perception/webapi/start_mesh_api.sh")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
