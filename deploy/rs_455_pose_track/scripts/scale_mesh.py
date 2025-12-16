#!/usr/bin/env python3
"""
Interactive Mesh Scaling Script with Visualization

This script helps you scale a PLY mesh to real-world dimensions by:
1. Visualizing the mesh with coordinate axes
2. Displaying current dimensions
3. Prompting for real-world measurements
4. Calling the mesh processing API to generate a scaled STL file

Usage:
    python scale_mesh.py --mesh input.ply --output scaled.stl
    python scale_mesh.py --mesh model.ply --output scaled.stl --api-url http://10.150.240.101:5001
    
The script will show you the mesh dimensions and ask you to measure the object
in real-world (e.g., "The toy bear is 0.25 meters tall").
"""

import argparse
import base64
import os
import sys
import requests
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("❌ Error: open3d is required")
    print("   Install it with: pip install open3d")
    sys.exit(1)


def show_mesh_dimensions(mesh, mesh_path):
    """
    Display mesh dimensions with instructions to visualize in VS Code
    
    Args:
        mesh: Open3D triangle mesh or point cloud
        mesh_path: Path to the mesh file
    """
    print("\n" + "="*70)
    print("📊 MESH DIMENSIONS")
    print("="*70)
    
    # Compute mesh properties
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [width, height, depth] = [X, Y, Z]
    center = mesh.get_center()
    
    # Display dimensions
    print(f"\n📏 Current Mesh Dimensions (arbitrary units):")
    print(f"   Width  (X-axis): {extent[0]:.4f}")
    print(f"   Height (Y-axis): {extent[1]:.4f}")
    print(f"   Depth  (Z-axis): {extent[2]:.4f}")
    print(f"\n📍 Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    
    # Instructions for visualization
    print("\n" + "="*70)
    print("🎨 VISUALIZATION INSTRUCTIONS")
    print("="*70)
    print("\nTo view the mesh with coordinate axes in VS Code:")
    print(f"\n  1. Install the 'Python PLY Preview' extension in VS Code")
    print(f"  2. Open the mesh file in VS Code: {os.path.abspath(mesh_path)}")
    print(f"  3. The preview will show the coordinate system:")
    print(f"     • RED axis   = X-axis (Width)")
    print(f"     • GREEN axis = Y-axis (Height)")
    print(f"     • BLUE axis  = Z-axis (Depth)")
    print(f"\n  4. Look at the mesh and note which dimension you can measure")
    print(f"     in the real world (e.g., height, width, or depth)")
    print("\n" + "="*70)
    
    return extent


def get_user_measurement(extent):
    """
    Prompt user for real-world measurement
    
    Args:
        extent: [width, height, depth] from mesh bounding box
        
    Returns:
        tuple: (reference_axis, reference_size_meters)
    """
    print("\n" + "="*70)
    print("📐 REAL-WORLD MEASUREMENT")
    print("="*70)
    print("\nNow measure the object in the real world!")
    print("\nFor example:")
    print("  • If the toy bear is 0.25 meters (25 cm) tall → measure HEIGHT")
    print("  • If the object is 0.15 meters (15 cm) wide → measure WIDTH")
    print("  • If the object is 0.30 meters (30 cm) deep → measure DEPTH")
    
    # Display current dimensions again for reference
    print(f"\n📊 Current mesh dimensions (arbitrary units):")
    print(f"   1. Width  (X-axis): {extent[0]:.4f}")
    print(f"   2. Height (Y-axis): {extent[1]:.4f}")
    print(f"   3. Depth  (Z-axis): {extent[2]:.4f}")
    
    # Get axis choice
    while True:
        print("\n" + "-"*70)
        axis_input = input("Which dimension did you measure? [width/height/depth or 1/2/3]: ").strip().lower()
        
        axis_map = {
            'width': 'width', 'w': 'width', '1': 'width', 'x': 'width',
            'height': 'height', 'h': 'height', '2': 'height', 'y': 'height',
            'depth': 'depth', 'd': 'depth', '3': 'depth', 'z': 'depth'
        }
        
        if axis_input in axis_map:
            reference_axis = axis_map[axis_input]
            break
        else:
            print("❌ Invalid input. Please enter: width, height, depth (or 1, 2, 3)")
    
    # Get measurement value
    while True:
        print(f"\n💡 You selected: {reference_axis.upper()}")
        size_input = input(f"Enter the real-world {reference_axis} in METERS (e.g., 0.25 for 25cm): ").strip()
        
        try:
            reference_size = float(size_input)
            if reference_size <= 0:
                print("❌ Size must be positive!")
                continue
            if reference_size > 10:
                confirm = input(f"⚠️  {reference_size}m seems large. Continue? [y/n]: ").strip().lower()
                if confirm != 'y':
                    continue
            break
        except ValueError:
            print("❌ Invalid number. Please enter a decimal value (e.g., 0.25)")
    
    print(f"\n✓ Reference: {reference_axis} = {reference_size} meters ({reference_size*100:.2f} cm)")
    
    return reference_axis, reference_size


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
                return_file: bool = True):
    """
    Call the process mesh API using client method pattern.
    
    Args:
        api_url: Base URL of the API (e.g., "http://localhost:5001")
        mesh_path: Path to input PLY mesh file
        reference_size_meters: Known real-world size in meters
        reference_axis: 'width', 'height', or 'depth'
        return_file: Whether to return the STL file
        
    Returns:
        requests.Response object
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


def call_process_mesh_api(api_url: str, mesh_path: str, 
                          reference_size: float, reference_axis: str,
                          output_path: str):
    """
    Call the mesh processing API to generate scaled STL
    
    Args:
        api_url: Base URL of the API
        mesh_path: Path to input PLY file
        reference_size: Real-world size in meters
        reference_axis: 'width', 'height', or 'depth'
        output_path: Where to save the output STL file
    """
    print("\n" + "="*70)
    print("🌐 CALLING MESH PROCESSING API")
    print("="*70)
    print(f"API URL: {api_url}")
    print(f"Input: {mesh_path}")
    print(f"Reference: {reference_axis} = {reference_size} meters")
    print(f"Output: {output_path}")
    print("="*70)
    
    try:
        # Call API using client method
        response = process_mesh(
            api_url,
            mesh_path,
            reference_size,
            reference_axis
        )
        
        print(f"\n📥 Response received")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Mesh processing successful!")
            
            # Display mesh info
            mesh_info = result.get('mesh_info', {})
            if mesh_info:
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
                decode_file(result['mesh_file_base64'], output_path)
                file_size = os.path.getsize(output_path)
                print(f"✓ Saved to: {output_path} ({file_size:,} bytes)")
                return True
            else:
                print("⚠️  No mesh file returned by API")
                return False
                
        else:
            error_data = response.json()
            print(f"\n❌ Error: {error_data.get('error', 'Unknown error')}")
            if 'details' in error_data:
                print(f"\nDetails:\n{error_data['details']}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout! The API took too long to respond.")
        print("   Try again or check if the API server is running.")
        return False
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error! Cannot reach API at {api_url}")
        print("   Make sure the mesh processing API server is running.")
        print("   Start it with: ./perception/webapi/start_mesh_api.sh")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Interactive mesh scaling with visualization and API processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with visualization
  python scale_mesh.py --mesh toy_bear.ply --output toy_bear_scaled.stl
  
  # Specify API URL
  python scale_mesh.py --mesh object.ply --output object_scaled.stl --api-url http://192.168.1.100:5001
  
  # Non-interactive mode (skip visualization)
  python scale_mesh.py --mesh toy.ply --output toy_scaled.stl --no-viz --axis height --size 0.25
        """
    )
    
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to input PLY mesh file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output STL file"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5001",
        help="Base URL of the mesh processing API server (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (requires --axis and --size)"
    )
    parser.add_argument(
        "--axis",
        choices=['width', 'height', 'depth'],
        help="Reference axis (required if --no-viz)"
    )
    parser.add_argument(
        "--size",
        type=float,
        help="Reference size in meters (required if --no-viz)"
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
    print("🔧 INTERACTIVE MESH SCALING TOOL")
    print("="*70)
    print(f"Input: {args.mesh}")
    print(f"Output: {args.output}")
    print(f"API: {args.api_url}")
    print("="*70)
    
    # Load mesh
    print("\n📂 Loading mesh file...")
    try:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if mesh.is_empty():
            # Try as point cloud
            mesh = o3d.io.read_point_cloud(args.mesh)
            if mesh.is_empty():
                print("❌ Failed to load mesh file")
                return 1
            print("✓ Loaded as point cloud")
        else:
            print("✓ Loaded as triangle mesh")
            # Compute normals for better visualization
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        return 1
    
    # Get dimensions and reference measurement
    if args.no_viz:
        # Non-interactive mode
        if not args.axis or not args.size:
            print("❌ Error: --axis and --size are required when using --no-viz")
            return 1
        
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        
        print(f"\n📏 Mesh Dimensions (arbitrary units):")
        print(f"   Width:  {extent[0]:.4f}")
        print(f"   Height: {extent[1]:.4f}")
        print(f"   Depth:  {extent[2]:.4f}")
        
        reference_axis = args.axis
        reference_size = args.size
        
    else:
        # Interactive mode with dimension display
        extent = show_mesh_dimensions(mesh, args.mesh)
        reference_axis, reference_size = get_user_measurement(extent)
    
    # Call API to process mesh
    success = call_process_mesh_api(
        args.api_url,
        args.mesh,
        reference_size,
        reference_axis,
        args.output
    )
    
    if success:
        print("\n" + "="*70)
        print("🎉 COMPLETE!")
        print("="*70)
        print(f"✓ Scaled STL file saved to: {args.output}")
        print(f"✓ You can now use this file for pose tracking or other applications")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("❌ FAILED")
        print("="*70)
        print("The mesh processing did not complete successfully.")
        print("Please check the error messages above and try again.")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
