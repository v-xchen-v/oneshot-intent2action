#!/usr/bin/env python
"""
Scale a 3D mesh (PLY/STL) to real-world metric dimensions and export as STL.

This script takes a 3D mesh file and scales it based on a known real-world dimension,
then exports it as an STL file suitable for FoundationPose and CAD applications.

Usage:
    python process_obj_mesh.py --mesh object.ply --reference_size_meters 0.3 --reference_axis height --output scaled.stl

Examples:
    # Scale a toy bear where the width is 0.5 meters
    python perception/scripts/process_obj_mesh.py \
        --mesh toy_bear.ply \
        --reference_size_meters 0.5 \
        --reference_axis width \
        --output toy_bear_scaled.stl
    
    # Scale based on height (Z-axis)
    python perception/scripts/process_obj_mesh.py \
        --mesh object.ply \
        --reference_size_meters 0.3 \
        --reference_axis height \
        --output object_scaled.stl
"""

import argparse
import os
import sys
import open3d as o3d
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Scale 3D mesh to real-world metric dimensions and export as STL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scale based on width (X-axis) being 0.5 meters:
  python process_obj_mesh.py \\
      --mesh toy_bear.ply \\
      --reference_size_meters 0.5 \\
      --reference_axis width \\
      --output toy_bear_scaled.stl

  # Scale based on height (Z-axis) being 0.3 meters:
  python process_obj_mesh.py \\
      --mesh object.ply \\
      --reference_size_meters 0.3 \\
      --reference_axis height \\
      --output object_scaled.stl

Reference Axes:
  - width:  X-axis dimension
  - height: Y-axis dimension  
  - depth:  Z-axis dimension

Notes:
  - Input can be PLY or STL format
  - Output is always STL format (triangular mesh)
  - Point clouds will be reconstructed into meshes using Poisson reconstruction
  - Large meshes (>50k triangles) will be simplified to reduce file size
"""
    )
    
    parser.add_argument('--mesh', type=str, required=True,
                        help='Path to input mesh file (PLY or STL)')
    parser.add_argument('--reference_size_meters', type=float, required=True,
                        help='Known real-world size in meters for the reference axis')
    parser.add_argument('--reference_axis', type=str, required=True,
                        choices=['width', 'height', 'depth'],
                        help='Which axis dimension to use as reference (width=X, height=Y, depth=Z)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output STL file path')
    
    args = parser.parse_args()
    
    # ========================================
    # Validation
    # ========================================
    
    print("\n" + "="*70)
    print(" Scale 3D Mesh to Real-World Metric Dimensions")
    print("="*70 + "\n")
    
    # Check input file exists
    if not os.path.exists(args.mesh):
        print(f"❌ Error: Input file not found: {args.mesh}")
        sys.exit(1)
    
    # Validate file format
    ext = args.mesh.split('.')[-1].lower()
    if ext not in ['ply', 'stl']:
        print(f"❌ Error: Unsupported file format: .{ext}")
        print("   Supported formats: .ply, .stl")
        sys.exit(1)
    
    # Validate reference size
    if args.reference_size_meters <= 0:
        print(f"❌ Error: Reference size must be positive, got {args.reference_size_meters}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}\n")
    
    # ========================================
    # Load Mesh
    # ========================================
    
    print(f"📦 Loading mesh...")
    print(f"   Input: {args.mesh}")
    
    try:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        
        if mesh.is_empty():
            raise RuntimeError("Failed to load mesh - file may be empty or corrupted")
        
        vertex_count = len(mesh.vertices)
        triangle_count = len(mesh.triangles)
        print(f"✓ Mesh loaded successfully")
        print(f"   Vertices: {vertex_count:,}")
        print(f"   Triangles: {triangle_count:,}\n")
        
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        sys.exit(1)
    
    # ========================================
    # Get Original Dimensions
    # ========================================
    
    print(f"📏 Computing original dimensions...")
    
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [width, height, depth] = [X, Y, Z]
    
    print(f"   Original dimensions (arbitrary units):")
    print(f"     Width (X):  {extent[0]:.6f}")
    print(f"     Height (Y): {extent[1]:.6f}")
    print(f"     Depth (Z):  {extent[2]:.6f}\n")
    
    # ========================================
    # Calculate Scale Factor
    # ========================================
    
    # Map axis name to index
    axis_map = {'width': 0, 'height': 1, 'depth': 2}
    axis_index = axis_map[args.reference_axis]
    
    current_dimension = extent[axis_index]
    scale_to_meters = args.reference_size_meters / current_dimension
    
    print(f"⚖️  Calculating scale factor...")
    print(f"   Reference axis: {args.reference_axis} ({['X', 'Y', 'Z'][axis_index]}-axis)")
    print(f"   Current {args.reference_axis}: {current_dimension:.6f} units")
    print(f"   Target {args.reference_axis}: {args.reference_size_meters:.6f} meters")
    print(f"   Scale factor: {scale_to_meters:.6f}\n")

    # ========================================
    # Clean Mesh
    # ========================================
    
    print(f"🧹 Cleaning mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    print(f"✓ Mesh cleaned\n")
    
    # ========================================
    # Apply Scale
    # ========================================
    
    print(f"🔧 Applying scale transformation...")
    mesh.scale(scale_to_meters, center=mesh.get_center())
    print(f"✓ Scale applied\n")
    
    # ========================================
    # Verify Scaled Dimensions
    # ========================================
    
    print(f"✅ Verifying scaled dimensions...")
    bbox_scaled = mesh.get_axis_aligned_bounding_box()
    extent_scaled = bbox_scaled.get_extent()
    
    print(f"   Scaled dimensions (meters):")
    print(f"     Width (X):  {extent_scaled[0]:.4f} m ({extent_scaled[0]*100:.2f} cm)")
    print(f"     Height (Y): {extent_scaled[1]:.4f} m ({extent_scaled[1]*100:.2f} cm)")
    print(f"     Depth (Z):  {extent_scaled[2]:.4f} m ({extent_scaled[2]*100:.2f} cm)\n")
    
    # Verify the reference axis matches expected size
    scaled_ref_dimension = extent_scaled[axis_index]
    if abs(scaled_ref_dimension - args.reference_size_meters) > 0.001:
        print(f"⚠️  Warning: Scaled {args.reference_axis} ({scaled_ref_dimension:.4f}m) differs from target ({args.reference_size_meters:.4f}m)")
    
    # ========================================
    # Prepare for STL Export
    # ========================================
    
    print(f"🔨 Preparing mesh for STL export...")
    
    # Check if mesh has triangles
    if len(mesh.triangles) == 0:
        print("   ⚠️  Mesh has no triangles (only vertices). Reconstructing surface from point cloud...")
        
        # Convert to point cloud for reconstruction
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        
        # Estimate normals (required for reconstruction)
        print("   Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson surface reconstruction
        print("   Running Poisson surface reconstruction...")
        mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices (optional cleanup)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh_reconstructed.remove_vertices_by_mask(vertices_to_remove)
        
        # Use the reconstructed mesh
        mesh = mesh_reconstructed
        print(f"   ✓ Reconstructed mesh with {len(mesh.triangles):,} triangles\n")
    else:
        print(f"   ✓ Mesh already has triangles ({len(mesh.triangles):,})\n")
    
    # Simplify mesh to reduce file size (if too many triangles)
    original_triangle_count = len(mesh.triangles)
    if original_triangle_count > 50000:
        print(f"🔻 Simplifying mesh...")
        print(f"   Current: {original_triangle_count:,} triangles")
        # Reduce to target number of triangles (adjust as needed)
        target_triangles = 50000
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        reduction_pct = (1 - len(mesh.triangles) / original_triangle_count) * 100
        print(f"   ✓ Simplified to {len(mesh.triangles):,} triangles ({reduction_pct:.1f}% reduction)\n")
    
    # Recompute normals (required for STL export)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    # ========================================
    # Export STL
    # ========================================
    
    print(f"💾 Exporting STL file...")
    print(f"   Output: {args.output}")
    
    try:
        o3d.io.write_triangle_mesh(args.output, mesh, write_ascii=False)
        
        # Show file size info
        file_size = os.path.getsize(args.output)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✓ STL file saved successfully")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Triangles: {len(mesh.triangles):,}")
        print(f"   Vertices: {len(mesh.vertices):,}\n")
        
    except Exception as e:
        print(f"❌ Error saving STL file: {e}")
        sys.exit(1)
    
    # ========================================
    # Summary
    # ========================================
    
    print("="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"\nScaled STL file: {args.output}")
    print(f"Reference: {args.reference_axis} = {args.reference_size_meters} meters")
    print(f"\nFinal dimensions:")
    print(f"  Width (X):  {extent_scaled[0]:.4f} m ({extent_scaled[0]*100:.2f} cm)")
    print(f"  Height (Y): {extent_scaled[1]:.4f} m ({extent_scaled[1]*100:.2f} cm)")
    print(f"  Depth (Z):  {extent_scaled[2]:.4f} m ({extent_scaled[2]*100:.2f} cm)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()