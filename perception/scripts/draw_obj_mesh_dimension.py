#!/usr/bin/env python
"""
Visualize 3D model dimensions from PLY or STL files.
Supports both Gaussian splat PLY files and mesh files (PLY/STL).
"""
import sys
import os
import argparse

# Skip heavy SAM3D initialization for lightweight tools
os.environ['LIDRA_SKIP_INIT'] = 'true'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# Import SAM3D utilities
import torch
from sam3d_objects.model.backbone.tdfy_dit.representations import Gaussian

def load_model(file_path):
    """
    Load 3D model from PLY or STL file.
    Returns a dictionary with vertices, faces (if mesh), and model type.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.ply':
        # Try to load as Gaussian splat first
        try:
            gs = Gaussian(
                aabb=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
                sh_degree=0
            )
            gs.load_ply(file_path)
            xyz = gs.get_xyz.detach().cpu().numpy()
            return {
                'vertices': xyz,
                'faces': None,
                'type': 'gaussian_splat',
                'model': gs
            }
        except:
            # Fall back to loading as mesh
            pass
    
    # Load as mesh (PLY or STL)
    try:
        mesh = trimesh.load(file_path)
        return {
            'vertices': np.array(mesh.vertices),
            'faces': np.array(mesh.faces) if hasattr(mesh, 'faces') else None,
            'type': 'mesh',
            'model': mesh
        }
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")

def get_dimensions(model_data):
    """Calculate bounding box dimensions from model data."""
    xyz = model_data['vertices']
    
    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)
    extent = max_bound - min_bound
    center = (max_bound + min_bound) / 2.0
    
    return {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'extent': extent,
        'center': center,
        'xyz': xyz,
        'faces': model_data['faces'],
        'type': model_data['type']
    }

def print_dimensions(dims):
    """Print dimension information."""
    print(f"\nModel type: {dims['type']}")
    if dims['type'] == 'gaussian_splat':
        print(f"Point count: {len(dims['xyz']):,}")
    else:
        print(f"Vertices: {len(dims['xyz']):,}")
        if dims['faces'] is not None:
            print(f"Faces: {len(dims['faces']):,}")
    
    print(f"\nBounding Box Dimensions (arbitrary units):")
    print(f"  Width (X):  {dims['extent'][0]:.6f}")
    print(f"  Height (Y): {dims['extent'][1]:.6f}")
    print(f"  Depth (Z):  {dims['extent'][2]:.6f}")
    print(f"\nCenter: [{dims['center'][0]:.6f}, {dims['center'][1]:.6f}, {dims['center'][2]:.6f}]")
    print(f"Min bound: [{dims['min_bound'][0]:.6f}, {dims['min_bound'][1]:.6f}, {dims['min_bound'][2]:.6f}]")
    print(f"Max bound: [{dims['max_bound'][0]:.6f}, {dims['max_bound'][1]:.6f}, {dims['max_bound'][2]:.6f}]")

def visualize_matplotlib(dims, output_dir="output_visualizations"):
    """Create lightweight matplotlib visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    xyz = dims['xyz']
    min_bound = dims['min_bound']
    max_bound = dims['max_bound']
    center = dims['center']
    extent = dims['extent']
    
    # Subsample points for faster visualization (only for point clouds)
    if dims['type'] != 'mesh' and len(xyz) > 10000:
        indices = np.random.choice(len(xyz), 10000, replace=False)
        xyz_viz = xyz[indices]
    else:
        xyz_viz = xyz
    
    # Create 2x2 subplot figure with higher DPI
    fig = plt.figure(figsize=(20, 16), dpi=100)
    
    views = [
        (1, "Front View (XY)", 0, 90),
        (2, "Top View (XZ)", 0, 0),
        (3, "Side View (YZ)", 90, 90),
        (4, "Isometric View", 45, 45),
    ]
    
    for idx, title, elev, azim in views:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Plot model (point cloud or mesh)
        if dims['type'] == 'mesh' and dims['faces'] is not None:
            # Render mesh faces with better quality
            faces = dims['faces']
            
            # Use more faces for better quality (up to 15000)
            if len(faces) > 15000:
                indices = np.random.choice(len(faces), 15000, replace=False)
                faces = faces[indices]
            
            # Create face collection
            face_collection = []
            for face in faces:
                face_collection.append(xyz[face])
            
            # Render mesh with improved settings
            # Main surface with better opacity and color
            poly = Poly3DCollection(face_collection, 
                                   alpha=0.7,  # More opaque for clarity
                                   facecolor='#87CEEB',  # Sky blue
                                   edgecolor='#1E3A5F',  # Dark blue edges
                                   linewidth=0.2)
            poly.set_edgecolor('#1E3A5F')
            poly.set_facecolor('#87CEEB')
            ax.add_collection3d(poly)
        else:
            # Plot as point cloud
            ax.scatter(xyz_viz[:, 0], xyz_viz[:, 1], xyz_viz[:, 2], 
                      c='lightblue', s=1, alpha=0.5, label='Points')
        
        # Draw bounding box
        # Define the 8 corners of the box
        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ])
        
        # Draw the 12 edges of the bounding box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
        ]
        
        for edge in edges:
            points = corners[edge]
            ax.plot3D(*points.T, 'r-', linewidth=2)
        
        # Draw dimension lines at corners
        # X-axis (Width) - Red
        ax.plot3D([min_bound[0], max_bound[0]], 
                 [min_bound[1], min_bound[1]], 
                 [min_bound[2], min_bound[2]], 
                 'r-', linewidth=3, label=f'Width (X): {extent[0]:.4f}')
        
        # Y-axis (Height) - Green
        ax.plot3D([min_bound[0], min_bound[0]], 
                 [min_bound[1], max_bound[1]], 
                 [min_bound[2], min_bound[2]], 
                 'g-', linewidth=3, label=f'Height (Y): {extent[1]:.4f}')
        
        # Z-axis (Depth) - Blue
        ax.plot3D([min_bound[0], min_bound[0]], 
                 [min_bound[1], min_bound[1]], 
                 [min_bound[2], max_bound[2]], 
                 'b-', linewidth=3, label=f'Depth (Z): {extent[2]:.4f}')
        
        # Draw coordinate frame at center
        arrow_length = max(extent) * 0.2
        ax.quiver(center[0], center[1], center[2], 
                 arrow_length, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2)
        ax.quiver(center[0], center[1], center[2], 
                 0, arrow_length, 0, color='green', arrow_length_ratio=0.3, linewidth=2)
        ax.quiver(center[0], center[1], center[2], 
                 0, 0, arrow_length, color='blue', arrow_length_ratio=0.3, linewidth=2)
        
        ax.set_xlabel('X (Width)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (Height)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (Depth)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio with slight padding for better view
        max_range = max(extent) / 2 * 1.1  # Add 10% padding
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
        
        # Enhanced grid and background
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    plt.suptitle('3D Model Dimensions Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = f"{output_dir}/dimensions_matplotlib.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nSaved visualization: {output_path}")
    plt.close()

def save_report(dims, output_dir="output_visualizations"):
    """Save text report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/dimensions_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("3D Model Dimensions Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Width (X-axis):  {dims['extent'][0]:.6f} units\n")
        f.write(f"Height (Y-axis): {dims['extent'][1]:.6f} units\n")
        f.write(f"Depth (Z-axis):  {dims['extent'][2]:.6f} units\n\n")
        f.write(f"Center position: [{dims['center'][0]:.6f}, {dims['center'][1]:.6f}, {dims['center'][2]:.6f}]\n\n")
        f.write(f"Min bound: [{dims['min_bound'][0]:.6f}, {dims['min_bound'][1]:.6f}, {dims['min_bound'][2]:.6f}]\n")
        f.write(f"Max bound: [{dims['max_bound'][0]:.6f}, {dims['max_bound'][1]:.6f}, {dims['max_bound'][2]:.6f}]\n\n")
        f.write("Axis colors in visualizations:\n")
        f.write("  - Red:   X-axis (Width)\n")
        f.write("  - Green: Y-axis (Height)\n")
        f.write("  - Blue:  Z-axis (Depth)\n")
    
    print(f"Saved report: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize 3D model dimensions from PLY or STL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize Gaussian splat PLY
  python draw_object_mesh_dimensions.py --input toy_bear.ply
  
  # Visualize STL mesh
  python draw_object_mesh_dimensions.py --input toy_bear.stl
  
  # Specify output directory
  python draw_object_mesh_dimensions.py --input toy_bear.ply --output my_visualizations
"""
    )
    
    parser.add_argument('--input', type=str, 
                        help='Input 3D model file (PLY or STL)')
    parser.add_argument('--output', '-o', type=str, default='output_visualizations',
                        help='Output directory for visualizations (default: output_visualizations)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: {args.input} not found!")
        sys.exit(1)
    
    print(f"Loading 3D model from: {args.input}")
    
    # Load model (PLY or STL)
    try:
        model_data = load_model(args.input)
        print(f"✓ Loaded {model_data['type']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Calculate dimensions
    dims = get_dimensions(model_data)
    
    # Print dimensions
    print_dimensions(dims)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_matplotlib(dims, args.output)
    
    # Save report
    save_report(dims, args.output)
    
    print(f"\n✓ Done! Check the '{args.output}/' directory for outputs.")
    print("Visualization shows: model + bounding box + dimension lines + coordinate axes")

if __name__ == "__main__":
    main()
