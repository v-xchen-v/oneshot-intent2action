import open3d as o3d
import numpy as np

# 1. Load your 3D output
file_path = "toy_bear.ply"  # Replace with your file path
ext = file_path.split('.')[-1].lower()
if ext not in ['ply', 'stl']:
    raise ValueError("Unsupported file format. Please use .ply or .stl")

if ext == 'ply':
    # pcd = o3d.io.read_point_cloud(file_path)
    # mesh = pcd
    mesh = o3d.io.read_triangle_mesh(file_path)
elif ext == 'stl':
    mesh = o3d.io.read_triangle_mesh(file_path)
    
if mesh.is_empty():
    raise RuntimeError("Failed to load mesh")


# 2. Get bounding box dimensions
bbox = mesh.get_axis_aligned_bounding_box()
extent = bbox.get_extent()  # [width, height, depth]
print(f"Object dimensions (arbitrary units): {extent}")

# 3. If you know a real-world dimension (e.g., object is 0.30m tall)
known_dimension_meters = 0.3  # Example: The object is 0.30 meters tall
current_dimension = extent[2]  # assume height is Z-axis
scale_to_meters = known_dimension_meters / current_dimension

mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

# 4. Apply scale
mesh.scale(scale_to_meters, center=mesh.get_center())

# 5. Get new dimensions after scaling
bbox_scaled = mesh.get_axis_aligned_bounding_box()
extent_scaled = bbox_scaled.get_extent()
print(f"Scaled dimensions (meters): {extent_scaled}")
print(f"  Width:  {extent_scaled[0]:.4f} m ({extent_scaled[0]*100:.2f} cm)")
print(f"  Height: {extent_scaled[1]:.4f} m ({extent_scaled[1]*100:.2f} cm)")
print(f"  Depth:  {extent_scaled[2]:.4f} m ({extent_scaled[2]*100:.2f} cm)")

# # 6. Save (use write_point_cloud for point clouds, not write_triangle_mesh)
# if ext == 'ply':
#     o3d.io.write_point_cloud("metric_scale_model.ply", mesh)
#     print("✓ Saved scaled model to: metric_scale_model.ply")
# else:
if True:
    # Check if mesh has triangles
    if len(mesh.triangles) == 0:
        print("⚠ Mesh has no triangles (only vertices). Reconstructing surface from point cloud...")
        
        # Convert to point cloud for reconstruction
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        
        # Estimate normals (required for reconstruction)
        print("  Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson surface reconstruction
        print("  Running Poisson surface reconstruction...")
        mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices (optional cleanup)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh_reconstructed.remove_vertices_by_mask(vertices_to_remove)
        
        # Use the reconstructed mesh
        mesh = mesh_reconstructed
        print(f"  ✓ Reconstructed mesh with {len(mesh.triangles):,} triangles")
    
    # Simplify mesh to reduce file size (if too many triangles)
    original_triangle_count = len(mesh.triangles)
    if original_triangle_count > 50000:
        print(f"  Simplifying mesh (current: {original_triangle_count:,} triangles)...")
        # Reduce to target number of triangles (adjust as needed)
        target_triangles = 50000
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        print(f"  ✓ Simplified to {len(mesh.triangles):,} triangles ({(1-len(mesh.triangles)/original_triangle_count)*100:.1f}% reduction)")
    
    # Recompute normals (required for STL export)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("metric_scale_model.stl", mesh, write_ascii=False)
    
    # Show file size info
    import os
    file_size = os.path.getsize("metric_scale_model.stl")
    print(f"✓ Saved scaled model to: metric_scale_model.stl ({file_size / 1024 / 1024:.2f} MB)")