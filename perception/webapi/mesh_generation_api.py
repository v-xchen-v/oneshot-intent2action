"""
REST API for 3D Mesh Generation from RGB Images

This API provides endpoints to:
1. Generate 3D mesh (PLY/STL) from RGB image with text prompt and/or bounding box
2. Health check and status endpoints

The pipeline:
1. SAM3: Segment object from image using text/bbox prompts -> mask
2. SAM3D: Generate 3D Gaussian splat from image + mask -> PLY
3. Optional: Convert PLY to STL format

Usage:
    python mesh_generation_api.py

API Endpoints:
    POST /api/generate_mesh - Generate 3D mesh from image
    GET /health - Health check
"""

import os
import sys
import json
import base64
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import traceback

# Set environment variables before importing sam3d_objects
os.environ["LIDRA_SKIP_INIT"] = "true"
if "CONDA_PREFIX" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]

# Add paths for SAM3 and SAM3D
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
notebook_path = Path(project_root) / "external" / "sam-3d-objects" / "notebook"
sys.path.insert(0, str(notebook_path))

# Import SAM3 for mask generation
from transformers import Sam3Processor, Sam3Model

# Import SAM3D for 3D mesh generation
from inference import Inference, load_image

# Import mesh processing utilities
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. STL conversion will be limited.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Advanced mesh processing will be limited.")


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


# Global model instances (loaded once at startup)
sam3_model = None
sam3_processor = None
sam3d_inference = None
device = None


def init_models():
    """Initialize SAM3 and SAM3D models at startup"""
    global sam3_model, sam3_processor, sam3d_inference, device
    
    print("="*70)
    print("Initializing models...")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load SAM3 for mask generation
    print("\n1. Loading SAM3 model (mask generation)...")
    try:
        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("✓ SAM3 loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load SAM3: {e}")
        raise
    
    # Load SAM3D for 3D mesh generation
    print("\n2. Loading SAM3D model (3D mesh generation)...")
    sam3d_config = os.path.join(project_root, "external/sam-3d-objects/checkpoints/hf/pipeline.yaml")
    
    if not os.path.exists(sam3d_config):
        print(f"✗ SAM3D config not found: {sam3d_config}")
        print("  Please download SAM3D checkpoints first.")
        raise FileNotFoundError(f"SAM3D config not found: {sam3d_config}")
    
    try:
        sam3d_inference = Inference(sam3d_config, compile=False)
        print("✓ SAM3D loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load SAM3D: {e}")
        raise
    
    print("\n" + "="*70)
    print("All models loaded successfully!")
    print("="*70 + "\n")


def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 encoded image to PIL Image"""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def encode_file(file_path: str) -> str:
    """Encode file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_mask_sam3(image: Image.Image, text: Optional[str] = None, 
                       bbox_xyxy: Optional[List[int]] = None,
                       mask_id: int = 0) -> np.ndarray:
    """
    Generate object mask using SAM3.
    
    Args:
        image: PIL Image
        text: Optional text description of object
        bbox_xyxy: Optional bounding box [x1, y1, x2, y2]
        mask_id: Which mask to return if multiple are generated (default: 0 = highest score)
        
    Returns:
        Binary mask as numpy array (H, W) with dtype bool
    """
    if text is None and bbox_xyxy is None:
        raise ValueError("Must provide either text or bbox_xyxy")
    
    # Prepare inputs based on what's provided
    if text and bbox_xyxy:
        # Combined mode: text + bbox
        inputs = sam3_processor(
            images=image,
            input_boxes=[[bbox_xyxy]],
            input_boxes_labels=[[1]],
            text=text,
            return_tensors="pt"
        ).to(device)
    elif bbox_xyxy:
        # Bbox only mode
        inputs = sam3_processor(
            images=image,
            input_boxes=[[bbox_xyxy]],
            input_boxes_labels=[[1]],
            return_tensors="pt"
        ).to(device)
    else:
        # Text only mode
        inputs = sam3_processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = sam3_model(**inputs)
    
    # Post-process results
    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    # Sort masks by scores descending
    if "scores" in results and len(results["scores"]) > 0:
        scores = results["scores"]
        sorted_indices = torch.argsort(scores, descending=True)
        masks = results["masks"][sorted_indices]
        scores = results["scores"][sorted_indices]
    else:
        masks = results["masks"]
        scores = torch.ones(len(masks))
    
    if len(masks) == 0:
        raise ValueError("No masks generated. Try adjusting text prompt or bounding box.")
    
    # Return requested mask
    if mask_id >= len(masks):
        raise ValueError(f"Requested mask_id={mask_id} but only {len(masks)} masks generated")
    
    mask = masks[mask_id].cpu().numpy().astype(bool)
    score = scores[mask_id].item()
    
    print(f"Generated {len(masks)} masks, using mask {mask_id} (score: {score:.3f})")
    
    return mask, score


def generate_mesh_sam3d(image_np: np.ndarray, mask: np.ndarray, 
                        output_path: str, seed: int = 42) -> Dict[str, Any]:
    """
    Generate 3D mesh using SAM3D.
    
    Args:
        image_np: Image as numpy array (H, W, 3)
        mask: Binary mask as numpy array (H, W)
        output_path: Path to save PLY file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with mesh info
    """
    # Run SAM3D inference
    output = sam3d_inference(image_np, mask, seed=seed)
    
    # Check if Gaussian splat was generated
    if 'gs' not in output or output['gs'] is None:
        raise ValueError("SAM3D failed to generate Gaussian splat")
    
    # Save PLY file
    output['gs'].save_ply(output_path)
    
    # Get file info
    file_size = os.path.getsize(output_path)
    point_count = output['gs'].get_xyz.shape[0]
    
    return {
        'file_path': output_path,
        'file_size': file_size,
        'point_count': point_count,
        'format': 'ply'
    }


def convert_ply_to_stl(ply_path: str, stl_path: str, 
                       poisson_depth: int = 8) -> Dict[str, Any]:
    """
    Convert PLY point cloud to STL mesh using Poisson surface reconstruction.
    
    Args:
        ply_path: Path to input PLY file
        stl_path: Path to save STL file
        poisson_depth: Poisson reconstruction depth (higher = more detail, slower)
        
    Returns:
        Dictionary with mesh info
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for STL conversion")
    
    # Load PLY as point cloud
    cloud = trimesh.load(ply_path)
    
    # If it's already a mesh, just export
    if hasattr(cloud, 'vertices') and hasattr(cloud, 'faces') and len(cloud.faces) > 0:
        mesh = cloud
    else:
        # Need to reconstruct surface from points
        # This is a basic approach - for better results, use dedicated tools
        try:
            # Try to get a convex hull as fallback
            if hasattr(cloud, 'vertices'):
                mesh = trimesh.convex.convex_hull(cloud.vertices)
            else:
                raise ValueError("Cannot extract mesh from PLY file")
        except Exception as e:
            raise ValueError(f"Failed to convert PLY to mesh: {e}")
    
    # Export to STL
    mesh.export(stl_path)
    
    # Get file info
    file_size = os.path.getsize(stl_path)
    
    return {
        'file_path': stl_path,
        'file_size': file_size,
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'format': 'stl'
    }


def process_mesh_with_scale(ply_path: str, stl_path: str,
                           reference_size_meters: float,
                           reference_axis: str = 'width') -> Dict[str, Any]:
    """
    Scale a PLY mesh to real-world dimensions and convert to STL.
    
    Args:
        ply_path: Path to input PLY file
        stl_path: Path to save output STL file
        reference_size_meters: Known real-world size in meters
        reference_axis: Which axis to use as reference ('width', 'height', 'depth')
        
    Returns:
        Dictionary with mesh info and dimensions
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("open3d is required for mesh processing with scaling")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    if mesh.is_empty():
        raise ValueError("Failed to load mesh from PLY file")
    
    # Get original dimensions
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [width, height, depth] = [X, Y, Z]
    
    # Map axis name to index
    axis_map = {'width': 0, 'height': 1, 'depth': 2}
    if reference_axis not in axis_map:
        raise ValueError(f"Invalid reference_axis: {reference_axis}. Must be 'width', 'height', or 'depth'")
    
    axis_index = axis_map[reference_axis]
    current_dimension = extent[axis_index]
    
    # Calculate scale factor
    scale_factor = reference_size_meters / current_dimension
    
    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    # Apply scale
    mesh.scale(scale_factor, center=mesh.get_center())
    
    # Get scaled dimensions
    bbox_scaled = mesh.get_axis_aligned_bounding_box()
    extent_scaled = bbox_scaled.get_extent()
    
    # Handle point cloud (no triangles) - reconstruct surface
    if len(mesh.triangles) == 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson surface reconstruction
        mesh_reconstructed, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh_reconstructed.remove_vertices_by_mask(vertices_to_remove)
        
        mesh = mesh_reconstructed
    
    # Simplify if too large
    original_triangle_count = len(mesh.triangles)
    if original_triangle_count > 50000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    
    # Recompute normals
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    # Export to STL
    o3d.io.write_triangle_mesh(stl_path, mesh, write_ascii=False)
    
    # Get file info
    file_size = os.path.getsize(stl_path)
    
    return {
        'file_path': stl_path,
        'file_size': file_size,
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.triangles),
        'format': 'stl',
        'scale_factor': scale_factor,
        'original_dimensions': {
            'width': float(extent[0]),
            'height': float(extent[1]),
            'depth': float(extent[2])
        },
        'scaled_dimensions_meters': {
            'width': float(extent_scaled[0]),
            'height': float(extent_scaled[1]),
            'depth': float(extent_scaled[2])
        },
        'reference_axis': reference_axis,
        'reference_size_meters': reference_size_meters
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': sam3_model is not None and sam3d_inference is not None,
        'device': str(device),
        'trimesh_available': TRIMESH_AVAILABLE,
        'open3d_available': OPEN3D_AVAILABLE
    })


@app.route('/api/generate_mesh', methods=['POST'])
def generate_mesh():
    """
    Generate 3D mesh from RGB image with text/bbox prompts.
    
    Request JSON:
    {
        "image": "base64_encoded_image",
        "text": "optional text description",
        "bbox_xywh": [x, y, width, height],  // optional
        "bbox_xyxy": [x1, y1, x2, y2],  // alternative to bbox_xywh
        "mask_id": 0,  // which mask to use if multiple generated
        "output_format": "ply" or "stl",  // default: "ply"
        "seed": 42,  // random seed for reproducibility
        "return_mask": false,  // whether to return the generated mask
        "return_file": true  // whether to return the 3D file (as base64 if true)
    }
    
    Response JSON:
    {
        "status": "success",
        "mesh_info": {
            "format": "ply" or "stl",
            "file_size": 12345,
            "point_count": 50000,  // for PLY
            "vertex_count": 10000,  // for STL
            "face_count": 20000  // for STL
        },
        "mask_score": 0.95,
        "mask_base64": "...",  // if return_mask=true
        "file_base64": "...",  // if return_file=true
        "session_id": "uuid"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'image' not in data:
            return jsonify({'error': 'Missing required field: image'}), 400
        
        if 'text' not in data and 'bbox_xywh' not in data and 'bbox_xyxy' not in data:
            return jsonify({'error': 'Must provide either text, bbox_xywh, or bbox_xyxy'}), 400
        
        # Parse parameters
        image_b64 = data['image']
        text = data.get('text')
        bbox_xywh = data.get('bbox_xywh')
        bbox_xyxy = data.get('bbox_xyxy')
        mask_id = data.get('mask_id', 0)
        output_format = data.get('output_format', 'ply').lower()
        seed = data.get('seed', 42)
        return_mask = data.get('return_mask', False)
        return_file = data.get('return_file', True)
        
        # Validate output format
        if output_format not in ['ply', 'stl']:
            return jsonify({'error': 'output_format must be "ply" or "stl"'}), 400
        
        # Convert bbox_xywh to bbox_xyxy if needed
        if bbox_xywh and not bbox_xyxy:
            x, y, w, h = bbox_xywh
            bbox_xyxy = [x, y, x + w, y + h]
        
        print(f"\n{'='*70}")
        print("Processing mesh generation request")
        print(f"{'='*70}")
        print(f"Text: {text}")
        print(f"BBox: {bbox_xyxy}")
        print(f"Output format: {output_format}")
        print(f"Mask ID: {mask_id}")
        
        # Decode image
        print("\n1. Decoding image...")
        image = decode_image(image_b64)
        print(f"✓ Image size: {image.size}")
        
        # Generate mask using SAM3
        print("\n2. Generating mask with SAM3...")
        mask, mask_score = generate_mask_sam3(image, text=text, bbox_xyxy=bbox_xyxy, mask_id=mask_id)
        print(f"✓ Mask generated (score: {mask_score:.3f})")
        print(f"  Coverage: {mask.sum() / mask.size * 100:.1f}% of image")
        
        # Create temporary directory for outputs
        session_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp(prefix=f"mesh_gen_{session_id}_")
        
        try:
            # Convert PIL image to numpy array for SAM3D
            image_np = np.array(image)
            
            # Generate 3D mesh with SAM3D
            print("\n3. Generating 3D mesh with SAM3D...")
            ply_path = os.path.join(temp_dir, "output.ply")
            mesh_info = generate_mesh_sam3d(image_np, mask, ply_path, seed=seed)
            print(f"✓ PLY generated: {mesh_info['point_count']:,} points, {mesh_info['file_size']/1024/1024:.2f} MB")
            
            # Convert to STL if requested
            if output_format == 'stl':
                print("\n4. Converting PLY to STL...")
                stl_path = os.path.join(temp_dir, "output.stl")
                mesh_info = convert_ply_to_stl(ply_path, stl_path)
                output_file = stl_path
                print(f"✓ STL generated: {mesh_info['vertex_count']:,} vertices, {mesh_info['face_count']:,} faces")
            else:
                output_file = ply_path
            
            # Prepare response
            response = {
                'status': 'success',
                'mesh_info': mesh_info,
                'mask_score': float(mask_score),
                'session_id': session_id
            }
            
            # Add mask if requested
            if return_mask:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_buffer = io.BytesIO()
                mask_img.save(mask_buffer, format='PNG')
                mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
                response['mask_base64'] = mask_b64
            
            # Add file if requested
            if return_file:
                file_b64 = encode_file(output_file)
                response['file_base64'] = file_b64
                response['file_name'] = os.path.basename(output_file)
            
            print(f"\n{'='*70}")
            print("✅ SUCCESS!")
            print(f"{'='*70}\n")
            
            return jsonify(response)
        
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/process_mesh', methods=['POST'])
def process_mesh():
    """
    Process an existing PLY mesh: scale to real-world dimensions and convert to STL.
    
    Request JSON:
    {
        "mesh_file": "base64_encoded_ply_file",
        "reference_size_meters": 0.5,
        "reference_axis": "width",  // 'width', 'height', or 'depth'
        "return_file": true  // whether to return the STL file as base64
    }
    
    Response JSON:
    {
        "status": "success",
        "mesh_info": {
            "format": "stl",
            "file_size": 12345,
            "vertex_count": 10000,
            "triangle_count": 20000,
            "scale_factor": 0.0123,
            "original_dimensions": {"width": 40.5, "height": 30.2, "depth": 25.1},
            "scaled_dimensions_meters": {"width": 0.5, "height": 0.373, "depth": 0.31},
            "reference_axis": "width",
            "reference_size_meters": 0.5
        },
        "mesh_file_base64": "..."  // if return_file=true
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'mesh_file' not in data:
            return jsonify({'error': 'Missing required field: mesh_file (base64 encoded PLY)'}), 400
        
        if 'reference_size_meters' not in data:
            return jsonify({'error': 'Missing required field: reference_size_meters'}), 400
        
        # Parse parameters
        reference_size_meters = float(data['reference_size_meters'])
        reference_axis = data.get('reference_axis', 'width')
        return_file = data.get('return_file', True)
        
        # Validate parameters
        if reference_size_meters <= 0:
            return jsonify({'error': 'reference_size_meters must be positive'}), 400
        
        if reference_axis not in ['width', 'height', 'depth']:
            return jsonify({'error': 'reference_axis must be "width", "height", or "depth"'}), 400
        
        print(f"\n{'='*60}")
        print("Processing mesh...")
        print(f"Reference: {reference_axis} = {reference_size_meters} meters")
        print(f"{'='*60}")
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Decode and save input PLY file
            ply_path = os.path.join(tmp_dir, 'input.ply')
            mesh_bytes = base64.b64decode(data['mesh_file'])
            with open(ply_path, 'wb') as f:
                f.write(mesh_bytes)
            
            print(f"✓ Input PLY saved: {len(mesh_bytes)} bytes")
            
            # Process mesh: scale and convert to STL
            stl_path = os.path.join(tmp_dir, 'output.stl')
            
            mesh_info = process_mesh_with_scale(
                ply_path=ply_path,
                stl_path=stl_path,
                reference_size_meters=reference_size_meters,
                reference_axis=reference_axis
            )
            
            print(f"✓ Mesh processed successfully")
            print(f"  Scale factor: {mesh_info['scale_factor']:.6f}")
            print(f"  Output size: {mesh_info['file_size']} bytes")
            print(f"  Triangles: {mesh_info['triangle_count']:,}")
            print(f"  Final dimensions (meters):")
            dims = mesh_info['scaled_dimensions_meters']
            print(f"    Width:  {dims['width']:.4f} m ({dims['width']*100:.2f} cm)")
            print(f"    Height: {dims['height']:.4f} m ({dims['height']*100:.2f} cm)")
            print(f"    Depth:  {dims['depth']:.4f} m ({dims['depth']*100:.2f} cm)")
            
            # Build response
            response = {
                'status': 'success',
                'mesh_info': mesh_info
            }
            
            # Encode and return STL file if requested
            if return_file:
                stl_b64 = encode_file(stl_path)
                response['mesh_file_base64'] = stl_b64
                print(f"✓ STL file encoded ({len(stl_b64)} chars base64)")
            
            print(f"{'='*60}\n")
            
            return jsonify(response), 200
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'details': traceback.format_exc()
        }), 500


@app.route('/api/generate_mesh_file', methods=['POST'])
def generate_mesh_file():
    """
    Generate 3D mesh and return as downloadable file.
    
    Same parameters as /api/generate_mesh but returns file directly.
    """
    try:
        # Process the request (reuse logic from generate_mesh)
        data = request.json
        
        # [Same validation and processing as generate_mesh endpoint]
        # For brevity, calling generate_mesh and converting response
        
        # This endpoint would be implemented similar to above but return send_file()
        return jsonify({'error': 'Not yet implemented - use /api/generate_mesh instead'}), 501
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Mesh Generation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize models
    init_models()
    
    # Start server
    print(f"\n{'='*70}")
    print(f"Starting Mesh Generation API server...")
    print(f"{'='*70}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"\nEndpoints:")
    print(f"  GET  {args.host}:{args.port}/health")
    print(f"  POST {args.host}:{args.port}/api/generate_mesh")
    print(f"  POST {args.host}:{args.port}/api/process_mesh")
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
