"""
REST API for 6D Object Pose Tracking with Camera Streaming

This API provides endpoints to:
1. Initialize tracking session with mesh and initial mask
2. Process streaming RGB-D frames
3. Get pose estimates in real-time
"""

import os
import sys
import json
import base64
import uuid
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import torch
import trimesh
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import io
from PIL import Image

# Add the FoundationPose++ source to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
foundationpose_src = os.path.join(project_root, "external/FoundationPose-plus-plus/src")
foundationpose_path = os.path.join(project_root, "external/FoundationPose-plus-plus/FoundationPose")

if foundationpose_src not in sys.path:
    sys.path.insert(0, foundationpose_src)
if foundationpose_path not in sys.path:
    sys.path.insert(0, foundationpose_path)

# Import from the original tracking code
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
from obj_pose_track import (
    adjust_pose_to_image_point,
    get_pose_xy_from_image_point,
    get_mat_from_6d_pose_arr,
    get_6d_pose_arr_from_mat
)


# FoundationPose imports
from FoundationPose.estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    FoundationPose,
    trimesh_add_pure_colored_texture,
    draw_posed_3d_box,
    draw_xyz_axis,
)


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


@dataclass
class TrackingSession:
    """Represents an active tracking session"""
    session_id: str
    estimator: FoundationPose
    tracker_2d: Optional[Tracker_2D]
    kalman_filter: Optional[KalmanFilter6D]
    cam_K: np.ndarray
    mesh: trimesh.Trimesh
    bbox: np.ndarray
    to_origin: np.ndarray
    is_initialized: bool = False
    frame_count: int = 0
    kf_mean: Optional[np.ndarray] = None
    kf_covariance: Optional[np.ndarray] = None
    track_refine_iter: int = 5
    pose_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.pose_history is None:
            self.pose_history = []


# Global session storage (in production, use Redis or similar)
active_sessions: Dict[str, TrackingSession] = {}


def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 encoded image to numpy array"""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def encode_image(image: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    pil_img = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def pose_to_dict(pose: np.ndarray) -> dict:
    """Convert 4x4 pose matrix to dictionary format"""
    if torch.is_tensor(pose):
        pose = pose.cpu().numpy()
    
    if pose.ndim == 3:
        pose = pose[0]
    
    return {
        'matrix': pose.tolist(),
        'translation': pose[:3, 3].tolist(),
        'rotation_matrix': pose[:3, :3].tolist(),
        'quaternion': rotation_matrix_to_quaternion(pose[:3, :3]).tolist()
    }


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z]"""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()  # Returns [x, y, z, w]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'gpu_available': torch.cuda.is_available()
    })


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """
    Create a new tracking session
    
    Expected JSON payload:
    {
        "mesh_file": "base64_encoded_mesh_file",  # STL/PLY/OBJ
        "mesh_scale": 0.01,  # Scale factor (default 0.01 for mm to m)
        "cam_K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  # Camera intrinsics
        "est_refine_iter": 10,  # Initial pose refinement iterations
        "track_refine_iter": 5,  # Tracking refinement iterations
        "activate_2d_tracker": true,  # Enable Cutie 2D tracker
        "activate_kalman_filter": true,  # Enable Kalman filtering
        "kf_measurement_noise_scale": 0.05,  # Kalman filter noise scale
        "force_apply_color": false,  # Force mesh color
        "apply_color": [0, 159, 237]  # RGB color if force_apply_color
    }
    
    Returns:
    {
        "session_id": "unique_session_id",
        "message": "Session created successfully"
    }
    """
    try:
        data = request.json
        
        # Auto-close existing sessions to prevent device conflicts
        if len(active_sessions) > 0:
            sessions_to_close = list(active_sessions.keys())
            for sid in sessions_to_close:
                try:
                    session = active_sessions[sid]
                    # Clean up all session resources including models
                    if hasattr(session, 'estimator') and session.estimator is not None:
                        # Delete models owned by this estimator
                        if hasattr(session.estimator, 'scorer'):
                            del session.estimator.scorer
                        if hasattr(session.estimator, 'refiner'):
                            del session.estimator.refiner
                        if hasattr(session.estimator, 'glctx'):
                            del session.estimator.glctx
                        # Clear estimator's internal tensors
                        if hasattr(session.estimator, 'pts'):
                            del session.estimator.pts
                        if hasattr(session.estimator, 'normals'):
                            del session.estimator.normals
                        if hasattr(session.estimator, 'rot_grid'):
                            del session.estimator.rot_grid
                        if hasattr(session.estimator, 'mesh_tensors'):
                            del session.estimator.mesh_tensors
                        if hasattr(session.estimator, 'pose_last'):
                            del session.estimator.pose_last
                        del session.estimator
                    
                    if session.tracker_2d and not isinstance(session.tracker_2d, Tracker_2D):
                        # Delete Cutie tracker
                        del session.tracker_2d
                    if session.kalman_filter:
                        del session.kalman_filter
                    del active_sessions[sid]
                except Exception as e:
                    print(f"Warning: Failed to close session {sid}: {e}")
            
            # Force garbage collection and clear CUDA cache multiple times
            import gc
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Decode and load mesh
        mesh_b64 = data.get('mesh_file')
        if mesh_b64:
            mesh_bytes = base64.b64decode(mesh_b64)
            mesh_file = io.BytesIO(mesh_bytes)
            # Determine file type from mesh_path hint or default to stl
            file_type = data.get('mesh_file_type', 'stl')
            mesh = trimesh.load(mesh_file, file_type=file_type)
        else:
            mesh_path = data.get('mesh_path')
            if not mesh_path or not os.path.exists(mesh_path):
                return jsonify({'error': 'mesh_file or mesh_path required'}), 400
            mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # Apply scale
        mesh_scale = data.get('mesh_scale', 0.01)
        mesh.apply_scale(mesh_scale)
        
        # Apply color if needed
        if data.get('force_apply_color', False):
            # TODO: now is hard-code color here, parse in color by webapi input
            color = np.array(data.get('apply_color', [0, 159, 237]))
            mesh = trimesh_add_pure_colored_texture(mesh, color=color, resolution=10)
        
        # Get mesh bounding box
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        
        # Camera intrinsics
        cam_K = np.array(data.get('cam_K'))
        
        # Ensure we're using the correct CUDA device and set default tensor type
        if torch.cuda.is_available():
            CUDA_DEVICE = int(os.environ.get('CUDA_DEVICE', '0'))
            torch.cuda.set_device(CUDA_DEVICE)
            torch.cuda.empty_cache()
        
        # Set default tensor type to CUDA to ensure all tensors are created on GPU
        original_type = torch.get_default_dtype()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        try:
            # Create fresh model instances for each session
            with torch.cuda.device(0):
                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                
                # Initialize FoundationPose estimator
                estimator = FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=mesh.vertex_normals,
                    mesh=mesh,
                    scorer=scorer,
                    refiner=refiner,
                    glctx=glctx,
                )
        finally:
            # Always reset default tensor type
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # Initialize 2D tracker
        tracker_2d = None
        if data.get('activate_2d_tracker', False):
            tracker_2d = Cutie()
        else:
            tracker_2d = Tracker_2D()
        
        # Initialize Kalman filter
        kalman_filter = None
        if data.get('activate_kalman_filter', False):
            kf_noise_scale = data.get('kf_measurement_noise_scale', 0.05)
            kalman_filter = KalmanFilter6D(kf_noise_scale)
        
        # Create session
        session = TrackingSession(
            session_id=session_id,
            estimator=estimator,
            tracker_2d=tracker_2d,
            kalman_filter=kalman_filter,
            cam_K=cam_K,
            mesh=mesh,
            bbox=bbox,
            to_origin=to_origin,
            track_refine_iter=data.get('track_refine_iter', 5)
        )
        
        active_sessions[session_id] = session
        
        return jsonify({
            'session_id': session_id,
            'message': 'Session created successfully',
            'mesh_vertices': len(mesh.vertices),
            'mesh_faces': len(mesh.faces)
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/initialize', methods=['POST'])
def initialize_tracking(session_id: str):
    """
    Initialize tracking with first frame and mask
    
    Expected JSON payload:
    {
        "rgb": "base64_encoded_rgb_image",
        "depth": "base64_encoded_depth_image",  # 16-bit PNG or raw depth in mm
        "mask": "base64_encoded_mask",  # Binary mask (0/255)
        "depth_scale": 1000  # Scale to convert depth to meters (default 1000 for mm)
    }
    
    Returns:
    {
        "success": true,
        "pose": {...},  # Initial pose estimate
        "frame_count": 1
    }
    """
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        data = request.json
        
        # Decode images
        rgb = decode_image(data['rgb'])
        if rgb.shape[-1] == 4:  # RGBA
            rgb = rgb[..., :3]
        
        depth_encoded = decode_image(data['depth'])
        if depth_encoded.ndim == 3:  # RGB depth image
            depth_encoded = depth_encoded[..., 0]
        
        # Convert depth to meters
        depth_scale = data.get('depth_scale', 1000.0)
        depth = depth_encoded.astype(np.float32) / depth_scale
        depth[(depth < 0.001) | (depth >= np.inf)] = 0
        
        mask_img = decode_image(data['mask'])
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        mask = (mask_img > 127).astype(np.uint8) * 255
        
        # Register initial pose
        est_refine_iter = data.get('est_refine_iter', 10)
        pose = session.estimator.register(
            K=session.cam_K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=est_refine_iter
        )
        
        # Initialize Kalman filter
        if session.kalman_filter is not None:
            session.kf_mean, session.kf_covariance = session.kalman_filter.initiate(
                get_6d_pose_arr_from_mat(pose)
            )
        
        # Initialize 2D tracker
        if session.tracker_2d is not None:
            session.tracker_2d.initialize(
                rgb,
                init_info={"mask": (mask_img > 127)},
            )
        
        session.is_initialized = True
        session.frame_count = 1
        session.pose_history.append(pose.reshape(4, 4))
        
        return jsonify({
            'success': True,
            'pose': pose_to_dict(pose),
            'frame_count': session.frame_count
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/track', methods=['POST'])
def track_frame(session_id: str):
    """
    Track object in new frame
    
    Expected JSON payload:
    {
        "rgb": "base64_encoded_rgb_image",
        "depth": "base64_encoded_depth_image",
        "depth_scale": 1000,
        "visualize": false  # Return visualization image
    }
    
    Returns:
    {
        "success": true,
        "pose": {...},
        "frame_count": N,
        "visualization": "base64_image" (if visualize=true)
    }
    """
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        if not session.is_initialized:
            return jsonify({'error': 'Session not initialized. Call /initialize first'}), 400
        
        data = request.json
        
        # Decode images
        rgb = decode_image(data['rgb'])
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        
        depth_encoded = decode_image(data['depth'])
        if depth_encoded.ndim == 3:
            depth_encoded = depth_encoded[..., 0]
        
        depth_scale = data.get('depth_scale', 1000.0)
        depth = depth_encoded.astype(np.float32) / depth_scale
        depth[(depth < 0.001) | (depth >= np.inf)] = 0
        
        # 2D tracking (if enabled)
        bbox_2d = None
        if session.tracker_2d is not None:
            bbox_2d = session.tracker_2d.track(rgb)
            
            # Adjust last pose using 2D bbox center
            if session.kalman_filter is None:
                session.estimator.pose_last = adjust_pose_to_image_point(
                    ob_in_cam=session.estimator.pose_last,
                    K=torch.from_numpy(session.cam_K).float(),
                    x=bbox_2d[0] + bbox_2d[2] / 2,
                    y=bbox_2d[1] + bbox_2d[3] / 2
                )
            else:
                # Update Kalman filter with pose and 2D measurement
                session.kf_mean, session.kf_covariance = session.kalman_filter.update(
                    session.kf_mean,
                    session.kf_covariance,
                    get_6d_pose_arr_from_mat(session.estimator.pose_last)
                )
                
                measurement_xy = np.array(get_pose_xy_from_image_point(
                    ob_in_cam=session.estimator.pose_last,
                    K=torch.from_numpy(session.cam_K).float(),
                    x=bbox_2d[0] + bbox_2d[2] / 2,
                    y=bbox_2d[1] + bbox_2d[3] / 2
                ))
                
                session.kf_mean, session.kf_covariance = session.kalman_filter.update_from_xy(
                    session.kf_mean,
                    session.kf_covariance,
                    measurement_xy
                )
                
                session.estimator.pose_last = torch.from_numpy(
                    get_mat_from_6d_pose_arr(session.kf_mean[:6])
                ).unsqueeze(0).to(session.estimator.pose_last.device)
        
        # 6D pose tracking
        pose = session.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=session.cam_K,
            iteration=session.track_refine_iter
        )
        
        # Kalman filter prediction
        if session.tracker_2d is not None and session.kalman_filter is not None:
            session.kf_mean, session.kf_covariance = session.kalman_filter.predict(
                session.kf_mean,
                session.kf_covariance
            )
        
        session.frame_count += 1
        session.pose_history.append(pose.reshape(4, 4))
        
        # Generate visualization if requested
        vis_image = None
        if data.get('visualize', False):
            center_pose = pose @ np.linalg.inv(session.to_origin)
            vis_color = draw_posed_3d_box(
                session.cam_K,
                img=rgb,
                ob_in_cam=center_pose,
                bbox=session.bbox
            )
            vis_color = draw_xyz_axis(
                vis_color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=session.cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            vis_image = encode_image(vis_color)
        
        response = {
            'success': True,
            'pose': pose_to_dict(pose),
            'frame_count': session.frame_count,
        }
        
        if vis_image:
            response['visualization'] = vis_image
        
        if bbox_2d:
            # Convert numpy types to native Python types for JSON serialization
            response['bbox_2d'] = [int(x) for x in bbox_2d]
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/history', methods=['GET'])
def get_pose_history(session_id: str):
    """
    Get pose history for a session
    
    Returns:
    {
        "session_id": "...",
        "frame_count": N,
        "poses": [...]  # List of poses
    }
    """
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        poses = [pose_to_dict(pose) for pose in session.pose_history]
        
        return jsonify({
            'session_id': session_id,
            'frame_count': session.frame_count,
            'poses': poses
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session/<session_id>/close', methods=['DELETE'])
def close_session(session_id: str):
    """
    Close a tracking session and free resources
    """
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        # Clear GPU memory
        del session.estimator
        if session.tracker_2d:
            del session.tracker_2d
        torch.cuda.empty_cache()
        
        del active_sessions[session_id]
        
        return jsonify({
            'success': True,
            'message': 'Session closed successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for sid, session in active_sessions.items():
        sessions.append({
            'session_id': sid,
            'frame_count': session.frame_count,
            'is_initialized': session.is_initialized
        })
    
    return jsonify({
        'active_sessions': len(active_sessions),
        'sessions': sessions
    }), 200


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='6D Pose Tracking API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting Pose Tracking API Server on {args.host}:{args.port}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
