"""
Example client for the 6D Pose Tracking API

This demonstrates how to:
1. Create a tracking session
2. Initialize with first frame and mask
3. Stream RGB-D frames for continuous tracking
4. Retrieve pose estimates
"""

import requests
import cv2
import numpy as np
import base64
import json
from typing import Optional
import io
from PIL import Image


class PoseTrackingClient:
    """Client for interacting with the Pose Tracking API"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.session_id: Optional[str] = None
    
    def encode_image(self, image: np.ndarray, format: str = 'PNG') -> str:
        """Encode numpy array to base64 string"""
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            # Normalize if float
            if image.dtype in [np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def encode_mesh_file(self, mesh_path: str) -> str:
        """Encode mesh file to base64"""
        with open(mesh_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def create_session(
        self,
        mesh_path: str,
        cam_K: np.ndarray,
        mesh_scale: float = 0.01,
        est_refine_iter: int = 10,
        track_refine_iter: int = 5,
        activate_2d_tracker: bool = True,
        activate_kalman_filter: bool = True,
        kf_measurement_noise_scale: float = 0.05
    ) -> bool:
        """Create a new tracking session"""
        
        mesh_b64 = self.encode_mesh_file(mesh_path)
        
        # Detect mesh file type from extension
        import os
        file_ext = os.path.splitext(mesh_path)[1].lower().lstrip('.')
        if file_ext not in ['stl', 'ply', 'obj']:
            file_ext = 'stl'  # Default to stl
        
        payload = {
            'mesh_file': mesh_b64,
            'mesh_file_type': file_ext,
            'mesh_scale': mesh_scale,
            'cam_K': cam_K.tolist(),
            'est_refine_iter': est_refine_iter,
            'track_refine_iter': track_refine_iter,
            'activate_2d_tracker': activate_2d_tracker,
            'activate_kalman_filter': activate_kalman_filter,
            'kf_measurement_noise_scale': kf_measurement_noise_scale
        }
        
        response = requests.post(f"{self.api_url}/api/session/create", json=payload)
        
        if response.status_code == 201:
            data = response.json()
            self.session_id = data['session_id']
            print(f"Session created: {self.session_id}")
            print(f"Mesh: {data['mesh_vertices']} vertices, {data['mesh_faces']} faces")
            return True
        else:
            print(f"Error creating session: {response.json()}")
            return False
    
    def initialize(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        depth_scale: float = 1000.0,
        est_refine_iter: int = 10
    ) -> Optional[dict]:
        """Initialize tracking with first frame"""
        
        if not self.session_id:
            print("No active session. Create a session first.")
            return None
        
        payload = {
            'rgb': self.encode_image(rgb),
            'depth': self.encode_image(depth),
            'mask': self.encode_image(mask),
            'depth_scale': depth_scale,
            'est_refine_iter': est_refine_iter
        }
        
        response = requests.post(
            f"{self.api_url}/api/session/{self.session_id}/initialize",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Initialized. Frame count: {data['frame_count']}")
            return data['pose']
        else:
            print(f"Error initializing: {response.json()}")
            return None
    
    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        depth_scale: float = 1000.0,
        visualize: bool = False
    ) -> Optional[dict]:
        """Track object in new frame"""
        
        if not self.session_id:
            print("No active session. Create a session first.")
            return None
        
        payload = {
            'rgb': self.encode_image(rgb),
            'depth': self.encode_image(depth),
            'depth_scale': depth_scale,
            'visualize': visualize
        }
        
        response = requests.post(
            f"{self.api_url}/api/session/{self.session_id}/track",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error tracking: {response.json()}")
            return None
    
    def get_history(self) -> Optional[dict]:
        """Get pose history for current session"""
        
        if not self.session_id:
            print("No active session.")
            return None
        
        response = requests.get(f"{self.api_url}/api/session/{self.session_id}/history")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting history: {response.json()}")
            return None
    
    def close_session(self) -> bool:
        """Close current tracking session"""
        
        if not self.session_id:
            print("No active session.")
            return False
        
        response = requests.delete(f"{self.api_url}/api/session/{self.session_id}/close")
        
        if response.status_code == 200:
            print(f"Session {self.session_id} closed")
            self.session_id = None
            return True
        else:
            print(f"Error closing session: {response.json()}")
            return False
    
    def health_check(self) -> dict:
        """Check API health"""
        response = requests.get(f"{self.api_url}/api/health")
        return response.json()


def example_with_camera_stream():
    """Example: Track object using live camera stream"""
    
    # Initialize client
    client = PoseTrackingClient(api_url="http://localhost:5000")
    
    # Check server health
    health = client.health_check()
    print(f"Server health: {health}")
    
    # Camera intrinsics (example values - replace with your camera's)
    cam_K = np.array([
        [912.7279, 0.0, 667.5955],
        [0.0, 911.0028, 360.5406],
        [0.0, 0.0, 1.0]
    ])
    
    # Create session
    mesh_path = "/path/to/your/object.stl"
    success = client.create_session(
        mesh_path=mesh_path,
        cam_K=cam_K,
        mesh_scale=0.01,  # Convert mm to m
        activate_2d_tracker=True,
        activate_kalman_filter=True
    )
    
    if not success:
        return
    
    # Open camera (or use RealSense, etc.)
    cap_rgb = cv2.VideoCapture(0)  # RGB camera
    # For depth, you'd need an RGB-D camera like RealSense
    
    frame_idx = 0
    
    try:
        while True:
            ret, rgb = cap_rgb.read()
            if not ret:
                break
            
            # Get depth frame (placeholder - replace with actual depth from RGB-D camera)
            depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)
            
            if frame_idx == 0:
                # First frame: need initial mask
                # Option 1: Load pre-generated mask
                mask = cv2.imread('/path/to/initial_mask.png', cv2.IMREAD_GRAYSCALE)
                
                # Option 2: Generate mask interactively (click to select object)
                # mask = generate_mask_interactive(rgb)
                
                # Initialize tracking
                pose = client.initialize(rgb, depth, mask, depth_scale=1000.0)
                print(f"Initial pose: {pose}")
            else:
                # Track in subsequent frames
                result = client.track(rgb, depth, depth_scale=1000.0, visualize=True)
                
                if result and result['success']:
                    pose = result['pose']
                    print(f"Frame {frame_idx}: Translation: {pose['translation']}")
                    
                    # Display visualization if available
                    if 'visualization' in result:
                        vis_img = base64.b64decode(result['visualization'])
                        vis_array = np.array(Image.open(io.BytesIO(vis_img)))
                        cv2.imshow('Pose Tracking', cv2.cvtColor(vis_array, cv2.COLOR_RGB2BGR))
                    
                    # Check for exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_idx += 1
    
    finally:
        # Cleanup
        cap_rgb.release()
        cv2.destroyAllWindows()
        client.close_session()


def example_with_image_sequence():
    """Example: Track object using pre-recorded image sequence"""
    
    client = PoseTrackingClient(api_url="http://localhost:5000")
    
    # Camera intrinsics
    cam_K = np.array([
        [912.7279, 0.0, 667.5955],
        [0.0, 911.0028, 360.5406],
        [0.0, 0.0, 1.0]
    ])
    
    # Create session
    mesh_path = "/workspace/data/toy_bear_scaled.stl"
    client.create_session(
        mesh_path=mesh_path,
        cam_K=cam_K,
        mesh_scale=1.0,  # Already scaled
        activate_2d_tracker=True,
        activate_kalman_filter=True
    )
    
    # Image sequence paths
    rgb_dir = "/workspace/data/rgb"
    depth_dir = "/workspace/data/depth"
    mask_path = "/workspace/data/init_mask.png"
    
    import os
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    for i, rgb_file in enumerate(rgb_files):
        rgb = cv2.imread(os.path.join(rgb_dir, rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(os.path.join(depth_dir, rgb_file), -1)
        
        if i == 0:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pose = client.initialize(rgb, depth, mask)
        else:
            result = client.track(rgb, depth, visualize=True)
            pose = result['pose']
            
            print(f"Frame {i}: {pose['translation']}")
            
            # Save visualization
            if 'visualization' in result:
                vis_img = base64.b64decode(result['visualization'])
                with open(f'/workspace/output/vis_{i:04d}.png', 'wb') as f:
                    f.write(vis_img)
    
    # Get full history
    history = client.get_history()
    print(f"Total frames: {history['frame_count']}")
    
    # Save poses
    poses = np.array([p['matrix'] for p in history['poses']])
    np.save('/workspace/output/poses.npy', poses)
    
    client.close_session()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['camera', 'sequence'], default='sequence',
                        help='Run with live camera or image sequence')
    
    args = parser.parse_args()
    
    if args.mode == 'camera':
        example_with_camera_stream()
    else:
        example_with_image_sequence()
