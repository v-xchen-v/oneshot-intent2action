#!/usr/bin/env python3
"""
Test script for WebAPI-based 6D pose tracking using test case data

This script tests the pose tracking API by:
1. Starting with a test case image sequence
2. Creating a tracking session via API
3. Initializing with first frame and mask
4. Tracking all frames in sequence
5. Comparing results with direct tracking method

Prerequisites:
- API server must be running (python perception/webapi/pose_tracking_api.py)
- Test case data must be available in test_case directory
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add webapi client to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../webapi'))
from client_example import PoseTrackingClient


def get_sorted_frame_list(directory):
    """Get sorted list of image files"""
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    
    # Sort by numeric prefix
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def test_webapi_tracking(
    api_url: str,
    test_case_dir: str,
    mesh_path: str,
    init_mask_path: str,
    cam_K: list,
    output_dir: str = None,
    visualize: bool = True
):
    """
    Test pose tracking via WebAPI
    
    Args:
        api_url: URL of the API server (e.g., http://localhost:5000)
        test_case_dir: Path to test case directory containing color/ and depth/
        mesh_path: Path to mesh file (.stl, .ply, .obj)
        init_mask_path: Path to initial segmentation mask
        cam_K: Camera intrinsics as list [[fx,0,cx],[0,fy,cy],[0,0,1]]
        output_dir: Directory to save results (optional)
        visualize: Whether to request visualization images
    """
    
    print("="*60)
    print("WebAPI Pose Tracking Test")
    print("="*60)
    
    # Validate paths
    test_case_path = Path(test_case_dir)
    rgb_dir = test_case_path / "color"
    depth_dir = test_case_path / "depth"
    
    if not rgb_dir.exists() or not depth_dir.exists():
        print(f"ERROR: Test case directory must contain 'color' and 'depth' subdirectories")
        return False
    
    if not os.path.exists(mesh_path):
        print(f"ERROR: Mesh file not found: {mesh_path}")
        return False
    
    if not os.path.exists(init_mask_path):
        print(f"ERROR: Initial mask not found: {init_mask_path}")
        return False
    
    # Get frame lists
    rgb_files = get_sorted_frame_list(str(rgb_dir))
    depth_files = get_sorted_frame_list(str(depth_dir))
    
    if not rgb_files or not depth_files:
        print(f"ERROR: No frames found in test case directory")
        return False
    
    print(f"\nTest Configuration:")
    print(f"  API URL: {api_url}")
    print(f"  Test Case: {test_case_dir}")
    print(f"  Mesh: {mesh_path}")
    print(f"  Frames: {len(rgb_files)} RGB, {len(depth_files)} depth")
    print(f"  Camera K: {cam_K}")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output: {output_dir}")
    
    # Initialize client
    print(f"\n{'='*60}")
    print("Step 1: Connecting to API server...")
    print(f"{'='*60}")
    
    client = PoseTrackingClient(api_url=api_url)
    
    # Check server health
    try:
        health = client.health_check()
        print(f"✓ Server Status: {health['status']}")
        print(f"  Active Sessions: {health['active_sessions']}")
        print(f"  GPU Available: {health['gpu_available']}")
    except Exception as e:
        print(f"✗ Failed to connect to API server: {e}")
        print(f"  Make sure the server is running:")
        print(f"  python perception/webapi/pose_tracking_api.py --host 0.0.0.0 --port 5000")
        return False
    
    # Ensure cleanup happens even if test fails
    session_created = False
    
    # Create tracking session
    print(f"\n{'='*60}")
    print("Step 2: Creating tracking session...")
    print(f"{'='*60}")
    
    try:
        success = client.create_session(
            mesh_path=mesh_path,
            cam_K=np.array(cam_K),
            mesh_scale=0.01,  # Assume mesh is in mm, convert to m
            est_refine_iter=10,
            track_refine_iter=3,
            activate_2d_tracker=True,
            activate_kalman_filter=False  # Disabled for test case
        )
        
        if not success:
            print("✗ Failed to create session")
            return False
        
        session_created = True
        print(f"✓ Session created: {client.session_id}")
    
        # Load and initialize with first frame
        print(f"\n{'='*60}")
        print("Step 3: Initializing with first frame...")
        print(f"{'='*60}")
        
        rgb_0 = cv2.imread(str(rgb_dir / rgb_files[0]))
        rgb_0 = cv2.cvtColor(rgb_0, cv2.COLOR_BGR2RGB)
        depth_0 = cv2.imread(str(depth_dir / depth_files[0]), -1)
        mask_0 = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"  RGB shape: {rgb_0.shape}")
        print(f"  Depth shape: {depth_0.shape}, dtype: {depth_0.dtype}")
        print(f"  Mask shape: {mask_0.shape}, unique values: {np.unique(mask_0)}")
        
        initial_pose = client.initialize(
            rgb=rgb_0,
            depth=depth_0,
            mask=mask_0,
            depth_scale=1000.0,  # Depth in mm
            est_refine_iter=10
        )
        
        if initial_pose is None:
            print("✗ Failed to initialize tracking")
            return False
        
        print(f"✓ Initialized successfully")
        print(f"  Initial translation: {initial_pose['translation']}")
        print(f"  Initial quaternion: {initial_pose['quaternion']}")
        
        # Track remaining frames
        print(f"\n{'='*60}")
        print("Step 4: Tracking frames...")
        print(f"{'='*60}")
        
        num_frames = min(len(rgb_files), len(depth_files))
        poses = [initial_pose]
        
        for i in range(1, num_frames):
            # Load frame
            rgb = cv2.imread(str(rgb_dir / rgb_files[i]))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_dir / depth_files[i]), -1)
            
            # Track with all visualization types
            result = client.track(
                rgb=rgb,
                depth=depth,
                depth_scale=1000.0,
                visualize=visualize and output_dir is not None,
                viz_mode='all'  # Request all visualization types
            )
            if result is None or not result.get('success'):
                print(f"✗ Frame {i}: Tracking failed")
                break
            
            pose = result['pose']
            poses.append(pose)
            
            # Print progress with quality metrics
            if i % 10 == 0 or i == num_frames - 1:
                progress_str = f"  Frame {i:3d}/{num_frames}: t={pose['translation']}"
                print(progress_str)
            
            # Save all visualization types
            if visualize and output_dir and 'visualizations' in result:
                import base64
                from PIL import Image
                import io
                
                visualizations = result['visualizations']
                
                # Save pose visualization (3D bbox + axes)
                if 'pose' in visualizations:
                    vis_data = base64.b64decode(visualizations['pose'])
                    vis_img = Image.open(io.BytesIO(vis_data))
                    vis_img.save(os.path.join(output_dir, f'pose_vis_{i:04d}.png'))
                
                # Save bbox visualization (2D bounding box)
                if 'bbox' in visualizations:
                    vis_data = base64.b64decode(visualizations['bbox'])
                    vis_img = Image.open(io.BytesIO(vis_data))
                    vis_img.save(os.path.join(output_dir, f'bbox_vis_{i:04d}.png'))
                
                # Save mask visualization (segmentation overlay)
                if 'mask' in visualizations:
                    vis_data = base64.b64decode(visualizations['mask'])
                    vis_img = Image.open(io.BytesIO(vis_data))
                    vis_img.save(os.path.join(output_dir, f'mask_vis_{i:04d}.png'))
        
        print(f"✓ Tracked {len(poses)} frames successfully")
        
        # Get pose history from server
        print(f"\n{'='*60}")
        print("Step 5: Retrieving pose history...")
        print(f"{'='*60}")
        
        history = client.get_history()
        if history:
            print(f"✓ Retrieved history: {history['frame_count']} frames")
            
            # Save poses to file
            if output_dir:
                pose_matrices = np.array([p['matrix'] for p in history['poses']])
                output_path = os.path.join(output_dir, 'webapi_poses.npy')
                np.save(output_path, pose_matrices)
                print(f"✓ Saved poses to: {output_path}")
                
                # Save as text for easy inspection
                txt_path = os.path.join(output_dir, 'webapi_poses.txt')
                with open(txt_path, 'w') as f:
                    for i, p in enumerate(history['poses']):
                        f.write(f"Frame {i}:\n")
                        f.write(f"  Translation: {p['translation']}\n")
                        f.write(f"  Quaternion: {p['quaternion']}\n")
                        f.write("\n")
                print(f"✓ Saved readable poses to: {txt_path}")
                
                # Print visualization summary
                print(f"\n{'='*60}")
                print("Visualization Summary:")
                print(f"{'='*60}")
                print(f"  pose_vis_*.png  - 3D mesh + bounding box + axes overlay")
                print(f"  bbox_vis_*.png  - 2D bounding box overlay")
                print(f"  mask_vis_*.png  - Segmentation mask overlay")
                print(f"  Total frames: {len(poses)}")
        
        print(f"\n{'='*60}")
        print("Test completed successfully!")
        print(f"{'='*60}")
        
        return True
        
    finally:
        # Always close session to prevent resource leaks
        if session_created and client.session_id:
            print(f"\n{'='*60}")
            print("Cleaning up session...")
            print(f"{'='*60}")
            try:
                client.close_session()
                print("✓ Session closed")
            except Exception as e:
                print(f"⚠ Warning: Failed to close session: {e}")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test WebAPI-based pose tracking with test case data'
    )
    
    parser.add_argument(
        '--api_url',
        type=str,
        default='http://localhost:5000',
        help='URL of the pose tracking API server'
    )
    
    parser.add_argument(
        '--test_case',
        type=str,
        # default='/root/workspace/main/test_case/lego_20fps',
        default='/root/workspace/main/test_case/debug_frames',
        help='Path to test case directory (contains color/ and depth/ subdirs)'
    )
    
    parser.add_argument(
        '--mesh',
        type=str,
        # default='/root/workspace/main/test_case/lego_20fps/mesh/1x4.stl',
        default='/root/workspace/main/test_case/debug_frames/mesh/1x4.stl',
        help='Path to mesh file'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        # default='/root/workspace/main/test_case/lego_20fps/0_mask.png',
        default='/root/workspace/main/test_case/debug_frames/0_mask.png',
        help='Path to initial mask file'
    )
    
    parser.add_argument(
        '--cam_K',
        type=str,
        # default='[[426.8704833984375, 0.0, 423.89471435546875], [0.0, 426.4277648925781, 243.5056915283203], [0.0, 0.0, 1.0]]',
        default='[[639.5, 0.0, 639.5], [0.0, 638.0, 361.5], [0.0, 0.0, 1.0]]',
        help='Camera intrinsics as JSON string'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="/root/workspace/main/test_case/debug_frames/webapi_viz",
        help='Output directory for results and visualizations'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization (faster)'
    )
    
    args = parser.parse_args()
    
    # Parse camera intrinsics
    import json
    cam_K = json.loads(args.cam_K)
    
    # Run test
    success = test_webapi_tracking(
        api_url=args.api_url,
        test_case_dir=args.test_case,
        mesh_path=args.mesh,
        init_mask_path=args.mask,
        cam_K=cam_K,
        output_dir=args.output,
        visualize=not args.no_visualize
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
