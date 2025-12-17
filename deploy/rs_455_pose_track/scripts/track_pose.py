#!/usr/bin/env python3
"""
Real-time 6D Object Pose Tracking using RealSense Camera

This script performs real-time 6D pose tracking by:
1. Connecting to RealSense camera
2. Creating a tracking session with mesh and camera intrinsics
3. Capturing initial frame and mask (interactive bbox or existing mask)
4. Tracking object pose in real-time
5. Visualizing pose with 3D axis overlay

Usage:
    # Interactive initialization with bbox
    python track_pose.py --mesh ../mesh/black_mug/scaled.stl \
        --api-url http://10.150.240.101:5000 \
        --mask \mask.png \
        --width 640 --height 480 --fps 30 \
        --target-hz 5 \
        --save-frames ./debug_frames
    
    
    # With existing mask
    --mask initial_mask.png
    
    # With specific API URL
    --api-url http://localhost:5000
    
    # With custom camera resolution
    --width 640 --height 480 --fps 30
    
    # With camera intrinsics file
    --intrinsics ../camera_calibration/cam_K.txt
    
    # With target tracking frequency
    --target-hz 5
    
    # With save debugging input frames
    --save-frames ./debug_frames
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import sys
import os
import json
import time
import base64
import io
from pathlib import Path
from typing import Optional, Tuple
from collections import deque
from PIL import Image

# Import the official client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../perception/webapi'))
from client_example import PoseTrackingClient
class RealSenseCamera:
    """RealSense camera manager"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.intrinsics = None
    
    def start(self):
        """Start camera streaming"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Wait for camera to stabilize
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        print(f"✓ Camera started: {self.width}x{self.height} @ {self.fps}fps")
    
    def stop(self):
        """Stop camera streaming"""
        if self.pipeline:
            self.pipeline.stop()
    
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get aligned RGB and depth frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        # Convert to numpy arrays
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        return rgb, depth
    
    def get_intrinsics_matrix(self) -> np.ndarray:
        """Get camera intrinsics as 3x3 matrix"""
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        return K


def load_intrinsics_from_file(filepath: str) -> np.ndarray:
    """Load camera intrinsics from file"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.txt':
        # Plain text format
        K = np.loadtxt(filepath)
    elif filepath.suffix == '.npy':
        # NumPy binary format
        K = np.load(filepath)
    elif filepath.suffix == '.json':
        # JSON format
        with open(filepath, 'r') as f:
            data = json.load(f)
            K = np.array(data.get('K', data.get('intrinsics', data.get('camera_matrix'))))
    else:
        raise ValueError(f"Unsupported intrinsics file format: {filepath.suffix}")
    
    return K


def interactive_bbox_selection(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Interactive bounding box selection"""
    print("\n" + "="*70)
    print("INTERACTIVE BBOX SELECTION")
    print("="*70)
    print("Instructions:")
    print("  1. Draw a bounding box around the object")
    print("  2. Press ENTER to confirm")
    print("  3. Press 'r' to redraw")
    print("  4. Press ESC to cancel")
    print("="*70)
    
    bbox = cv2.selectROI("Select Object", rgb, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    
    if bbox[2] == 0 or bbox[3] == 0:
        print("✗ No bbox selected")
        return None
    
    x, y, w, h = bbox
    print(f"✓ BBox selected: x={x}, y={y}, w={w}, h={h}")
    
    return np.array([x, y, w, h])


def bbox_to_mask(bbox: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert bbox to binary mask"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x, y, w, h = bbox.astype(int)
    mask[y:y+h, x:x+w] = 255
    return mask


def draw_pose_overlay(image: np.ndarray, pose_dict: dict, cam_K: np.ndarray, 
                      axis_length: float = 0.1) -> np.ndarray:
    """Draw 3D axis on image using pose"""
    overlay = image.copy()
    
    # Get pose matrix
    pose_matrix = np.array(pose_dict['matrix'])

    # Define axis points in object frame
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([axis_length, 0, 0, 1])
    y_axis = np.array([0, axis_length, 0, 1])
    z_axis = np.array([0, 0, axis_length, 1])
    
    # Transform to camera frame
    points_3d = np.array([origin, x_axis, y_axis, z_axis])
    points_cam = (pose_matrix @ points_3d.T).T
    
    # Project to image
    points_2d = []
    for pt in points_cam[:, :3]:
        px = int(cam_K[0, 0] * pt[0] / pt[2] + cam_K[0, 2])
        py = int(cam_K[1, 1] * pt[1] / pt[2] + cam_K[1, 2])
        points_2d.append((px, py))
    
    # Draw axes
    origin_pt = points_2d[0]
    cv2.line(overlay, origin_pt, points_2d[1], (0, 0, 255), 3)  # X - Red
    cv2.line(overlay, origin_pt, points_2d[2], (0, 255, 0), 3)  # Y - Green
    cv2.line(overlay, origin_pt, points_2d[3], (255, 0, 0), 3)  # Z - Blue
    
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Real-time 6D pose tracking with RealSense camera",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to scaled mesh file (.stl, .ply, .obj)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5000",
        help="Pose tracking API URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--mask",
        help="Path to initial mask image (if not provided, use interactive bbox)"
    )
    parser.add_argument(
        "--intrinsics",
        help="Path to camera intrinsics file (cam_K.txt/npy/json)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)"
    )
    parser.add_argument(
        "--save-video",
        help="Path to save output video"
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Mesh scale factor (default: 1.0 for meters, use 0.001 for mm)"
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="Target tracking frequency in Hz (e.g., 5 for 5 Hz). If not set, runs at maximum speed."
    )
    parser.add_argument(
        "--save-frames",
        help="Directory to save RGB and depth frames for debugging (e.g., ./debug_frames)"
    )
    
    args = parser.parse_args()
    
    # Verify mesh file exists
    if not os.path.exists(args.mesh):
        print(f"❌ Error: Mesh file not found: {args.mesh}")
        return 1
    
    # Create save frames directory if specified
    save_frames_dir = None
    if args.save_frames:
        save_frames_dir = Path(args.save_frames)
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        (save_frames_dir / "color").mkdir(exist_ok=True)
        (save_frames_dir / "depth").mkdir(exist_ok=True)
        print(f"📁 Saving frames to: {save_frames_dir}")
    
    print("="*70)
    print("REAL-TIME 6D POSE TRACKING")
    print("="*70)
    print(f"Mesh: {args.mesh}")
    print(f"API: {args.api_url}")
    print(f"Camera: {args.width}x{args.height} @ {args.fps}fps")
    if args.target_hz:
        print(f"Target Tracking: {args.target_hz} Hz")
    else:
        print(f"Target Tracking: Maximum speed")
    print("="*70)
    
    # Initialize camera
    print("\n📷 Initializing RealSense camera...")
    camera = RealSenseCamera(args.width, args.height, args.fps)
    
    try:
        camera.start()
    except Exception as e:
        print(f"❌ Failed to start camera: {e}")
        return 1
    
    # # Get camera intrinsics
    # if args.intrinsics:
    #     print(f"\n📐 Loading camera intrinsics from: {args.intrinsics}")
    #     cam_K = load_intrinsics_from_file(args.intrinsics)
    # else:
    print(f"\n📐 Using camera intrinsics from RealSense")
    cam_K = camera.get_intrinsics_matrix()
    
    print(f"Camera K matrix:\n{cam_K}")
    
    # Initialize API client
    print(f"\n🌐 Connecting to tracking API: {args.api_url}")
    client = PoseTrackingClient(api_url=args.api_url)
    
    try:
        health = client.health_check()
        print(f"✓ API Status: {health.get('status', 'OK')}")
        print(f"  GPU Available: {health.get('gpu_available', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print("  Make sure the server is running:")
        print("  cd perception/webapi && ./start_api.sh")
        camera.stop()
        return 1
    
    # Create tracking session
    print(f"\n🔧 Creating tracking session...")
    try:
        success = client.create_session(
            mesh_path=args.mesh,
            cam_K=cam_K,
            mesh_scale=1.0,
            est_refine_iter=10,
            track_refine_iter=3,
            activate_2d_tracker=True,
            activate_kalman_filter=False,
            kf_measurement_noise_scale=0.05
        )
        if not success:
            print(f"❌ Failed to create session (see error above)")
            camera.stop()
            return 1
        print(f"✓ Session created: {client.session_id}")
    except Exception as e:
        print(f"❌ Exception during session creation: {e}")
        camera.stop()
        return 1
    
    # Capture initial frame
    print(f"\n📸 Capturing initial frame...")
    print("Position the object in view and press ENTER...")
    
    while True:
        rgb, depth = camera.get_frames()
        if rgb is None:
            continue
        
        cv2.imshow("Initial Frame - Press ENTER when ready", rgb)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            print("✗ Cancelled")
            camera.stop()
            cv2.destroyAllWindows()
            return 1
    
    cv2.destroyAllWindows()
    
    # Get initial mask
    if args.mask and os.path.exists(args.mask):
        print(f"Loading mask from: {args.mask}")
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    else: 
        # No mask provided, exit with error
        print(f"❌ No initial mask provided. Please provide --mask argument with valid mask image.")
        camera.stop()
        return 1
    # else:
    #     bbox = interactive_bbox_selection(rgb)
    #     if bbox is None:
    #         camera.stop()
    #         return 1
    #     mask = bbox_to_mask(bbox, rgb.shape)
    
    # Initialize tracking
    print(f"\n🎯 Initializing tracking...")
    try:
        pose = client.initialize(rgb, depth, mask, depth_scale=1000.0)
        if pose:
            print(f"✓ Tracking initialized")
            trans = pose['translation']
            print(f"  Initial Pose: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
        else:
            raise Exception("Initialize returned None")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        camera.stop()
        return 1
    
    # Setup video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, args.fps, 
                                       (args.width, args.height))
        print(f"📹 Recording to: {args.save_video}")
    
    # Real-time tracking loop
    print(f"\n🚀 Starting real-time tracking...")
    print("="*70)
    print("Controls:")
    print("  ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  's' - Save current frame")
    print("  'd' - Toggle debug info")
    print("="*70)
    
    frame_count = 0
    paused = False
    show_debug = True
    
    # Performance monitoring
    fps_buffer = deque(maxlen=30)  # Track last 30 frames
    last_time = time.time()
    track_times = deque(maxlen=30)
    capture_times = deque(maxlen=30)
    
    # Frame rate limiting
    target_frame_interval = 1.0 / args.target_hz if args.target_hz else 0
    last_track_time = time.time()
    
    if args.save_frames:
        # save mask to the directory
        mask_path = save_frames_dir / "0_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"✓ Saved initial mask to: {mask_path}")
    
    try:
        while True:
            if not paused:
                loop_start = time.time()
                
                # Capture frame
                capture_start = time.time()
                rgb, depth = camera.get_frames()
                capture_time = time.time() - capture_start
                capture_times.append(capture_time * 1000)  # ms
                
                if rgb is None:
                    continue
                
                # Frame rate limiting - skip tracking if target Hz is set and interval not met
                current_time = time.time()
                if args.target_hz and (current_time - last_track_time) < target_frame_interval:
                    # Just display the last frame without tracking
                    cv2.waitKey(1)
                    continue
                
                last_track_time = current_time
                
                # Track pose (disable visualize for better performance)
                try:
                    track_start = time.time()
                    return_vis = True
                    result = client.track(rgb, depth, depth_scale=1000.0, visualize=return_vis)
                    track_time = time.time() - track_start
                    track_times.append(track_time * 1000)  # ms
                    
                    
                    if return_vis:
                        # "visualizations": {
                        #     "pose": "base64_image",  # 3D mesh + bbox + axes overlay
                        #     "bbox": "base64_image",  # 2D bounding box overlay
                        #     "mask": "base64_image"   # Segmentation mask overlay
                        # } (if visualize=true)
                        pose_vis_b64 = result.get('visualizations', {}).get('pose')
                        if pose_vis_b64:
                            vis_bytes = base64.b64decode(pose_vis_b64)
                            vis_image = np.array(Image.open(io.BytesIO(vis_bytes)))
                            

                    
                    if not result:
                        raise Exception("Track returned None")
                    
                    # Get pose and draw overlay
                    pose = result.get('pose')
                    if not pose:
                        raise Exception("No pose in result")
                    
                    # vis_image = draw_pose_overlay(rgb, pose, cam_K)
                    
                    # Calculate actual FPS
                    current_time = time.time()
                    frame_time = current_time - last_time
                    last_time = current_time
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_buffer.append(fps)
                    avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
                    
                    # Add info overlay
                    trans = pose['translation']
                    y_offset = 30
                    line_height = 30
                    
                    if show_debug:
                        # FPS info
                        cv2.putText(vis_image, f"FPS: {avg_fps:.1f} (Current: {fps:.1f})", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += line_height
                        
                        # Timing breakdown
                        avg_track = sum(track_times) / len(track_times) if track_times else 0
                        avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
                        cv2.putText(vis_image, f"Track: {avg_track:.1f}ms | Capture: {avg_capture:.1f}ms", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        y_offset += line_height
                        
                        # Pose
                        cv2.putText(vis_image, f"Pose: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        y_offset += line_height
                        
                        # Frame count
                        cv2.putText(vis_image, f"Frame: {frame_count}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Minimal overlay
                        cv2.putText(vis_image, f"FPS: {avg_fps:.1f}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show
                    cv2.imshow("6D Pose Tracking", vis_image)
                    
                    # Save to video
                    if video_writer:
                        video_writer.write(vis_image)
                    
                    # Save frames for debugging
                    if save_frames_dir:
                        rgb_path = save_frames_dir / "color" / f"{frame_count}.png"
                        depth_path = save_frames_dir / "depth" / f"{frame_count}.png"
                        cv2.imwrite(str(rgb_path), rgb)
                        # Save depth as 16-bit PNG
                        cv2.imwrite(str(depth_path), depth)
                    
                    frame_count += 1
                    
                    # Print periodic stats
                    if frame_count % 100 == 0:
                        print(f"📊 Frame {frame_count}: Avg FPS={avg_fps:.1f}, "
                              f"Track={avg_track:.1f}ms, Capture={avg_capture:.1f}ms")
                    
                except Exception as e:
                    print(f"⚠️  Tracking error: {e}")
                    cv2.imshow("6D Pose Tracking", rgb)
            else:
                cv2.waitKey(100)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"{'⏸' if paused else '▶'}  {status}")
            elif key == ord('s'):
                filename = f"frame_{frame_count:06d}.png"
                cv2.imwrite(filename, vis_image)
                print(f"💾 Saved: {filename}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"🔧 Debug info: {'ON' if show_debug else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        print(f"\n🛑 Stopping tracking...")
        camera.stop()
        cv2.destroyAllWindows()
        
        if video_writer:
            video_writer.release()
            print(f"✓ Video saved: {args.save_video}")
        
        # Final statistics
        avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
        avg_track = sum(track_times) / len(track_times) if track_times else 0
        avg_capture = sum(capture_times) / len(capture_times) if capture_times else 0
        
        print(f"✓ Tracked {frame_count} frames")
        print(f"📊 Average FPS: {avg_fps:.2f}")
        print(f"📊 Average tracking time: {avg_track:.1f}ms")
        print(f"📊 Average capture time: {avg_capture:.1f}ms")
        if save_frames_dir:
            print(f"📁 Frames saved to: {save_frames_dir}")
            print(f"   RGB images: {save_frames_dir / 'color'}")
            print(f"   Depth images: {save_frames_dir / 'depth'}")
        print("="*70)
        print("Done!")
        print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
