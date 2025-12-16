#!/usr/bin/env python3
"""
Script to get RealSense camera intrinsic matrix (K matrix) and save it for FoundationPose.

This script allows you to:
1. List all available camera resolutions and frame rates
2. Check the current active resolution
3. Capture camera intrinsics at a specific resolution
4. Save intrinsic matrix in multiple formats (txt, npy, json)

Usage Examples:
    # List all available resolutions supported by the camera
    python get_camera_intrinsics.py --list-resolutions

    # Check current active resolution
    python get_camera_intrinsics.py --current-resolution

    # Get intrinsics at default resolution (640x480 @ 30fps)
    python get_camera_intrinsics.py

    # Get intrinsics at specific resolution
    python get_camera_intrinsics.py --width 1280 --height 720 --fps 30

    # Save only txt format (for FoundationPose)
    python get_camera_intrinsics.py --format txt --output-dir ./config

    # High resolution with custom output directory
    python get_camera_intrinsics.py --width 1920 --height 1080 --fps 30 --output-dir ./camera_calib

Output Files:
    - cam_K.txt: 3x3 intrinsic matrix in plain text (FoundationPose compatible)
    - cam_K.npy: NumPy binary format for easy loading in Python
    - camera_info.json: Detailed camera parameters including distortion coefficients

Intrinsic Matrix Format:
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    
    Where:
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point (optical center) coordinates
"""
import pyrealsense2 as rs
import numpy as np
import json
import argparse
from pathlib import Path


def list_available_resolutions():
    """
    List all available resolutions and formats for the RealSense camera.
    
    Returns:
        List of tuples: (width, height, fps, format)
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        raise RuntimeError("No RealSense device connected")
    
    device = devices[0]
    print(f"\nDevice: {device.get_info(rs.camera_info.name)}")
    print(f"Serial Number: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
    
    resolutions = []
    
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        print(f"\n{sensor_name}:")
        
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color:
                video_profile = profile.as_video_stream_profile()
                width = video_profile.width()
                height = video_profile.height()
                fps = video_profile.fps()
                fmt = video_profile.format()
                
                resolutions.append((width, height, fps, fmt))
                print(f"  {width}x{height} @ {fps}fps ({fmt})")
    
    return resolutions


def get_current_resolution():
    """
    Get the current active resolution from RealSense camera.
    
    Returns:
        Tuple: (width, height, fps, format)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Start with default settings
    profile = pipeline.start(config)
    
    try:
        color_profile = profile.get_stream(rs.stream.color)
        video_profile = color_profile.as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        
        width = intrinsics.width
        height = intrinsics.height
        fps = video_profile.fps()
        fmt = video_profile.format()
        
        print(f"\nCurrent Active Resolution:")
        print(f"  {width}x{height} @ {fps}fps ({fmt})")
        
        return width, height, fps, fmt
        
    finally:
        pipeline.stop()


def get_camera_intrinsics(width=640, height=480, fps=30):
    """
    Get camera intrinsic matrix from RealSense camera.
    
    Args:
        width: Image width
        height: Image height
        fps: Frame rate
        
    Returns:
        K: 3x3 intrinsic matrix
        intrinsics: RealSense intrinsics object
    """
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    try:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    except Exception as e:
        print(f"Warning: Could not set requested resolution {width}x{height} @ {fps}fps")
        print(f"Error: {e}")
        print("\nTrying with default settings...")
        config = rs.config()  # Reset config
    
    # Start pipeline
    print(f"\nStarting RealSense pipeline with resolution {width}x{height} @ {fps}fps...")
    profile = pipeline.start(config)
    
    try:
        # Get stream profile and camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Build intrinsic matrix K
        # K = [[fx,  0, cx],
        #      [ 0, fy, cy],
        #      [ 0,  0,  1]]
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        
        print("\nCamera Intrinsics:")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  Principal Point: ({intrinsics.ppx:.2f}, {intrinsics.ppy:.2f})")
        print(f"  Focal Length: ({intrinsics.fx:.2f}, {intrinsics.fy:.2f})")
        print(f"  Distortion Model: {intrinsics.model}")
        print(f"  Distortion Coefficients: {intrinsics.coeffs}")
        
        print("\nIntrinsic Matrix K:")
        print(K)
        
        return K, intrinsics
        
    finally:
        # Stop pipeline
        pipeline.stop()
        print("\nPipeline stopped.")


def save_intrinsics(K, intrinsics, output_dir, format='txt'):
    """
    Save camera intrinsics to file(s).
    
    Args:
        K: 3x3 intrinsic matrix
        intrinsics: RealSense intrinsics object
        output_dir: Output directory path
        format: Output format ('txt', 'npy', 'json', or 'all')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formats_to_save = ['txt', 'npy', 'json'] if format == 'all' else [format]
    
    for fmt in formats_to_save:
        if fmt == 'txt':
            # Save as plain text (FoundationPose compatible format)
            txt_path = output_dir / 'cam_K.txt'
            np.savetxt(txt_path, K, fmt='%.6f')
            print(f"\nSaved intrinsic matrix to: {txt_path}")
            
        elif fmt == 'npy':
            # Save as numpy binary
            npy_path = output_dir / 'cam_K.npy'
            np.save(npy_path, K)
            print(f"Saved intrinsic matrix to: {npy_path}")
            
        elif fmt == 'json':
            # Save detailed info as JSON
            json_path = output_dir / 'camera_info.json'
            camera_info = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'model': str(intrinsics.model),
                'coeffs': intrinsics.coeffs,
                'K_matrix': K.tolist()
            }
            with open(json_path, 'w') as f:
                json.dump(camera_info, f, indent=2)
            print(f"Saved camera info to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Get RealSense camera intrinsic matrix (K matrix) for FoundationPose'
    )
    parser.add_argument(
        '--list-resolutions',
        action='store_true',
        help='List all available resolutions and exit'
    )
    parser.add_argument(
        '--current-resolution',
        action='store_true',
        help='Show current active resolution and exit'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=640,
        help='Image width (default: 640)'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=480,
        help='Image height (default: 480)'
    )
    parser.add_argument(
        '--fps', 
        type=int, 
        default=30,
        help='Frame rate (default: 30)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./camera_calibration',
        help='Output directory for calibration files (default: ./camera_calibration)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'npy', 'json', 'all'],
        default='all',
        help='Output format: txt (plain text), npy (numpy), json (detailed info), or all (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        # Handle list resolutions command
        if args.list_resolutions:
            list_available_resolutions()
            return 0
        
        # Handle current resolution command
        if args.current_resolution:
            get_current_resolution()
            return 0
        
        # Get camera intrinsics
        K, intrinsics = get_camera_intrinsics(args.width, args.height, args.fps)
        
        # Save to file(s)
        save_intrinsics(K, intrinsics, args.output_dir, args.format)
        
        print("\n✓ Successfully captured and saved camera intrinsics!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
