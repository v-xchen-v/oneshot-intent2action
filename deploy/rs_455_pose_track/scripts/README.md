# RealSense 455 6D Pose Tracking Scripts

Scripts for 6D object pose tracking pipeline using RealSense cameras.

## Pipeline

```
Camera Intrinsics → Generate Mesh → Scale Mesh → Track Pose
```

1. **get_camera_intrinsics.py** - Extract camera K matrix
2. **gen_mesh.py** - Capture RGB-D and generate PLY mesh
3. **scale_mesh.py** - Scale to real-world dimensions and convert to STL
4. **track_pose.py** - Real-time 6D pose tracking with RealSense

---

## Prerequisites

```bash
pip install pyrealsense2 opencv-python open3d numpy requests
```

**Start API Server** (required for steps 2 & 3):
```bash
cd perception/webapi
./start_mesh_api.sh
```

---

## ⚠️ Important: Camera Setup for Depth Quality

**The quality of depth images is critical for accurate mesh generation and pose tracking.**

### Common Issues:
- **D455 too close**: RealSense D455 has minimum depth range (~280mm). Objects closer than this will have poor/no depth data
- **Poor lighting**: Infrared-based depth sensing needs adequate ambient light
- **Reflective surfaces**: Shiny objects may cause depth artifacts
- **Distance range**: Keep objects within camera's optimal range (0.3m - 3m for D455)

### Before Running Pipeline:
1. **Check depth image quality** - Verify depth data is clean and covers the object
2. **Adjust camera distance** - Move camera back if object is too close (>30cm recommended)
3. **Verify lighting** - Ensure good ambient lighting for better depth quality
4. **Test depth stream** - Use `realsense-viewer` to preview depth quality before capturing

**Tip**: Run `realsense-viewer` first to verify depth quality before using the tracking scripts.

---

## Usage

### 1. Get Camera Intrinsics

```bash
# Default resolution (640x480)
python get_camera_intrinsics.py

# Custom resolution
python get_camera_intrinsics.py --width 1280 --height 720 --fps 30
```

Output: `../camera_calibration/cam_K.txt`

### 2. Generate Mesh

```bash
# Interactive with text prompt
python gen_mesh.py --prompt "red mug" --output model.ply

# With remote API server
python gen_mesh.py --api-url http://10.150.240.101:5001 --prompt "toy bear" --output toy_bear.ply
```

Interaction: Press `SPACE` → Draw bbox → Press `ENTER`

### 3. Scale Mesh

```bash
# Interactive mode
python scale_mesh.py --mesh model.ply --output scaled.stl

# With remote API
python scale_mesh.py --mesh model.ply --output scaled.stl --api-url http://10.150.240.101:5001

# Non-interactive
python scale_mesh.py --mesh model.ply --output scaled.stl --no-viz --axis height --size 0.25
```

**Steps:**
1. View dimensions in terminal
2. Visualize mesh in VS Code (install 'Python PLY Preview' extension)
3. Measure object in real world (e.g., 0.25 meters tall)
4. Enter measurement when prompted

**Coordinate System:** X=Width (Red), Y=Height (Green), Z=Depth (Blue)

### 4. Track Object Pose

```bash
# Real-time tracking with interactive initialization
python track_pose.py --mesh scaled.stl --api-url http://localhost:5000

# With remote API server
python track_pose.py --mesh scaled.stl --api-url http://10.150.240.101:5000

# With existing mask and save video
python track_pose.py --mesh scaled.stl --mask init_mask.png --save-video output.mp4

# With camera intrinsics file
python track_pose.py --mesh scaled.stl --intrinsics ../camera_calibration/cam_K.txt
```

**Required:** Pose tracking API server must be running
```bash
cd perception/webapi
python pose_tracking_api.py --host 0.0.0.0 --port 5000
```

**Controls:**
- `SPACE` - Pause/Resume tracking
- `s` - Save current frame
- `ESC` - Quit

---

## Complete Example

```bash
# 1. Get camera intrinsics
python get_camera_intrinsics.py
# Output: ../camera_calibration/cam_K.txt

# 2. Start API server (separate terminal)
cd perception/webapi && ./start_mesh_api.sh

# 3. Generate mesh
python gen_mesh.py --prompt "toy bear" --output toy_bear.ply

# 4. Scale mesh
python scale_mesh.py --mesh toy_bear.ply --output toy_bear_scaled.stl
# Enter real-world measurement when prompted

# 5. Use scaled STL for pose tracking
```

---

## Troubleshooting

**Camera not found:** Check connection with `rs-enumerate-devices`

**API connection error:** Verify mesh API at `http://localhost:5001` or pose tracking API at `http://localhost:5000`

**Low mesh quality:** Use higher resolution or more specific text prompt

**Tracking lost:** Reinitialize by restarting the script or improve lighting conditions
