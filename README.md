# oneshot-intent2action

This repository uses SAM 3D Objects to generate 3D mesh files from images for intent-to-action tasks.

## Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-compatible GPU (recommended for SAM 3D Objects)

### Installation

1. **Clone the repository with submodules**

```bash
git clone --recurse-submodules https://github.com/v-xchen-v/oneshot-intent2action.git
cd oneshot-intent2action
```

If you already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

2. **Set up Python environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If you need to install SAM 3D Objects dependencies separately:

```bash
cd external/sam-3d-objects
pip install -e .
cd ../..
```

### Purpose of External Libraries

This repository integrates external libraries as submodules:

**[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)**
- **Generate 3D mesh files (.stl format)** from 2D images
- Provide 3D object understanding for intent-to-action workflows
- Enable spatial reasoning and manipulation planning

The SAM 3D Objects model takes input images and produces high-quality 3D mesh representations in .stl file format, which is required by FoundationPose for 6D object pose estimation. These mesh files can be used for downstream robotic tasks, scene understanding, and action planning.

**[FoundationPose](https://github.com/teal024/FoundationPose-plus-plus)**
- **Track real-time 3D object pose** for dynamic scenes
- Provide 6D pose estimation for robotic manipulation
- Enable continuous pose tracking during object interaction

## Usage

### Using SAM 3D Objects to Generate Meshes

**Quick Setup (Minimal Dependencies)**

The SAM3D environment is easy to prepare with minimal dependencies:

```bash
# Create a Python 3.10+ environment
conda create -n sam3d python=3.10
conda activate sam3d

# Install SAM3D Objects
cd external/sam-3d-objects
pip install -e .

# Install minimal required dependencies
pip install transformers==5.0.0rc0
pip install torch torchvision
pip install matplotlib
```

**Using SAM3D Scripts**

After installation, you can use the SAM3D scripts to generate 3D mesh files:

```bash
conda activate sam3d
cd external/sam-3d-objects

# Use the provided scripts from the notebook directory
python notebook/inference.py --image /path/to/image.jpg --mask /path/to/mask.png --output model.ply
```

The generated mesh files can then be used in your intent-to-action pipeline for 3D object understanding and manipulation planning.

For detailed setup and usage, refer to the [official SAM 3D Objects setup guide](https://github.com/facebookresearch/sam-3d-objects).

### 3D Mesh Generation WebAPI

**Generate 3D meshes from RGB images using a REST API**

The Mesh Generation WebAPI combines SAM3 (for object segmentation) and SAM3D (for 3D reconstruction) to generate high-quality 3D mesh files from single RGB images.

**Key Features:**
- Generate 3D meshes (PLY/STL) from RGB images with text or bounding box prompts
- Scale meshes to real-world dimensions and convert formats
- REST API for easy integration with any language or platform
- Remote GPU processing for resource-intensive 3D reconstruction

**Quick Start**

1. **Activate the conda environment**:
```bash
conda activate sam3d-objects
```

2. **Install dependencies**:
```bash
pip install -r perception/webapi/requirements_mesh.txt
```

3. **Start the API server**:
```bash
# Using the start script (recommended)
./perception/webapi/start_mesh_api.sh

# Or with custom options
./perception/webapi/start_mesh_api.sh --port 5001 --debug
```

4. **Test the API**:
```bash
# Basic test
./perception/tests/test_webapi_mesh.sh

# Test mesh processing
./perception/tests/test_webapi_mesh.sh --test-process-mesh --ply-file output.ply
```

**API Endpoints:**
- `POST /api/generate_mesh` - Generate 3D mesh from image with text/bbox prompts
- `POST /api/process_mesh` - Scale PLY mesh to real dimensions and convert to STL
- `GET /health` - Check API server status

For detailed API documentation, see [perception/webapi/README.md](perception/webapi/README.md).

### 6D Pose Tracking with FoundationPose WebAPI

**Why We Created the WebAPI**

The FoundationPose++ tracking code was originally designed for batch processing of pre-recorded image sequences. However, real-world robotic applications require:

1. **Remote GPU Processing**: Run pose estimation on a powerful server while the robot/camera operates elsewhere
2. **Real-time Streaming**: Process live camera feeds frame-by-frame rather than loading entire sequences
3. **Multi-client Support**: Track multiple objects or serve multiple robots simultaneously
4. **Language Agnostic**: Enable clients in any language (Python, JavaScript, C++, ROS) to use pose tracking
5. **Stateful Sessions**: Maintain tracking state across frames without reloading models

The WebAPI wraps the original FoundationPose++ code in a REST API service, enabling these capabilities without modifying the core tracking algorithms.

**Architecture Overview**

```
Camera/Robot → HTTP/REST → WebAPI Server → FoundationPose++
                                        ├─ Cutie (2D Tracker)
                                        └─ Kalman Filter
                           ← Pose Data ←
```

**Quick Start: Running the API Server**

1. **Install dependencies**:
```bash
cd perception/webapi
pip install -r requirements.txt
```

2. **Start the server**:
```bash
# Using the start script
./start_api.sh

# Or directly with Python
python pose_tracking_api.py --host 0.0.0.0 --port 5000
```

3. **Check server health**:
```bash
curl http://localhost:5000/api/health
```

**Using Docker (Recommended for Production)**

```bash
# Start the FoundationPose Docker container
cd perception/docker/foundation_pose
./start.sh

# Enter the container
docker exec -it foundation_pose bash

# Verify environment setup by running test script
./perception/scripts/test_foundationpose.sh

# If test runs successfully, the FoundationPose environment is ready
# You can then use the API or run custom tracking scripts
```

**Note**: Running `./perception/scripts/test_foundationpose.sh` inside the container is the recommended way to verify that the FoundationPose environment is properly set up with all dependencies (including Cutie, Kalman filter, and model weights).

**Basic Usage Flow**

The API follows a three-step workflow:

**Step 1: Create a Tracking Session**

Create a session with your object's 3D mesh and camera parameters:

```python
import requests
import base64
import numpy as np

# Encode mesh file
with open('toy_bear_scaled.stl', 'rb') as f:
    mesh_b64 = base64.b64encode(f.read()).decode('utf-8')

# Camera intrinsics (replace with your camera's)
cam_K = [[912.7, 0.0, 667.6],
         [0.0, 911.0, 360.5],
         [0.0, 0.0, 1.0]]

# Create session
response = requests.post('http://localhost:5000/api/session/create', json={
    'mesh_file': mesh_b64,
    'mesh_scale': 1.0,  # Already scaled
    'cam_K': cam_K,
    'activate_2d_tracker': True,
    'activate_kalman_filter': True,
    'track_refine_iter': 5
})

session_id = response.json()['session_id']
print(f"Session created: {session_id}")
```

**Step 2: Initialize with First Frame**

Provide the first RGB-D frame with an initial segmentation mask:

```python
from PIL import Image
import io

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Initialize tracking
response = requests.post(
    f'http://localhost:5000/api/session/{session_id}/initialize',
    json={
        'rgb': encode_image('frame_0000.png'),
        'depth': encode_image('depth_0000.png'),
        'mask': encode_image('mask_0000.png'),
        'depth_scale': 1000.0  # Convert mm to meters
    }
)

initial_pose = response.json()['pose']
print(f"Initial position: {initial_pose['translation']}")
```

**Step 3: Track Streaming Frames**

Send subsequent frames to get real-time pose estimates:

```python
# Track each frame in sequence
for frame_id in range(1, 100):
    response = requests.post(
        f'http://localhost:5000/api/session/{session_id}/track',
        json={
            'rgb': encode_image(f'frame_{frame_id:04d}.png'),
            'depth': encode_image(f'depth_{frame_id:04d}.png'),
            'depth_scale': 1000.0,
            'visualize': True  # Get visualization image
        }
    )
    
    result = response.json()
    pose = result['pose']
    
    print(f"Frame {frame_id}:")
    print(f"  Translation: {pose['translation']}")
    print(f"  Quaternion: {pose['quaternion']}")
    
    # Optionally save visualization
    if 'visualization' in result:
        vis_data = base64.b64decode(result['visualization'])
        Image.open(io.BytesIO(vis_data)).save(f'vis_{frame_id:04d}.png')

# Close session when done
requests.delete(f'http://localhost:5000/api/session/{session_id}/close')
```

**Using the Python Client Helper**

For convenience, use the provided client class:

```python
from perception.webapi.client_example import PoseTrackingClient
import cv2

# Initialize client
client = PoseTrackingClient(api_url="http://localhost:5000")

# Create session
client.create_session(
    mesh_path="toy_bear_scaled.stl",
    cam_K=np.array(cam_K),
    activate_2d_tracker=True
)

# Initialize tracking
rgb = cv2.imread("frame_0000.png")
depth = cv2.imread("depth_0000.png", -1)
mask = cv2.imread("mask_0000.png", cv2.IMREAD_GRAYSCALE)
client.initialize(rgb, depth, mask)

# Track frames
for i in range(1, 100):
    rgb = cv2.imread(f"frame_{i:04d}.png")
    depth = cv2.imread(f"depth_{i:04d}.png", -1)
    result = client.track(rgb, depth, visualize=True)
    print(result['pose']['translation'])

client.close_session()
```

**Live Camera Streaming Example**

For real-time tracking with an RGB-D camera (e.g., RealSense):

```python
import pyrealsense2 as rs
import numpy as np

# Configure camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get camera intrinsics
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy],
                  [0, 0, 1]])

# Create tracking session
client = PoseTrackingClient(api_url="http://server-ip:5000")
client.create_session(mesh_path="object.stl", cam_K=cam_K)

# Initialize with first frame + mask
frames = pipeline.wait_for_frames()
rgb = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data())
mask = get_initial_mask(rgb)  # Use SAM or manual annotation
client.initialize(rgb, depth, mask)

# Stream and track
try:
    while True:
        frames = pipeline.wait_for_frames()
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        
        result = client.track(rgb, depth, depth_scale=1000.0)
        pose = result['pose']
        
        # Use pose for robot control, AR overlay, etc.
        print(f"Object at: {pose['translation']}")
        
finally:
    pipeline.stop()
    client.close_session()
```

**API Configuration Options**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_scale` | float | 0.01 | Scale factor for mesh (e.g., 0.01 for mm→m) |
| `est_refine_iter` | int | 10 | Initial pose refinement iterations |
| `track_refine_iter` | int | 5 | Per-frame tracking refinement (reduce for speed) |
| `activate_2d_tracker` | bool | false | Enable Cutie 2D tracker for robustness |
| `activate_kalman_filter` | bool | false | Enable Kalman filtering for smooth poses |
| `kf_measurement_noise_scale` | float | 0.05 | Kalman filter noise scale (higher = smoother) |

**API Endpoints**

- `GET /api/health` - Check server status
- `POST /api/session/create` - Create tracking session
- `POST /api/session/{id}/initialize` - Initialize with first frame
- `POST /api/session/{id}/track` - Track new frame
- `GET /api/session/{id}/history` - Get pose history
- `DELETE /api/session/{id}/close` - Close session
- `GET /api/sessions` - List active sessions

**Performance Tips**

- **Speed vs Accuracy**: Reduce `track_refine_iter` (3-5) for real-time performance
- **Memory**: Each session uses ~2-4GB GPU memory
- **Throughput**: Expect 10-30 FPS depending on refinement iterations
- **Network**: Use local network or compress images for remote servers

For complete API documentation, see [`perception/webapi/README.md`](perception/webapi/README.md).

## Project Structure

The repository is organized into three main categories for working with external libraries:

```
oneshot-intent2action/
├── perception/
│   ├── examples/        # Simple standalone playground scripts
│   │                    # - No command-line arguments
│   │                    # - Quick experimentation with libraries
│   │                    # - Self-contained test scripts
│   ├── scripts/         # Complex pipeline scripts with arguments
│   │                    # - Production-ready with argparse/CLI
│   │                    # - Part of the main pipeline
│   │                    # - Configurable via command-line
│   ├── modules/         # Reusable modules and utilities
│   │                    # - Shared code for scripts
│   │                    # - Library wrappers and helpers
│   ├── webapi/          # REST API for remote pose tracking
│   │                    # - Flask-based API server
│   │                    # - Client libraries and examples
│   │                    # - Docker deployment configs
│   ├── docker/          # Docker configurations
│   │                    # - Dockerfiles for different services
│   │                    # - Container build scripts
│   └── tests/           # Test scripts for validation
│                        # - Test individual scripts and modules
│                        # - Verify functionality with sample data
├── external/
│   ├── sam-3d-objects/           # SAM 3D Objects for 3D mesh generation
│   └── FoundationPose-plus-plus/ # 6D pose tracking (imported by webapi)
├── README.md
└── requirements.txt
```

**Folder Design Philosophy:**

- **`perception/examples/`**: Start here when exploring new libraries. Write simple, hardcoded scripts to understand how things work. No arguments needed - just run and see results.

- **`perception/scripts/`**: Once you understand the library, create production scripts here. These accept arguments, handle errors properly, and integrate into the pipeline.

- **`perception/modules/`**: Extract common functionality into reusable modules that both examples and scripts can import.

- **`perception/tests/`**: Write test scripts to validate your scripts and modules work correctly with sample data.

## Updating Submodules

To update the SAM 3D Objects submodule to the latest version:

```bash
git submodule update --remote external/sam-3d-objects
```

## Contributing

When working with this repository, remember to commit submodule updates if you change them:

```bash
git add external/sam-3d-objects
git commit -m "Update SAM 3D Objects submodule"
```

## License

Please refer to the individual licenses of dependencies, particularly SAM 3D Objects.
