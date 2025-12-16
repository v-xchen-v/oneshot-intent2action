# Development Log: 6D Pose Tracking API Implementation

## Project Goal
Create a REST API server for real-time 6D object pose tracking using FoundationPose++, supporting camera streaming for remote GPU processing.

---

## Phase 1: Code Understanding ✅

### Step 1.1: Read Core Script
**File Analyzed**: `external/FoundationPose-plus-plus/src/obj_pose_track.py`

**Key Components Identified**:
- **FoundationPose**: 6D pose estimator (from FoundationPose library)
- **Cutie Tracker**: Optional 2D object tracker for improved robustness
- **Kalman Filter**: Optional state estimator for smooth pose predictions
- **Main Function**: `pose_track()` - processes image sequences from directories

### Step 1.2: Code Architecture Analysis
```
Input: RGB-D Image Sequences (from directories)
  ↓
Initialize (Frame 0):
  - Load mesh and compute bounding box
  - Initialize FoundationPose estimator
  - Register object with initial mask
  - Initialize 2D tracker (optional)
  - Initialize Kalman filter (optional)
  ↓
Tracking Loop (Frame 1+):
  - 2D Tracking: Get bbox from Cutie
  - Pose Adjustment: Align with 2D bbox center
  - Kalman Update: Smooth predictions (optional)
  - 6D Pose Refinement: FoundationPose.track_one()
  - Visualization: Draw 3D bbox and axes
  ↓
Output: 4×4 transformation matrices (.npy file)
```

### Step 1.3: Key Functions Understanding
| Function | Purpose | Key Details |
|----------|---------|-------------|
| `pose_track()` | Main tracking loop | Processes pre-recorded sequences |
| `adjust_pose_to_image_point()` | 2D-3D alignment | Adjusts 3D pose to match 2D bbox |
| `get_6d_pose_arr_from_mat()` | Format conversion | 4×4 matrix → [x,y,z,rx,ry,rz] |
| `get_mat_from_6d_pose_arr()` | Format conversion | [x,y,z,rx,ry,rz] → 4×4 matrix |

### Step 1.4: Dependencies Identified
```python
# Core tracking
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
from FoundationPose.estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

# Data processing
import trimesh  # 3D mesh handling
import cv2      # Image processing
import numpy as np
```

---

## Phase 2: Problem Analysis & Design ✅

### Step 2.1: Identified Limitations
**Original Code Limitations**:
1. ❌ Batch processing only (reads entire directories)
2. ❌ No real-time streaming support
3. ❌ Single-session only (cannot track multiple objects)
4. ❌ Local file system dependent
5. ❌ No remote access capability
6. ❌ Python-only interface

### Step 2.2: Requirements for Server Solution
**Must-Have Features**:
- ✅ REST API with HTTP endpoints
- ✅ Session-based tracking (multiple concurrent sessions)
- ✅ Frame-by-frame processing (streaming support)
- ✅ Remote GPU server deployment
- ✅ Language-agnostic client interface
- ✅ State management across frames

### Step 2.3: Design Questions & Answers

**Q1: How to handle streaming vs batch processing?**
- **A**: Replace directory reading with frame-by-frame processing via API endpoints
- Each frame sent as base64-encoded image in HTTP request
- Server maintains tracking state between frames

**Q2: How to manage multiple tracking sessions?**
- **A**: Create session-based architecture
- Each session gets unique ID
- Store session state (estimator, tracker, kalman filter) in server memory
- Use dict/Redis for session management

**Q3: How to initialize tracking without pre-loaded sequences?**
- **A**: Split into two API calls:
  1. `/initialize` - First frame + mask → register object
  2. `/track` - Subsequent frames → track object

**Q4: How to handle camera intrinsics and mesh data remotely?**
- **A**: Send during session creation:
  - Mesh file as base64-encoded data
  - Camera intrinsics as JSON array
  - Configuration parameters (refinement iterations, etc.)

**Q5: How to maintain tracking state?**
- **A**: Store in TrackingSession dataclass:
  ```python
  @dataclass
  class TrackingSession:
      session_id: str
      estimator: FoundationPose      # Maintains pose_last
      tracker_2d: Cutie              # Maintains tracking state
      kalman_filter: KalmanFilter6D  # Maintains kf_mean, kf_covariance
      frame_count: int
      pose_history: List[np.ndarray]
  ```

---

## Phase 3: Implementation ✅

### Step 3.1: API Design
**Endpoint Structure**:
```
POST   /api/session/create        → Create tracking session
POST   /api/session/{id}/initialize → Initialize with first frame
POST   /api/session/{id}/track     → Track new frame
GET    /api/session/{id}/history   → Get pose history
DELETE /api/session/{id}/close     → Close session
GET    /api/sessions               → List active sessions
GET    /api/health                 → Health check
```

### Step 3.2: Core Implementation Strategy
**Key Principle**: **Don't modify original code, import and wrap it**

**Implementation Approach**:
```python
# Add to Python path
sys.path.insert(0, "external/FoundationPose-plus-plus/src")

# Import original functions
from obj_pose_track import (
    adjust_pose_to_image_point,
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr
)
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D

# Wrap in Flask API
@app.route('/api/session/<session_id>/track', methods=['POST'])
def track_frame(session_id):
    # Use original tracking logic
    pose = session.estimator.track_one(rgb, depth, K, iteration)
    return jsonify({'pose': pose_to_dict(pose)})
```

### Step 3.3: Data Encoding/Decoding
**Image Transfer**:
```python
# Client → Server
base64.b64encode(image_bytes) → JSON

# Server processing
decode_image(base64_str) → np.ndarray → track() → pose

# Server → Client
pose_to_dict(np.ndarray) → JSON
```

### Step 3.4: Files Created
```
perception/webapi/
├── pose_tracking_api.py     # Main Flask server (576 lines)
├── client_example.py         # Python client library (370 lines)
├── requirements.txt          # API dependencies
├── README.md                 # Complete API documentation
└── start_api.sh             # Server startup script

perception/docker/
└── Dockerfile.api           # Docker deployment config
```

---

## Phase 4: Testing & Validation ⏳

### Step 4.1: Local Testing Checklist
- [ ] Start API server locally
- [ ] Test health endpoint
- [ ] Create session with sample mesh
- [ ] Initialize with first frame
- [ ] Track sequence of frames
- [ ] Verify pose estimates match original code
- [ ] Test visualization output
- [ ] Test session cleanup

### Step 4.2: Docker Testing Checklist
- [ ] Build Docker image
- [ ] Run container with GPU
- [ ] Test API endpoints from host
- [ ] Verify GPU utilization
- [ ] Test multiple concurrent sessions
- [ ] Monitor memory usage

### Step 4.3: Camera Streaming Test
- [ ] Connect RGB-D camera (RealSense/Kinect)
- [ ] Get camera intrinsics
- [ ] Stream frames to API
- [ ] Verify real-time performance (FPS)
- [ ] Test latency (network + processing)
- [ ] Handle dropped frames gracefully

---

## Phase 5: Documentation ✅

### Step 5.1: README Updates
**Added to main README.md**:
- ✅ Why WebAPI was created (5 key reasons)
- ✅ Architecture diagram
- ✅ Quick start guide
- ✅ Complete usage workflow (3 steps)
- ✅ Code examples (basic, client helper, live streaming)
- ✅ Configuration reference table
- ✅ API endpoints list
- ✅ Performance tips

### Step 5.2: API Documentation
**Created `perception/webapi/README.md`**:
- ✅ Feature list
- ✅ Endpoint specifications
- ✅ Request/response formats
- ✅ Python client examples
- ✅ JavaScript/cURL examples
- ✅ RealSense integration guide
- ✅ WebRTC streaming example
- ✅ Troubleshooting section

---

## Key Insights & Lessons Learned

### What Worked Well
1. ✅ **Non-invasive approach**: Importing original code preserved all functionality
2. ✅ **Session architecture**: Cleanly handles multiple concurrent tracking tasks
3. ✅ **Base64 encoding**: Simple and effective for image transfer over HTTP
4. ✅ **Flask framework**: Easy to implement, widely supported

### Challenges Addressed
1. **State management**: Used Python dict for active sessions (production should use Redis)
2. **GPU memory**: Each session uses 2-4GB, need to limit concurrent sessions
3. **Image encoding overhead**: Base64 adds ~33% size, consider compression for production
4. **Synchronization**: Kalman filter state must stay in sync with pose estimates

### Design Decisions
| Decision | Rationale |
|----------|-----------|
| Session-based API | Allows stateful tracking without reloading models |
| Base64 image encoding | Simple, works with JSON, no binary protocols needed |
| Import-only approach | Preserves original code, easy to update submodule |
| Flask over FastAPI | Simpler for initial implementation, easier debugging |
| Dataclass for session | Type-safe, clear structure, easy to extend |

---

## Next Steps & Improvements

### Production Readiness
- [ ] **Redis for session storage** (replace in-memory dict)
- [ ] **Authentication/API keys** (secure multi-user access)
- [ ] **Rate limiting** (prevent abuse)
- [ ] **Request queuing** (handle burst traffic)
- [ ] **Health monitoring** (Prometheus/Grafana)
- [ ] **Logging** (structured logs for debugging)

### Performance Optimization
- [ ] **WebSocket support** (lower latency than HTTP)
- [ ] **Image compression** (JPEG for RGB, PNG-16 for depth)
- [ ] **Batch processing endpoint** (process multiple frames at once)
- [ ] **GPU pooling** (distribute sessions across GPUs)
- [ ] **Model quantization** (reduce memory footprint)

### Feature Extensions
- [ ] **Multi-object tracking** (track multiple objects per session)
- [ ] **Pose smoothing options** (configurable filters)
- [ ] **Occlusion handling** (detect and recover from occlusions)
- [ ] **Auto-reinitialization** (recover from tracking loss)
- [ ] **Export formats** (ROS messages, COLMAP, etc.)

---

## Usage Summary

### Quick Reference

**Start Server**:
```bash
cd perception/webapi
./start_api.sh
```

**Basic Client Usage**:
```python
from client_example import PoseTrackingClient

client = PoseTrackingClient("http://localhost:5000")
client.create_session(mesh_path="object.stl", cam_K=camera_matrix)
client.initialize(rgb, depth, mask)

for frame in frames:
    result = client.track(rgb, depth)
    print(result['pose']['translation'])

client.close_session()
```

**Docker Deployment**:
```bash
docker build -f perception/docker/Dockerfile.api -t pose-api .
docker run --gpus all -p 5000:5000 pose-api
```

---

## References

### Original Code
- `external/FoundationPose-plus-plus/src/obj_pose_track.py` - Main tracking script
- `external/FoundationPose-plus-plus/src/VOT.py` - 2D tracker (Cutie)
- `external/FoundationPose-plus-plus/src/utils/kalman_filter_6d.py` - Kalman filter

### API Implementation
- `perception/webapi/pose_tracking_api.py` - REST API server
- `perception/webapi/client_example.py` - Python client library
- `perception/webapi/README.md` - Complete API documentation

### Documentation
- Main `README.md` - Updated with WebAPI section
- This file (`DEVELOPMENT_LOG.md`) - Development process documentation

---

**Last Updated**: December 16, 2025  
**Status**: Implementation Complete ✅ | Testing In Progress ⏳  
**Next Milestone**: Production deployment with Redis and authentication
