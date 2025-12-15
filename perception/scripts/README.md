# Perception Pipeline Scripts

Complex scripts with command-line arguments that are part of the pipeline.

These scripts are production-ready and accept arguments for flexible usage.

## FoundationPose Initial Frame Processing Pipeline

FoundationPose requires processing only the **initial frame** to obtain the target object's 3D model. The pipeline consists of two stages:

### Stage 1: Object Segmentation (SAM3)
**Input**: RGB image + text prompt + bounding box  
**Output**: Segmentation mask image

Uses SAM3 to segment the target object from the initial frame by combining:
- Text description (e.g., "red mug")
- Bounding box to specify spatial region
- RGB image

### Stage 2: 3D Mesh Generation (SAM 3D Objects)
**Input**: RGB image + mask image  
**Output**: 3D mesh file (.stl)

Generates a real-world scaled 3D CAD model of the segmented object through a three-step process:

#### Step 2.1: Generate PLY from RGB + Mask (SAM3D)
**Script**: `gen_obj_mesh.py`  
**Input**: RGB image + mask image  
**Output**: Gaussian splat point cloud (.ply)

Uses SAM3D to generate a 3D Gaussian splat representation from the RGB image and mask.

```bash
python perception/scripts/gen_obj_mesh.py \
    --image image.png \
    --mask mask.png \
    --output object.ply
```

#### Step 2.2: Scale to Real-World Meters
**Script**: `process_obj_mesh.py`  
**Input**: PLY file + known dimension  
**Output**: Scaled mesh (.stl)

Scales the 3D model to real-world metric units (required by FoundationPose):
- Computes bounding box dimensions
- Applies scale factor based on known real-world dimension
- Converts point cloud to triangular mesh
- Exports as STL CAD file

```bash
# Edit process_obj_mesh.py to set:
# - file_path = "object.ply"
# - known_dimension_meters = 0.3  # Real height in meters
python perception/scripts/process_obj_mesh.py
```

#### Step 2.3: Output - CAD File (.stl)
The final `.stl` file contains a real-world scaled triangular mesh suitable for:
- FoundationPose 6D pose tracking
- CAD software import
- 3D printing
- Robot manipulation planning

**Complete Flow**:
```
(rgb_image, text, bbox) → [SAM3] → mask_image
(rgb_image, mask_image) → [SAM3D] → gaussian_splat.ply
(gaussian_splat.ply, real_dimension) → [Scale & Convert] → mesh.stl
```

This real-world scaled 3D model is then used by FoundationPose for accurate pose tracking across subsequent frames.

## Scripts

### `obj_mask.py`
Object segmentation with configurable parameters (SAM3 Stage 1)

### `gen_obj_mesh.py`
Generate PLY Gaussian splat from RGB image and mask using SAM3D (Stage 2.1). Takes a single RGB image with a segmentation mask and outputs a 3D point cloud representation.

### `process_obj_mesh.py`
Scale PLY to real-world meters and convert to STL CAD file (Stage 2.2). Applies metric scaling based on known dimensions and exports a triangular mesh suitable for FoundationPose.

### `draw_obj_mesh_dimension.py`
Visualize 3D model dimensions and coordinate axes from PLY or STL files. Generates a comprehensive plot showing the object from multiple viewpoints with axis directions, bounding box, and dimension measurements.

**Features:**
- Supports both Gaussian splat PLY files and mesh files (PLY/STL)
- Multi-view visualization: front, top, side, and isometric views
- Displays coordinate axes (X=red, Y=green, Z=blue) showing object orientation
- Shows bounding box with dimension measurements
- Generates both visual plot and detailed text report

**Usage:**
```bash
# Basic usage with PLY file
python draw_obj_mesh_dimension.py --input toy_bear.ply

# Specify custom output directory
python draw_obj_mesh_dimension.py --input toy_bear.ply --output my_visualizations

# Works with STL mesh files too
python draw_obj_mesh_dimension.py --input object.stl
```

**Output:**
- `dimensions_matplotlib.png` - Multi-view plot showing object geometry, axes, and dimensions
- `dimensions_report.txt` - Text report with detailed measurements
