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
**Output**: 3D mesh file (.stl/.obj)

Generates a 3D model of the segmented object using:
- Original RGB image for texture/geometry
- Mask image to isolate the object
- Outputs STL/OBJ file for pose tracking

**Complete Flow**:
```
(rgb_image, text, bbox) → [SAM3] → mask_image
(rgb_image, mask_image) → [SAM3D] → mesh.stl
```

This 3D model is then used by FoundationPose for tracking the object across subsequent frames.

## Scripts

- `obj_mask.py` - Object segmentation with configurable parameters (SAM3 Stage 1)
- Add more pipeline scripts here...
