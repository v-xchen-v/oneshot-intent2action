#!/bin/bash
#
# track_pipe.sh - Complete 6D Pose Tracking Pipeline
#
# This script runs the complete pipeline:
#   1. Generate mesh from RealSense camera (gen_mesh.py)
#   2. Scale mesh to real-world dimensions (scale_mesh.py)
#   3. Track object pose in real-time (track_pose.py)
#
# Usage:
#   ./track_pipe.sh [OPTIONS]
#
# Options:
#   --prompt TEXT          Object description (e.g., "red mug", "toy bear")
#   --mesh-api URL         Mesh generation API URL (default: http://localhost:5001)
#   --track-api URL        Pose tracking API URL (default: http://localhost:5000)
#   --width WIDTH          Camera width (default: 1280)
#   --height HEIGHT        Camera height (default: 720)
#   --fps FPS              Camera FPS (default: 30)
#   --target-hz HZ         Tracking frequency (default: 5)
#   --intrinsics FILE      Camera intrinsics file (default: ../camera_calibration/cam_K.txt)
#   --output-dir DIR       Output directory for all files (default: ./pipeline_output)
#   --save-frames DIR      Save debug frames (optional)
#   --mask-id NUM          Which mask to use if multiple detected (default: 0)
#   --skip-mesh-gen        Skip mesh generation (use existing mesh in output-dir)
#   --skip-mesh-scale      Skip mesh scaling (use existing scaled mesh in output-dir)
#   --help                 Show this help message
#
# Examples:
#   # Basic usage with prompt
#   ./track_pipe.sh --prompt "toy bear"
#   # Full example with all options
#   ./track_pipe.sh --prompt "toy bear" \
#       --mesh-api http://10.150.240.101:5001 \
#       --track-api http://10.150.240.101:5000 \
#       --save-frames ./debug_frames \
#       --width 640 --height 480 --target-hz 5 \
#       --mask-id 0 \
#       --skip-mesh-gen \
#       --skip-mesh-scale
#       --skip-mesh-scale
#
#   # With custom API servers
#   ./track_pipe.sh --prompt "red mug" --mesh-api http://10.150.240.101:5001 --track-api http://10.150.240.101:5000
#
#   # With debug frame saving
#   ./track_pipe.sh --prompt "bottle" --save-frames ./debug_frames
#
#   # Custom resolution and tracking frequency
#   ./track_pipe.sh --prompt "cup" --width 640 --height 480 --target-hz 10
#
#   # Skip mesh generation, use existing mesh
#   ./track_pipe.sh --skip-mesh-gen --output-dir ./pipeline_output
#
#   # Skip both mesh generation and scaling, only track
#   ./track_pipe.sh --skip-mesh-gen --skip-mesh-scale --output-dir ./pipeline_output
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
PROMPT=""
MESH_API_URL="http://localhost:5001"
TRACK_API_URL="http://localhost:5000"
WIDTH=1280
HEIGHT=720
FPS=30
TARGET_HZ=5
INTRINSICS="../camera_calibration/cam_K.txt"
OUTPUT_DIR="./pipeline_output"
SAVE_FRAMES=""
MASK_ID=0
SKIP_MESH_GEN=false
SKIP_MESH_SCALE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --mesh-api)
            MESH_API_URL="$2"
            shift 2
            ;;
        --track-api)
            TRACK_API_URL="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --target-hz)
            TARGET_HZ="$2"
            shift 2
            ;;
        --intrinsics)
            INTRINSICS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save-frames)
            SAVE_FRAMES="$2"
            shift 2
            ;;
        --mask-id)
            MASK_ID="$2"
            shift 2
            ;;
        --skip-mesh-gen)
            SKIP_MESH_GEN=true
            shift
            ;;
        --skip-mesh-scale)
            SKIP_MESH_SCALE=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${CYAN}"
echo "========================================================================"
echo "  6D Pose Tracking Pipeline"
echo "========================================================================"
echo -e "${NC}"

# Validate required parameters
if [ -z "$PROMPT" ]; then
    echo -e "${YELLOW}⚠️  No --prompt specified. Interactive bbox selection will be used.${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${BLUE}📁 Output directory: $OUTPUT_DIR${NC}\n"

# Define file paths
RAW_MESH="$OUTPUT_DIR/raw_mesh.ply"
INITIAL_MASK="$OUTPUT_DIR/initial_mask.png"
SCALED_MESH="$OUTPUT_DIR/scaled.stl"

# ============================================================================
# STEP 1: Generate Mesh
# ============================================================================
if [ "$SKIP_MESH_GEN" = true ]; then
    echo -e "${YELLOW}========================================================================"
    echo "STEP 1/3: Generate Mesh - SKIPPED"
    echo -e "========================================================================${NC}"
    echo -e "${YELLOW}Using existing files:${NC}"
    
    if [ ! -f "$RAW_MESH" ]; then
        echo -e "${RED}❌ Error: Raw mesh not found: $RAW_MESH${NC}"
        echo -e "${YELLOW}   Run without --skip-mesh-gen to generate mesh${NC}"
        exit 1
    fi
    
    if [ ! -f "$INITIAL_MASK" ]; then
        echo -e "${RED}❌ Error: Initial mask not found: $INITIAL_MASK${NC}"
        echo -e "${YELLOW}   Run without --skip-mesh-gen to generate mask${NC}"
        exit 1
    fi
    
    echo -e "  ${GREEN}✓${NC} Mesh: $RAW_MESH"
    echo -e "  ${GREEN}✓${NC} Mask: $INITIAL_MASK\n"
else
    echo -e "${GREEN}========================================================================"
    echo "STEP 1/3: Generate Mesh from RealSense Camera"
    echo -e "========================================================================${NC}"

    GEN_MESH_CMD="python gen_mesh.py --output \"$RAW_MESH\" --save-mask \"$INITIAL_MASK\" --api-url \"$MESH_API_URL\" --width $WIDTH --height $HEIGHT --fps $FPS --mask-id $MASK_ID"

    if [ -n "$PROMPT" ]; then
        GEN_MESH_CMD="$GEN_MESH_CMD --prompt \"$PROMPT\""
    fi

    echo -e "${CYAN}Running: $GEN_MESH_CMD${NC}\n"
    eval $GEN_MESH_CMD

    if [ ! -f "$RAW_MESH" ]; then
        echo -e "${RED}❌ Error: Mesh generation failed. File not found: $RAW_MESH${NC}"
        exit 1
    fi

    if [ ! -f "$INITIAL_MASK" ]; then
        echo -e "${RED}❌ Error: Mask generation failed. File not found: $INITIAL_MASK${NC}"
        exit 1
    fi

    echo -e "\n${GREEN}✓ Step 1 Complete: Mesh and mask generated${NC}"
    echo -e "  Mesh: $RAW_MESH"
    echo -e "  Mask: $INITIAL_MASK\n"
fi

# ============================================================================
# STEP 2: Scale Mesh
# ============================================================================
if [ "$SKIP_MESH_SCALE" = true ]; then
    echo -e "${YELLOW}========================================================================"
    echo "STEP 2/3: Scale Mesh - SKIPPED"
    echo -e "========================================================================${NC}"
    echo -e "${YELLOW}Using existing scaled mesh:${NC}"
    
    if [ ! -f "$SCALED_MESH" ]; then
        echo -e "${RED}❌ Error: Scaled mesh not found: $SCALED_MESH${NC}"
        echo -e "${YELLOW}   Run without --skip-mesh-scale to scale mesh${NC}"
        exit 1
    fi
    
    echo -e "  ${GREEN}✓${NC} Scaled mesh: $SCALED_MESH\n"
else
    echo -e "${GREEN}========================================================================"
    echo "STEP 2/3: Scale Mesh to Real-World Dimensions"
    echo -e "========================================================================${NC}"

    SCALE_MESH_CMD="python scale_mesh.py --mesh \"$RAW_MESH\" --output \"$SCALED_MESH\" --api-url \"$MESH_API_URL\""

    echo -e "${CYAN}Running: $SCALE_MESH_CMD${NC}\n"
    eval $SCALE_MESH_CMD

    if [ ! -f "$SCALED_MESH" ]; then
        echo -e "${RED}❌ Error: Mesh scaling failed. File not found: $SCALED_MESH${NC}"
        exit 1
    fi

    echo -e "\n${GREEN}✓ Step 2 Complete: Mesh scaled${NC}"
    echo -e "  Scaled mesh: $SCALED_MESH\n"
fi

# ============================================================================
# STEP 3: Track Pose
# ============================================================================
echo -e "${GREEN}========================================================================"
echo "STEP 3/3: Real-Time 6D Pose Tracking"
echo -e "========================================================================${NC}"

TRACK_POSE_CMD="python track_pose.py --mesh \"$SCALED_MESH\" --api-url \"$TRACK_API_URL\" --mask \"$INITIAL_MASK\" --intrinsics \"$INTRINSICS\" --width $WIDTH --height $HEIGHT --fps $FPS --target-hz $TARGET_HZ"

if [ -n "$SAVE_FRAMES" ]; then
    TRACK_POSE_CMD="$TRACK_POSE_CMD --save-frames \"$SAVE_FRAMES\""
fi

echo -e "${CYAN}Running: $TRACK_POSE_CMD${NC}\n"
eval $TRACK_POSE_CMD

# ============================================================================
# Pipeline Complete
# ============================================================================
echo -e "\n${GREEN}"
echo "========================================================================"
echo "  Pipeline Complete!"
echo "========================================================================"
echo -e "${NC}"
echo -e "Generated files:"
echo -e "  ${BLUE}Raw mesh:${NC}    $RAW_MESH"
echo -e "  ${BLUE}Mask:${NC}        $INITIAL_MASK"
echo -e "  ${BLUE}Scaled mesh:${NC} $SCALED_MESH"
if [ -n "$SAVE_FRAMES" ]; then
    echo -e "  ${BLUE}Debug frames:${NC} $SAVE_FRAMES"
fi
echo ""
echo -e "${YELLOW}💡 Tip: You can restart tracking with the same mesh:${NC}"
echo -e "   python track_pose.py --mesh \"$SCALED_MESH\" --mask \"$INITIAL_MASK\" --intrinsics \"$INTRINSICS\" --api-url \"$TRACK_API_URL\""
echo ""
