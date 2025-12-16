#!/bin/bash
# Test script for the 3D Mesh Generation Web API
#
# This script tests the mesh generation API by:
# 1. Checking API health
# 2. Generating a 3D mesh from an image
# 3. Processing the mesh (scaling and converting to STL)
#
# Prerequisites:
# - API server must be running: ./perception/webapi/start_mesh_api.sh
# - conda environment sam3d-objects must be activated
# - Test image must be available
# Usage:
# # Basic test (health check + mesh generation)
# ./perception/tests/test_webapi_mesh.sh
#
# # Test with custom parameters
# ./perception/tests/test_webapi_mesh.sh --text "coffee mug" --format stl
#
# # Test with bounding box
# ./perception/tests/test_webapi_mesh.sh --bbox 100,150,200,300
#
# # Test mesh processing only (with existing PLY file)
# ./perception/tests/test_webapi_mesh.sh --test-process-mesh --ply-file output_mesh.ply
#
# # Test mesh processing with custom reference dimensions
# ./perception/tests/test_webapi_mesh.sh --test-process-mesh --ply-file test_output_mesh/output_mesh.ply --reference-size 0.3 --reference-axis height
#
# # See all options
# ./perception/tests/test_webapi_mesh.sh --help

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
API_URL="http://localhost:5001"
IMAGE_PATH="test_case/images/shutterstock_stylish_kidsroom_1640806567/image.png"
TEXT_PROMPT="toy bear"
OUTPUT_DIR="test_output_mesh"
BBOX=""
FORMAT="ply"
SKIP_HEALTH=false
TEST_PROCESS=false
PLY_FILE=""
REFERENCE_SIZE=0.5
REFERENCE_AXIS="width"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --text)
            TEXT_PROMPT="$2"
            shift 2
            ;;
        --bbox)
            BBOX="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-health)
            SKIP_HEALTH=true
            shift
            ;;
        --test-process-mesh)
            TEST_PROCESS=true
            shift
            ;;
        --ply-file)
            PLY_FILE="$2"
            shift 2
            ;;
        --reference-size)
            REFERENCE_SIZE="$2"
            shift 2
            ;;
        --reference-axis)
            REFERENCE_AXIS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Test the 3D Mesh Generation Web API"
            echo ""
            echo "Options:"
            echo "  --api-url URL           API server URL (default: http://localhost:5001)"
            echo "  --image PATH            Path to input image (default: test_case/images/.../image.png)"
            echo "  --text PROMPT           Text prompt for segmentation (default: 'toy bear')"
            echo "  --bbox X,Y,W,H          Bounding box (optional)"
            echo "  --format FORMAT         Output format: ply or stl (default: ply)"
            echo "  --output-dir DIR        Output directory (default: test_output_mesh)"
            echo "  --skip-health           Skip health check test"
            echo "  --test-process-mesh     Test only mesh processing (requires --ply-file)"
            echo "  --ply-file PATH         Path to PLY file for process_mesh test"
            echo "  --reference-size SIZE   Reference size in meters (default: 0.5)"
            echo "  --reference-axis AXIS   Reference axis: width/height/depth (default: width)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic test"
            echo "  $0"
            echo ""
            echo "  # Test with custom image and text"
            echo "  $0 --image path/to/image.jpg --text 'coffee mug'"
            echo ""
            echo "  # Test with bounding box"
            echo "  $0 --bbox 100,150,200,300"
            echo ""
            echo "  # Test mesh processing only (with existing PLY)"
            echo "  $0 --test-process-mesh --ply-file output_mesh.ply --reference-size 0.5"
            echo ""
            echo "  # Generate STL directly"
            echo "  $0 --format stl"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo -e "${BLUE} 3D Mesh Generation API Test${NC}"
echo "========================================================================"
echo "API URL:         $API_URL"
echo "Image:           $IMAGE_PATH"
echo "Text Prompt:     $TEXT_PROMPT"
echo "BBox:            ${BBOX:-none}"
echo "Output Format:   $FORMAT"
echo "Output Dir:      $OUTPUT_DIR"
if [ "$TEST_PROCESS" = true ]; then
    echo "Test Mode:       Process Mesh Only"
    echo "PLY File:        ${PLY_FILE:-none}"
    echo "Reference:       $REFERENCE_AXIS = $REFERENCE_SIZE meters"
else
    echo "Test Mode:       Mesh Generation"
fi
echo "========================================================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Build Python command
PYTHON_CMD="python perception/tests/test_webapi_mesh.py"
PYTHON_CMD="$PYTHON_CMD --api-url $API_URL"
PYTHON_CMD="$PYTHON_CMD --image $IMAGE_PATH"
PYTHON_CMD="$PYTHON_CMD --text \"$TEXT_PROMPT\""
PYTHON_CMD="$PYTHON_CMD --format $FORMAT"
PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"

if [ "$SKIP_HEALTH" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --skip-health"
fi

if [ -n "$BBOX" ]; then
    PYTHON_CMD="$PYTHON_CMD --bbox $BBOX"
fi

if [ "$TEST_PROCESS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --test-process-mesh"
    if [ -n "$PLY_FILE" ]; then
        PYTHON_CMD="$PYTHON_CMD --ply-file $PLY_FILE"
    fi
    PYTHON_CMD="$PYTHON_CMD --reference-size $REFERENCE_SIZE"
    PYTHON_CMD="$PYTHON_CMD --reference-axis $REFERENCE_AXIS"
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: No conda environment detected${NC}"
    echo -e "${YELLOW}   Consider activating sam3d-objects environment:${NC}"
    echo -e "${YELLOW}   conda activate sam3d-objects${NC}"
    echo ""
fi

# Run the Python test script
echo -e "${GREEN}Running tests...${NC}"
echo ""
echo "Command: $PYTHON_CMD"
echo ""

eval $PYTHON_CMD
TEST_EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests completed successfully!${NC}"
    echo "========================================================================"
    echo ""
    echo "Output files saved to: $OUTPUT_DIR"
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
    fi
else
    echo -e "${RED}❌ Tests failed with exit code: $TEST_EXIT_CODE${NC}"
    echo "========================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure the API server is running:"
    echo "   ./perception/webapi/start_mesh_api.sh"
    echo ""
    echo "2. Check the API is accessible:"
    echo "   curl $API_URL/health"
    echo ""
    echo "3. Verify the conda environment is activated:"
    echo "   conda activate sam3d-objects"
fi
echo "========================================================================"
echo ""

exit $TEST_EXIT_CODE
