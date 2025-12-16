#!/bin/bash
# Simple test script for WebAPI tracking using test case data

echo "=== WebAPI Tracking Test ==="
echo ""

# Configuration
API_URL="${1:-http://localhost:5000}"
TEST_CASE="${2:-/root/workspace/main/test_case/lego_20fps}"
OUTPUT_DIR="${3:-$TEST_CASE/webapi_results}"

echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Test Case: $TEST_CASE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check if API server is running
echo "Checking API server..."
if curl -s "$API_URL/api/health" > /dev/null 2>&1; then
    echo "✓ API server is running"
else
    echo "✗ API server is not responding at $API_URL"
    echo ""
    echo "Please start the API server first:"
    echo "  python perception/webapi/pose_tracking_api.py --host 0.0.0.0 --port 5000"
    echo ""
    exit 1
fi

echo ""
echo "Running test..."
echo ""

# Run the test
python perception/tests/test_webapi_tracking.py \
    --api_url "$API_URL" \
    --test_case "$TEST_CASE" \
    --mesh "$TEST_CASE/mesh/1x4.stl" \
    --mask "$TEST_CASE/0_mask.png" \
    --cam_K "[[426.8704833984375, 0.0, 423.89471435546875], [0.0, 426.4277648925781, 243.5056915283203], [0.0, 0.0, 1.0]]" \
    --output "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Test Completed Successfully ==="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - webapi_poses.npy (pose matrices)"
    echo "  - webapi_poses.txt (readable format)"
    echo "  - vis_*.png (visualizations)"
else
    echo ""
    echo "=== Test Failed ==="
    exit 1
fi
