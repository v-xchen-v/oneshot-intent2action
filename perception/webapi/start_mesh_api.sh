#!/bin/bash
# Start the 3D Mesh Generation API server

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Warning: No conda environment detected"
    echo "   Consider activating the sam3d-objects environment:"
    echo "   conda activate sam3d-objects"
    echo ""
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
HOST="0.0.0.0"
PORT="5001"
DEBUG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT    Port to bind to (default: 5001)"
            echo "  --debug        Enable debug mode"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --port 5001"
            echo "  $0 --host localhost --port 8080 --debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo " Starting 3D Mesh Generation API Server"
echo "========================================================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Debug: ${DEBUG:-disabled}"
echo ""
echo "The server combines SAM3 and SAM3D to generate 3D meshes from images."
echo "This may take a moment to load the models..."
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================================================"
echo ""

# Run the API server
python "$SCRIPT_DIR/mesh_generation_api.py" \
    --host "$HOST" \
    --port "$PORT" \
    $DEBUG
