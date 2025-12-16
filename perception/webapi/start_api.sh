#!/bin/bash
# Quick start script for Pose Tracking API

echo "=== Pose Tracking API Setup ==="

# Parse arguments
HOST="${1:-0.0.0.0}"
PORT="${2:-5000}"

echo ""
echo "=== Starting Pose Tracking API ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo ""
echo "API Endpoints:"
echo "  Health Check: http://$HOST:$PORT/api/health"
echo "  Create Session: POST http://$HOST:$PORT/api/session/create"
echo "  Initialize: POST http://$HOST:$PORT/api/session/{id}/initialize"
echo "  Track Frame: POST http://$HOST:$PORT/api/session/{id}/track"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the API server
cd "$PROJECT_ROOT"
python3 perception/webapi/pose_tracking_api.py --host "$HOST" --port "$PORT"
