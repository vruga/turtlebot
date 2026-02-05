#!/bin/bash
#
# Agricultural Dashboard - Standalone Startup Script
#
# Starts only the web dashboard for development/testing.
# Does not require ROS2 to be running (will work in standalone mode).
#
# Usage: ./start_dashboard.sh [options]
#   Options:
#     --port PORT    Set dashboard port (default: 8080)
#     --debug        Enable debug mode
#
# Author: Agricultural Robotics Team

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default options
PORT=8080
DEBUG_MODE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo -e "${BLUE}"
echo "============================================================"
echo "   Agricultural Dashboard (Standalone Mode)"
echo "============================================================"
echo -e "${NC}"

echo -e "Starting dashboard on port ${PORT}..."
echo ""
echo -e "${GREEN}Dashboard URL: http://${LOCAL_IP}:${PORT}${NC}"
echo -e "${GREEN}Local URL: http://localhost:${PORT}${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Source ROS2 if available (for full functionality)
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash 2>/dev/null || true
fi

# Run dashboard
export FLASK_APP="${PROJECT_DIR}/src/dashboard/app.py"
export FLASK_ENV=development

if [ -n "$DEBUG_MODE" ]; then
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
from dashboard.app import run_dashboard
run_dashboard(host='0.0.0.0', port=${PORT})
"
else
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
from dashboard.app import run_dashboard
run_dashboard(host='0.0.0.0', port=${PORT})
"
fi
