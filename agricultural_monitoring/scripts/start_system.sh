#!/bin/bash
#
# Agricultural Disease Detection System - Startup Script
#
# This script launches the complete agricultural monitoring system:
# - Verifies dependencies
# - Checks hardware connections
# - Launches ROS2 nodes
# - Starts web dashboard
#
# Usage: ./start_system.sh [options]
#   Options:
#     --no-dashboard    Don't start web dashboard
#     --no-llm          Don't start LLM client
#     --debug           Enable debug logging
#
# Author: Agricultural Robotics Team

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default options
ENABLE_DASHBOARD=true
ENABLE_LLM=true
DEBUG_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-dashboard)
            ENABLE_DASHBOARD=false
            shift
            ;;
        --no-llm)
            ENABLE_LLM=false
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "============================================================"
echo "   Agricultural Disease Detection System"
echo "   Version 1.0.0"
echo "============================================================"
echo -e "${NC}"

# Function to check if command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not installed"
        return 1
    fi
}

# Function to check serial port
check_serial_port() {
    if [ -e "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $1 available"
        return 0
    else
        echo -e "  ${YELLOW}!${NC} $1 not found"
        return 1
    fi
}

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

DEPS_OK=true

echo "Commands:"
check_command python3 || DEPS_OK=false
check_command ros2 || DEPS_OK=false

echo ""
echo "Python packages:"
check_python_package rclpy || DEPS_OK=false
check_python_package cv2 || DEPS_OK=false
check_python_package flask || DEPS_OK=false
check_python_package serial || DEPS_OK=false
check_python_package yaml || DEPS_OK=false
check_python_package numpy || DEPS_OK=false

# Check TFLite runtime
if python3 -c "import tflite_runtime" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} tflite_runtime"
elif python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "  ${YELLOW}!${NC} tensorflow (tflite_runtime preferred for Pi)"
else
    echo -e "  ${RED}✗${NC} No TFLite runtime found"
    DEPS_OK=false
fi

if [ "$ENABLE_LLM" = true ]; then
    check_python_package anthropic || echo -e "  ${YELLOW}!${NC} LLM disabled without anthropic package"
fi

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo -e "${RED}Missing dependencies. Please install them first:${NC}"
    echo "  pip install -r ${PROJECT_DIR}/requirements.txt"
    exit 1
fi

echo ""

# Check hardware
echo -e "${BLUE}Checking hardware...${NC}"

# Check camera
if [ -e /dev/video0 ]; then
    echo -e "  ${GREEN}✓${NC} Camera (/dev/video0)"
else
    echo -e "  ${YELLOW}!${NC} Camera not found at /dev/video0"
fi

# Check ESP32
ESP32_PORT="/dev/ttyUSB0"
if [ -e "$ESP32_PORT" ]; then
    echo -e "  ${GREEN}✓${NC} ESP32 ($ESP32_PORT)"
else
    # Try alternative ports
    for port in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$port" ]; then
            ESP32_PORT="$port"
            echo -e "  ${GREEN}✓${NC} ESP32 found at $ESP32_PORT"
            break
        fi
    done
    if [ ! -e "$ESP32_PORT" ]; then
        echo -e "  ${YELLOW}!${NC} ESP32 not found (spray control disabled)"
    fi
fi

echo ""

# Check model file
echo -e "${BLUE}Checking model...${NC}"
MODEL_PATH="${PROJECT_DIR}/models/plant_disease_model.tflite"
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Model found ($MODEL_SIZE)"
else
    echo -e "  ${RED}✗${NC} Model not found at $MODEL_PATH"
    echo ""
    echo "Please either:"
    echo "  1. Copy your model to: $MODEL_PATH"
    echo "  2. Convert from .h5: python3 ${PROJECT_DIR}/models/convert_to_tflite.py --model your_model.h5"
    exit 1
fi

echo ""

# Check environment variables
echo -e "${BLUE}Checking environment...${NC}"

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "  ${GREEN}✓${NC} ANTHROPIC_API_KEY set"
else
    echo -e "  ${YELLOW}!${NC} ANTHROPIC_API_KEY not set (LLM recommendations will use fallback)"
fi

if [ -n "$ROS_DISTRO" ]; then
    echo -e "  ${GREEN}✓${NC} ROS_DISTRO: $ROS_DISTRO"
else
    echo -e "  ${YELLOW}!${NC} ROS2 environment not sourced"
    echo "  Attempting to source ROS2 Humble..."

    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "  ${GREEN}✓${NC} ROS2 Humble sourced"
    else
        echo -e "  ${RED}✗${NC} ROS2 Humble not found"
        exit 1
    fi
fi

echo ""

# Create necessary directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/data/captures"
mkdir -p "${PROJECT_DIR}/data/results"
mkdir -p "${PROJECT_DIR}/cache"
mkdir -p /tmp/agricultural_captures
echo -e "  ${GREEN}✓${NC} Directories ready"

echo ""

# Get local IP for dashboard URL
LOCAL_IP=$(hostname -I | awk '{print $1}')
DASHBOARD_PORT=8080

# Launch message
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Starting Agricultural Disease Detection System${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Components:"
echo "  - Frame Capture Node (normal priority)"
echo "  - Inference Worker (nice +15)"
echo "  - ESP32 Controller (normal priority)"

if [ "$ENABLE_LLM" = true ]; then
    echo "  - Claude LLM Client (nice +10)"
fi

if [ "$ENABLE_DASHBOARD" = true ]; then
    echo "  - Web Dashboard"
    echo ""
    echo -e "${BLUE}Dashboard URL: http://${LOCAL_IP}:${DASHBOARD_PORT}${NC}"
fi

echo ""
echo "Press Ctrl+C to stop all components"
echo ""

# Set up cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    # Kill all child processes
    pkill -P $$ 2>/dev/null || true
    echo -e "${GREEN}Shutdown complete${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Launch using ROS2 launch file
if [ "$DEBUG_MODE" = true ]; then
    export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG
fi

ros2 launch "${PROJECT_DIR}/launch/agricultural_system.launch.py" \
    enable_dashboard:=$ENABLE_DASHBOARD \
    enable_llm:=$ENABLE_LLM \
    dashboard_port:=$DASHBOARD_PORT

# Wait for all background processes
wait
