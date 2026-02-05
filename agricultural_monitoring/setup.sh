#!/bin/bash
#
# Agricultural Disease Detection System - One-Command Setup
#
# This script sets up the complete system on a Raspberry Pi 4B
# running ROS2 Humble.
#
# Usage: ./setup.sh
#
# Author: Agricultural Robotics Team

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}"
echo "============================================================"
echo "   Agricultural Disease Detection System - Setup"
echo "============================================================"
echo -e "${NC}"

# Check if running on Raspberry Pi
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo -e "Detected: ${GREEN}$MODEL${NC}"
else
    echo -e "${YELLOW}Not running on Raspberry Pi - some features may differ${NC}"
fi

# Check ROS2
echo -e "\n${BLUE}Checking ROS2...${NC}"
if [ -d "/opt/ros/humble" ]; then
    echo -e "${GREEN}ROS2 Humble found${NC}"
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}ROS2 Humble not found!${NC}"
    echo "Please install ROS2 Humble first:"
    echo "  https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

# Install system dependencies
echo -e "\n${BLUE}Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    v4l-utils \
    || true

# Create virtual environment (optional)
echo -e "\n${BLUE}Setting up Python environment...${NC}"
cd "$SCRIPT_DIR"

# Install Python dependencies
echo -e "\n${BLUE}Installing Python packages...${NC}"
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Install TFLite runtime for ARM
if [ "$(uname -m)" = "aarch64" ]; then
    echo -e "\n${BLUE}Installing TFLite runtime for ARM64...${NC}"
    pip3 install tflite-runtime
fi

# Create directories
echo -e "\n${BLUE}Creating directories...${NC}"
mkdir -p logs data/captures data/results cache models

# Set permissions for scripts
echo -e "\n${BLUE}Setting script permissions...${NC}"
chmod +x scripts/*.sh
chmod +x launch/*.py

# Add user to dialout group for serial access
echo -e "\n${BLUE}Configuring serial port access...${NC}"
sudo usermod -a -G dialout $USER 2>/dev/null || true
echo -e "${YELLOW}Note: You may need to log out and back in for serial access${NC}"

# Check for model file
echo -e "\n${BLUE}Checking for model file...${NC}"
if [ -f "models/plant_disease_model.tflite" ]; then
    echo -e "${GREEN}Model file found${NC}"
else
    echo -e "${YELLOW}No model file found${NC}"
    echo "Please either:"
    echo "  1. Copy your .tflite model to: models/plant_disease_model.tflite"
    echo "  2. Convert from .h5: python3 models/convert_to_tflite.py --model your_model.h5"
fi

# Environment setup hint
echo -e "\n${BLUE}Environment setup...${NC}"
echo "Add to your ~/.bashrc:"
echo "  source /opt/ros/humble/setup.bash"
echo "  export ANTHROPIC_API_KEY='your-api-key-here'"

# Summary
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Add your TFLite model to models/"
echo "  2. Upload ESP32 firmware from esp32/spray_controller/"
echo "  3. Set ANTHROPIC_API_KEY environment variable"
echo "  4. Run: ./scripts/test_components.sh"
echo "  5. Start: ./scripts/start_system.sh"
echo ""
echo -e "Dashboard will be at: ${BLUE}http://$(hostname -I | awk '{print $1}'):8080${NC}"
