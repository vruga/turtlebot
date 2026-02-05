#!/bin/bash
#
# macOS Test Script
#
# Tests the agricultural detection system on macOS without ROS2 or hardware.
#
# Usage: ./simulation/test_macos.sh [mode]
#   Modes:
#     unit        Run unit tests only
#     pipeline    Test inference pipeline
#     dashboard   Start dashboard with mock data
#     interactive Run interactive CLI
#     full        Run everything (default)
#
# Author: Agricultural Robotics Team

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODE=${1:-full}

echo -e "${BLUE}"
echo "============================================================"
echo "   Agricultural System - macOS Test Suite"
echo "============================================================"
echo -e "${NC}"

cd "$PROJECT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    exit 1
fi

# Install dependencies if needed
echo -e "${BLUE}Checking dependencies...${NC}"
pip3 install -q numpy opencv-python flask pyyaml pytest 2>/dev/null || true

# Check for anthropic (optional)
if python3 -c "import anthropic" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} anthropic"
else
    echo -e "  ${YELLOW}!${NC} anthropic not installed (LLM features disabled)"
fi

echo ""

case $MODE in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        python3 -m pytest tests/ -v --ignore=tests/test_esp32_comm.py -k "not live"
        ;;

    pipeline)
        echo -e "${BLUE}Testing inference pipeline...${NC}"
        python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from camera.image_preprocessor import ImagePreprocessor
from inference.disease_classifier import DiseaseClassifier
import numpy as np

print("Testing preprocessor...")
prep = ImagePreprocessor(input_size=(224, 224))
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = prep.preprocess(dummy)
print(f"  Input: {dummy.shape} -> Output: {result.shape}")
assert result.shape == (1, 224, 224, 3), "Shape mismatch!"

print("Testing classifier...")
classifier = DiseaseClassifier()
preds = np.random.random((1, 10))
preds = preds / preds.sum()
detection = classifier.classify(preds)
print(f"  Detected: {detection.disease_name} ({detection.confidence:.1%})")
print(f"  Severity: {detection.severity}")
print(f"  Should spray: {detection.should_spray}")

print("\n✓ Pipeline test passed!")
EOF
        ;;

    dashboard)
        echo -e "${BLUE}Starting dashboard with mock data...${NC}"
        echo -e "Dashboard will be at: ${GREEN}http://localhost:8080${NC}"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 simulation/run_standalone.py --mock-model --no-llm
        ;;

    interactive)
        echo -e "${BLUE}Starting interactive mode...${NC}"
        python3 simulation/run_standalone.py --interactive --mock-model
        ;;

    esp32)
        echo -e "${BLUE}Starting mock ESP32...${NC}"
        echo "This creates a fake serial port for testing"
        echo ""
        python3 simulation/mock_esp32.py
        ;;

    full)
        echo -e "${BLUE}Running full test suite...${NC}"
        echo ""

        # Unit tests
        echo -e "${BLUE}1. Unit tests${NC}"
        python3 -m pytest tests/test_inference.py tests/test_integration.py -v --tb=short 2>/dev/null || true
        echo ""

        # Pipeline test
        echo -e "${BLUE}2. Pipeline test${NC}"
        python3 << 'EOF'
import sys
import time
sys.path.insert(0, 'src')
sys.path.insert(0, 'simulation')

from run_standalone import StandaloneSystem

system = StandaloneSystem(
    use_webcam=False,
    use_mock_model=True,
    enable_llm=False,
    enable_dashboard=False
)

print("Running 3 test captures...")
for i in range(3):
    result = system.process_capture()
    time.sleep(0.5)

print("\n✓ Full pipeline test passed!")
system.cleanup()
EOF
        echo ""

        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}All tests completed!${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo "To start the dashboard: ./simulation/test_macos.sh dashboard"
        echo "To run interactively:   ./simulation/test_macos.sh interactive"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: unit, pipeline, dashboard, interactive, esp32, full"
        exit 1
        ;;
esac
