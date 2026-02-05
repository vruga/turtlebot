#!/bin/bash
#
# Component Test Script
#
# Tests individual system components to verify proper configuration.
#
# Usage: ./test_components.sh [component]
#   Components:
#     camera     Test camera connection
#     model      Test TFLite model loading
#     esp32      Test ESP32 serial connection
#     inference  Run inference on sample image
#     all        Run all tests (default)
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
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default to all tests
COMPONENT=${1:-all}

echo -e "${BLUE}"
echo "============================================================"
echo "   Agricultural System - Component Tests"
echo "============================================================"
echo -e "${NC}"

# Test camera
test_camera() {
    echo -e "\n${BLUE}Testing Camera...${NC}"

    if [ ! -e /dev/video0 ]; then
        echo -e "${RED}FAIL: Camera device not found at /dev/video0${NC}"
        return 1
    fi

    python3 << 'EOF'
import cv2
import sys

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL: Cannot open camera")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("FAIL: Cannot read frame")
        sys.exit(1)

    print(f"SUCCESS: Camera working")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Camera test: PASSED${NC}"
    else
        echo -e "${RED}Camera test: FAILED${NC}"
        return 1
    fi
}

# Test model
test_model() {
    echo -e "\n${BLUE}Testing TFLite Model...${NC}"

    MODEL_PATH="${PROJECT_DIR}/models/plant_disease_model.tflite"

    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${YELLOW}SKIP: Model not found at $MODEL_PATH${NC}"
        return 0
    fi

    python3 << EOF
import sys
import numpy as np
import time

sys.path.insert(0, '${PROJECT_DIR}/src')

try:
    from inference.model_loader import ModelLoader

    loader = ModelLoader(model_path='${MODEL_PATH}')

    if not loader.is_loaded():
        print("FAIL: Model failed to load")
        sys.exit(1)

    # Test inference
    input_shape = loader.get_input_shape()
    dummy_input = np.random.random(input_shape).astype(np.float32)

    # Warm up
    loader.predict(dummy_input)

    # Measure time
    times = []
    for _ in range(10):
        start = time.time()
        output = loader.predict(dummy_input)
        times.append((time.time() - start) * 1000)

    avg_time = sum(times) / len(times)

    print(f"SUCCESS: Model loaded and working")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Avg inference time: {avg_time:.1f}ms")
    print(f"  Classes: {loader.get_num_classes()}")

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model test: PASSED${NC}"
    else
        echo -e "${RED}Model test: FAILED${NC}"
        return 1
    fi
}

# Test ESP32
test_esp32() {
    echo -e "\n${BLUE}Testing ESP32 Connection...${NC}"

    # Find serial port
    ESP32_PORT=""
    for port in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$port" ]; then
            ESP32_PORT="$port"
            break
        fi
    done

    if [ -z "$ESP32_PORT" ]; then
        echo -e "${YELLOW}SKIP: No serial port found${NC}"
        return 0
    fi

    python3 << EOF
import sys
import serial
import time

try:
    ser = serial.Serial('${ESP32_PORT}', 115200, timeout=5)
    time.sleep(2)  # Wait for ESP32 to initialize

    # Send ping
    ser.write(b'PING\n')
    ser.flush()

    # Wait for response
    response = ser.readline().decode().strip()

    if response in ['PONG', 'OK']:
        print(f"SUCCESS: ESP32 responding on ${ESP32_PORT}")
    else:
        print(f"WARNING: Unexpected response: {response}")

    ser.close()

except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}ESP32 test: PASSED${NC}"
    else
        echo -e "${RED}ESP32 test: FAILED${NC}"
        return 1
    fi
}

# Test inference pipeline
test_inference() {
    echo -e "\n${BLUE}Testing Inference Pipeline...${NC}"

    MODEL_PATH="${PROJECT_DIR}/models/plant_disease_model.tflite"

    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${YELLOW}SKIP: Model not found${NC}"
        return 0
    fi

    python3 << EOF
import sys
import numpy as np

sys.path.insert(0, '${PROJECT_DIR}/src')

try:
    from camera.image_preprocessor import ImagePreprocessor
    from inference.model_loader import ModelLoader
    from inference.disease_classifier import DiseaseClassifier

    # Create components
    preprocessor = ImagePreprocessor()
    loader = ModelLoader(model_path='${MODEL_PATH}')
    classifier = DiseaseClassifier()

    if not loader.is_loaded():
        print("FAIL: Model not loaded")
        sys.exit(1)

    # Create dummy image (224x224 RGB)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Process
    processed = preprocessor.preprocess(dummy_image)
    predictions = loader.predict(processed)
    result = classifier.classify(predictions)

    print("SUCCESS: Inference pipeline working")
    print(f"  Detected: {result.disease_name}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Severity: {result.severity}")
    print(f"  Spray: {result.should_spray} ({result.spray_duration}ms)")

except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Inference test: PASSED${NC}"
    else
        echo -e "${RED}Inference test: FAILED${NC}"
        return 1
    fi
}

# Run tests
case $COMPONENT in
    camera)
        test_camera
        ;;
    model)
        test_model
        ;;
    esp32)
        test_esp32
        ;;
    inference)
        test_inference
        ;;
    all)
        PASS=0
        FAIL=0

        test_camera && ((PASS++)) || ((FAIL++))
        test_model && ((PASS++)) || ((FAIL++))
        test_esp32 && ((PASS++)) || ((FAIL++))
        test_inference && ((PASS++)) || ((FAIL++))

        echo ""
        echo -e "${BLUE}============================================================${NC}"
        echo -e "Test Summary: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
        echo -e "${BLUE}============================================================${NC}"
        ;;
    *)
        echo "Unknown component: $COMPONENT"
        echo "Valid options: camera, model, esp32, inference, all"
        exit 1
        ;;
esac
