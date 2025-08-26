#!/bin/bash
# Full system: TurtleBot + SLAM (Cartographer) + OpenCV camera on Raspberry Pi (no RVIZ).
# Run from repo root: cd ~/turtlebot_slam_camera && ./start_full_system.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
[ -f "$HOME/turtlebot3_ws/install/setup.bash" ] && source "$HOME/turtlebot3_ws/install/setup.bash"

export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
export LDS_MODEL=${LDS_MODEL:-LDS-01}

usage() {
    echo "Full system: TurtleBot + SLAM + OpenCV camera (RPi, no RVIZ)"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo "  (no args)     Start full system: robot + SLAM + camera"
    echo "  --no-camera   Start robot + SLAM only (no camera node)"
    echo "  --camera-only Start OpenCV camera node only (/camera/image_raw)"
    echo "  --help        Show this help"
    echo ""
    echo "After starting full system:"
    echo "  Teleop (other terminal): source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL && ros2 run turtlebot3_teleop teleop_keyboard"
    echo "  Save map:               $SCRIPT_DIR/save_map.sh my_map  (run in another terminal while this is running)"
    echo ""
    echo "Topics: /scan  /map  /odom  /camera/image_raw  /camera/image_compressed"
    echo "Web server (after full system): $SCRIPT_DIR/start_server.sh  → http://<RPI_IP>:5000"
    echo "Camera device: $SCRIPT_DIR/check_camera.sh  then launch with camera_device_id:=N"
    echo "Build camera package once: cd ~/turtlebot3_ws && colcon build --packages-select opencv_camera_node && source install/setup.bash"
}

MODE=full
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --no-camera) MODE=no_camera ;;
        --camera-only) MODE=camera_only ;;
    esac
done

if [ "$MODE" = "camera_only" ]; then
    echo "Starting OpenCV camera node only (topics: /camera/image_raw, /camera/image_compressed)"
    ros2 run opencv_camera_node opencv_camera_node
    exit 0
fi

if [ "$MODE" = "no_camera" ]; then
    echo "Starting robot + SLAM (no camera)"
    if ls /dev/ttyACM* /dev/ttyUSB* 1>/dev/null 2>&1; then echo "Robot devices found."; else echo "Warning: No /dev/ttyACM* or /dev/ttyUSB*"; fi
    ros2 launch "$SCRIPT_DIR/turtlebot3_slam_combined.launch.py"
    exit 0
fi

echo "=========================================="
echo "Full system: TurtleBot + SLAM + OpenCV camera"
echo "=========================================="
echo "  Robot + LiDAR + Cartographer SLAM + Camera"
echo "  Topics: /scan  /map  /camera/image_raw"
echo ""
echo "  Teleop in another terminal:"
echo "    source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"
echo "    ros2 run turtlebot3_teleop teleop_keyboard"
echo "=========================================="
if ls /dev/ttyACM* /dev/ttyUSB* 1>/dev/null 2>&1; then echo "Robot devices found."; else echo "Warning: No /dev/ttyACM* or /dev/ttyUSB*"; fi

ros2 launch opencv_camera_node turtlebot_slam_camera.launch.py
