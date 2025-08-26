#!/bin/bash
# Start the SLAM + camera web server. Run after start_full_system.sh.
# Run from repo root: cd ~/turtlebot_slam_camera && ./start_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash

if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing flask (once): pip3 install flask"
    pip3 install flask
fi

echo "Starting SLAM + camera server at http://0.0.0.0:5000"
echo "On this Pi: http://localhost:5000   From another device: http://<RPI_IP>:5000"
python3 "$SCRIPT_DIR/slam_camera_server.py"
