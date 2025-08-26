#!/bin/bash
# Quick check: is /map being published? (needed for map to show on the web server)

source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash

echo "Checking if /map is published (SLAM must be running)..."
TOPICS=$(ros2 topic list 2>/dev/null) || true
if echo "$TOPICS" | grep -q '^/map$'; then
    echo "  OK: /map topic exists"
    echo "  One message (first lines):"
    timeout 3 ros2 topic echo /map --once 2>/dev/null | head -15 || echo "  (timeout or no message)"
else
    echo "  NOT FOUND: /map is not published."
    echo ""
    echo "  You need start_full_system.sh running (robot + SLAM + camera)."
    echo "  In one terminal:  cd ~/turtlebot_slam_camera && ./start_full_system.sh"
    echo "  Then in another: teleop to drive, and ./start_server.sh for the web page."
fi
