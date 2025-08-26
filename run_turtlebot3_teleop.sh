#!/bin/bash
# TurtleBot3 Teleop. Run from repo root or anywhere.

source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}

echo "=========================================="
echo "TurtleBot3 Teleop Keyboard Control"
echo "=========================================="
echo "Model: $TURTLEBOT3_MODEL"
echo "w/x = linear, a/d = angular, space/s = stop, Ctrl+C = quit"
echo "=========================================="
ros2 run turtlebot3_teleop teleop_keyboard
