#!/bin/bash
# List USB cameras and show which device_id to use for the OpenCV camera node.

echo "USB video devices:"
echo "  device_id  →  /dev/videoN  (use this number for camera_node)"
echo ""

if ! ls /dev/video* 1>/dev/null 2>&1; then
    echo "No /dev/video* found. Plug in a USB camera."
    exit 1
fi

for dev in /dev/video*; do
    n="${dev#/dev/video}"
    name=""
    [ -r "/sys/class/video4linux/video$n/name" ] && name=$(cat "/sys/class/video4linux/video$n/name" 2>/dev/null)
    echo "  device_id $n  →  $dev  ${name:+($name)}"
done

echo ""
echo "Use device_id 0 for /dev/video0, or 2 for /dev/video2, etc."
echo "Full system with camera on /dev/video2:"
echo "  ros2 launch opencv_camera_node turtlebot_slam_camera.launch.py camera_device_id:=2"
echo "Camera-only test:"
echo "  ros2 run opencv_camera_node opencv_camera_node --ros-args -p device_id:=0"
