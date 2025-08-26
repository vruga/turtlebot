#!/bin/bash
# Save the current SLAM map to files. No Nav2 required.
# Run WHILE the full system or SLAM is running. Run from repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
[ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash

MAP_NAME=${1:-my_map}
OUT_PATH="${HOME}/maps/${MAP_NAME}"
mkdir -p "$(dirname "$OUT_PATH")"

echo "Saving map to $OUT_PATH (need one /map message from SLAM)..."
python3 "$SCRIPT_DIR/save_map_from_topic.py" "$OUT_PATH"
echo "Files: ${OUT_PATH}.yaml and ${OUT_PATH}.pgm"
