#!/usr/bin/env python3
"""
Save the current /map topic to a .yaml and .pgm file.
Does NOT require nav2_map_server (Navigation2). Uses only ROS2 + the /map topic.
Run while SLAM is running (start_full_system.sh or DO_SLAM.sh).
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import sys
import os


def main():
    rclpy.init()
    node = Node("save_map_once")
    map_msg = [None]

    def cb(msg):
        map_msg[0] = msg

    sub = node.create_subscription(OccupancyGrid, "map", cb, 10)
    node.get_logger().info("Waiting for one /map message (drive robot if map is empty)...")
    while rclpy.ok() and map_msg[0] is None:
        rclpy.spin_once(node, timeout_sec=0.5)

    if map_msg[0] is None:
        node.get_logger().error("No /map received. Is SLAM running? Run start_full_system.sh first.")
        rclpy.shutdown()
        sys.exit(1)

    msg = map_msg[0]
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/maps/my_map")
    out_dir = os.path.dirname(out_path)
    base = os.path.basename(out_path)
    if not base:
        base = "my_map"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."
    base = base.replace(".yaml", "").replace(".pgm", "")
    yaml_path = os.path.join(out_dir, base + ".yaml")
    pgm_path = os.path.join(out_dir, base + ".pgm")

    # Write PGM (occupancy: 0 free, 100 occupied, 205 unknown in standard map format)
    data = np.array(msg.data, dtype=np.int32).reshape((msg.info.height, msg.info.width))
    # -1 -> 205 (unknown), 0 -> 0 (free), 100 -> 100 (occupied)
    pgm_data = np.where(data == -1, 205, np.clip(data, 0, 100).astype(np.uint8))
    with open(pgm_path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (msg.info.width, msg.info.height))
        f.write(pgm_data.tobytes())

    # Write YAML
    with open(yaml_path, "w") as f:
        f.write("image: %s\n" % (base + ".pgm"))
        f.write("resolution: %s\n" % msg.info.resolution)
        f.write("origin: [%s, %s, 0.0]\n" % (msg.info.origin.position.x, msg.info.origin.position.y))
        f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

    node.get_logger().info("Saved: %s and %s" % (yaml_path, pgm_path))
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
