#!/usr/bin/env python3
"""
Web server for TurtleBot SLAM + camera on Raspberry Pi.
Shows live camera feed and SLAM map. Run after start_full_system.sh.
"""

import io
import threading
from flask import Flask, Response, render_template_string
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2

app = Flask(__name__)

# Shared buffers (updated by ROS callbacks)
latest_camera = None
latest_map_png = None
lock = threading.Lock()


class ROSBridge(Node):
    def __init__(self):
        super().__init__("slam_camera_server_bridge")
        self._map_count = 0
        self.sub_cam = self.create_subscription(
            CompressedImage, "camera/image_compressed", self.cam_cb, 10
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, "map", self.map_cb, 10
        )

    def cam_cb(self, msg):
        global latest_camera
        with lock:
            latest_camera = bytes(msg.data)

    def map_cb(self, msg):
        global latest_map_png
        try:
            h, w = msg.info.height, msg.info.width
            if h == 0 or w == 0:
                if self._map_count == 0:
                    self.get_logger().info("Map message has 0 size (%dx%d), waiting for non-empty map" % (w, h))
                return
            data = np.array(msg.data, dtype=np.int8).reshape((h, w))
            # -1 unknown, 0 free, 100 occupied -> grayscale image
            img = np.uint8(np.clip((data + 1) * 127.5, 0, 255))
            _, buf = cv2.imencode(".png", img)
            with lock:
                latest_map_png = buf.tobytes()
            self._map_count += 1
            if self._map_count == 1:
                self.get_logger().info("First /map received (%dx%d) - map will show in browser" % (w, h))
        except Exception as e:
            self.get_logger().warn("map_cb error: %s" % e)


def gen_frames():
    while True:
        with lock:
            frame = latest_camera
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        import time
        time.sleep(0.05)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/map_image")
def map_image():
    with lock:
        data = latest_map_png
    if data:
        return Response(data, mimetype="image/png")
    return Response(status=204)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TurtleBot SLAM + Camera</title>
    <style>
        body { font-family: sans-serif; margin: 16px; background: #1e1e1e; color: #ddd; }
        h1 { color: #4CAF50; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; }
        .panel { background: #2d2d2d; border-radius: 8px; padding: 16px; border: 1px solid #444; }
        .panel h2 { margin-top: 0; color: #81C784; }
        img { max-width: 100%; height: auto; border: 1px solid #555; border-radius: 4px; }
        .info { font-size: 14px; line-height: 1.6; color: #aaa; }
        code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; }
        a { color: #4CAF50; }
    </style>
</head>
<body>
    <h1>TurtleBot SLAM + Camera</h1>

    <div class="grid">
        <div class="panel">
            <h2>Camera</h2>
            <img src="/video_feed" alt="Camera" style="width:100%;">
            <p class="info">
                <strong>Connect camera:</strong> Plug USB camera. List devices: <code>ls /dev/video*</code>.
                <code>/dev/video0</code> = device_id 0, <code>/dev/video2</code> = device_id 2.
                If camera doesn't show, restart full system with: <code>ros2 launch opencv_camera_node turtlebot_slam_camera.launch.py camera_device_id:=2</code> (use your device number).
            </p>
        </div>
        <div class="panel">
            <h2>SLAM Map</h2>
            <img id="mapimg" src="/map_image" alt="Map" style="width:100%;" onerror="this.style.display='none'">
            <p id="nomap" class="info" style="display:none;">No map yet. Drive the robot with teleop to build the map.</p>
            <p class="info">
                <strong>Where are SLAM files stored?</strong> You map the area by driving the robot (teleop). The map is built live in memory. To save it, run in a terminal:<br>
                <code>./save_map.sh my_map</code> (no Nav2 needed)<br>
                Files are saved to <code>~/maps/</code>: <code>my_map.yaml</code> and <code>my_map.pgm</code>. Use any name instead of <code>my_map</code>.
            </p>
        </div>
    </div>

    <div class="panel" style="margin-top: 20px;">
        <h2>Quick reference</h2>
        <p class="info">
            <strong>Start full system:</strong> <code>./start_full_system.sh</code><br>
            <strong>Teleop (other terminal):</strong> <code>./run_turtlebot3_teleop.sh</code> or <code>ros2 run turtlebot3_teleop teleop_keyboard</code><br>
            <strong>Save map:</strong> <code>./save_map.sh NAME</code> → files in <code>~/maps/NAME.yaml</code> and <code>~/maps/NAME.pgm</code> (run while full system is running)
        </p>
    </div>

    <script>
        setInterval(function() {
            var img = document.getElementById('mapimg');
            img.src = '/map_image?t=' + Date.now();
        }, 2000);
        document.getElementById('mapimg').onload = function() {
            document.getElementById('nomap').style.display = 'none';
            this.style.display = 'block';
        };
        document.getElementById('mapimg').onerror = function() {
            document.getElementById('nomap').style.display = 'block';
            this.style.display = 'none';
        };
    </script>
</body>
</html>
"""


def main():
    import sys
    import os
    rclpy.init(args=sys.argv)
    node = ROSBridge()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    port = 5000
    host = "0.0.0.0"
    print("Server: http://{}:{}/  (from another device: http://<RPI_IP>:{})".format(host, port, port))
    print("")
    print("ROS_DOMAIN_ID=%s (server and full system must use the same value)" % os.environ.get("ROS_DOMAIN_ID", "0"))
    print("If map stays empty: 1) start_full_system.sh must be running (robot+SLAM+camera)")
    print("                   2) Drive the robot with teleop so SLAM builds the map")
    print("                   3) Start server AFTER full system (same terminal env / same Pi)")
    print("")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
