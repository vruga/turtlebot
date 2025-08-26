# TurtleBot SLAM + Camera (Raspberry Pi, no RVIZ)

TurtleBot3 + Cartographer SLAM + OpenCV USB camera on Raspberry Pi. Web server for live camera and map. No laptop, no RVIZ.

## Setup on the Pi

1. **Clone and install the ROS2 package**
   ```bash
   cd ~
   git clone <YOUR_REPO_URL> turtlebot_slam_camera
   cd turtlebot_slam_camera
   cp -r opencv_camera_node ~/turtlebot3_ws/src/
   cd ~/turtlebot3_ws
   source /opt/ros/humble/setup.bash
   colcon build --packages-select opencv_camera_node
   source install/setup.bash
   ```

2. **Environment** (add to `~/.bashrc` or run each time)
   ```bash
   source /opt/ros/humble/setup.bash
   [ -f ~/turtlebot3_ws/install/setup.bash ] && source ~/turtlebot3_ws/install/setup.bash
   export TURTLEBOT3_MODEL=burger
   export LDS_MODEL=LDS-01
   ```

## Run (from repo folder)

```bash
cd ~/turtlebot_slam_camera
./start_full_system.sh
```

In other terminals (from repo folder):

- **Teleop:** `./run_turtlebot3_teleop.sh` or `ros2 run turtlebot3_teleop teleop_keyboard`
- **Web server:** `./start_server.sh` → open http://\<RPI_IP\>:5000
- **Save map:** `./save_map.sh my_map` → files in `~/maps/`
- **Check /map:** `./check_map_topic.sh`
- **List cameras:** `./check_camera.sh`

See `HOW_IT_WORKS.txt` for details.

## Repo layout

- `start_full_system.sh` – main entry (robot + SLAM + camera, or --no-camera / --camera-only)
- `start_server.sh` – web server (camera + map)
- `save_map.sh` / `save_map_from_topic.py` – save /map to ~/maps (no Nav2)
- `check_camera.sh` / `check_map_topic.sh` – diagnostics
- `turtlebot3_slam_combined.launch.py` – robot + SLAM only (no camera)
- `opencv_camera_node/` – ROS2 package (copy to turtlebot3_ws/src and build)
