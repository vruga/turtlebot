#!/usr/bin/env python3
"""
TurtleBot + SLAM + OpenCV camera on Raspberry Pi (no RVIZ).
Launches: robot bringup, Cartographer SLAM (use_rviz:=false), OpenCV camera node.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_device_id = LaunchConfiguration('camera_device_id', default='0')
    # Robot bringup
    turtlebot3_bringup_dir = get_package_share_directory('turtlebot3_bringup')
    robot_launch = os.path.join(turtlebot3_bringup_dir, 'launch', 'robot.launch.py')

    robot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_launch)
    )

    # SLAM (Cartographer) without RVIZ
    turtlebot3_cartographer_dir = get_package_share_directory('turtlebot3_cartographer')
    cartographer_launch = os.path.join(
        turtlebot3_cartographer_dir, 'launch', 'cartographer.launch.py'
    )
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cartographer_launch),
        launch_arguments={'use_rviz': 'false'}.items(),
    )

    # OpenCV camera node (camera_device_id: 0 = /dev/video0, 2 = /dev/video2, etc.)
    camera_node = Node(
        package='opencv_camera_node',
        executable='opencv_camera_node',
        name='opencv_camera_node',
        output='screen',
        parameters=[
            {'device_id': camera_device_id},
            {'width': 640},
            {'height': 480},
            {'fps': 30.0},
            {'frame_id': 'camera_link'},
            {'publish_compressed': True},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_device_id', default_value='0',
                             description='Camera device id: 0 = /dev/video0, 2 = /dev/video2'),
        robot_bringup,
        slam_launch,
        camera_node,
    ])
