#!/usr/bin/env python3
"""
Combined TurtleBot3 Robot Bringup + SLAM Launch File
Launches both robot hardware and Cartographer SLAM together (no RVIZ).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Get TurtleBot3 model from environment
    turtlebot3_model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    # Robot bringup launch
    robot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_bringup'),
                'launch',
                'robot.launch.py'
            ])
        ])
    )

    # SLAM launch (without RViz for headless systems)
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_cartographer'),
                'launch',
                'cartographer.launch.py'
            ])
        ]),
        launch_arguments={
            'use_rviz': 'false'
        }.items()
    )

    return LaunchDescription([
        robot_bringup,
        slam_launch,
    ])
