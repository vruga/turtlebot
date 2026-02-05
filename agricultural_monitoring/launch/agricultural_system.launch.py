#!/usr/bin/env python3
"""
ROS2 Launch File for Agricultural Disease Detection System

Launches all system components with proper priority settings:
- Frame capture node (normal priority)
- Inference worker (nice +15, low priority)
- ESP32 controller (normal priority)
- Claude client (nice +10, slightly low priority)

Author: Agricultural Robotics Team
License: MIT
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    TimerAction
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate the launch description for agricultural monitoring system."""

    # Get package directory
    pkg_dir = Path(__file__).parent.parent

    # Declare launch arguments
    declare_config_dir = DeclareLaunchArgument(
        'config_dir',
        default_value=str(pkg_dir / 'config'),
        description='Path to configuration directory'
    )

    declare_enable_dashboard = DeclareLaunchArgument(
        'enable_dashboard',
        default_value='true',
        description='Enable web dashboard'
    )

    declare_enable_llm = DeclareLaunchArgument(
        'enable_llm',
        default_value='true',
        description='Enable LLM recommendations'
    )

    declare_dashboard_port = DeclareLaunchArgument(
        'dashboard_port',
        default_value='8080',
        description='Dashboard web server port'
    )

    # Source paths
    src_dir = pkg_dir / 'src'

    # Frame Capture Node (normal priority)
    frame_capture_node = ExecuteProcess(
        cmd=[
            'python3',
            str(src_dir / 'camera' / 'frame_capture_node.py')
        ],
        name='frame_capture_node',
        output='screen',
        emulate_tty=True
    )

    # Inference Worker (nice +15, low priority to not interfere with teleop)
    # Using nice command to set lower priority
    inference_worker = ExecuteProcess(
        cmd=[
            'nice', '-n', '15',
            'python3',
            str(src_dir / 'inference' / 'inference_worker.py')
        ],
        name='inference_worker',
        output='screen',
        emulate_tty=True
    )

    # ESP32 Controller (normal priority for safety responsiveness)
    esp32_controller = ExecuteProcess(
        cmd=[
            'python3',
            str(src_dir / 'spray_control' / 'esp32_controller.py')
        ],
        name='esp32_controller',
        output='screen',
        emulate_tty=True
    )

    # Claude Client (nice +10, slightly low priority)
    claude_client = ExecuteProcess(
        cmd=[
            'nice', '-n', '10',
            'python3',
            str(src_dir / 'llm' / 'claude_client.py')
        ],
        name='claude_client',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('enable_llm'))
    )

    # Dashboard (Flask web server)
    dashboard_server = ExecuteProcess(
        cmd=[
            'python3',
            str(src_dir / 'dashboard' / 'app.py')
        ],
        name='dashboard',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('enable_dashboard'))
    )

    # Log startup message
    startup_log = LogInfo(
        msg='\n' + '='*60 + '\n'
            'Agricultural Disease Detection System Starting...\n'
            '='*60 + '\n'
            'Components:\n'
            '  - Frame Capture Node (normal priority)\n'
            '  - Inference Worker (nice +15)\n'
            '  - ESP32 Controller (normal priority)\n'
            '  - Claude Client (nice +10)\n'
            '  - Web Dashboard\n'
            '='*60
    )

    # Delayed start for components that depend on others
    delayed_inference = TimerAction(
        period=2.0,  # Wait 2 seconds
        actions=[inference_worker]
    )

    delayed_esp32 = TimerAction(
        period=1.0,  # Wait 1 second
        actions=[esp32_controller]
    )

    delayed_claude = TimerAction(
        period=3.0,  # Wait 3 seconds
        actions=[claude_client]
    )

    delayed_dashboard = TimerAction(
        period=2.0,  # Wait 2 seconds
        actions=[dashboard_server]
    )

    return LaunchDescription([
        # Launch arguments
        declare_config_dir,
        declare_enable_dashboard,
        declare_enable_llm,
        declare_dashboard_port,

        # Startup message
        startup_log,

        # Start frame capture first
        frame_capture_node,

        # Start other components with delays
        delayed_esp32,
        delayed_inference,
        delayed_claude,
        delayed_dashboard,
    ])


if __name__ == '__main__':
    generate_launch_description()
