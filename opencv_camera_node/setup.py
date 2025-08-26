from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'opencv_camera_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='TurtleBot User',
    maintainer='TurtleBot User',
    keywords=['ROS', 'camera', 'OpenCV'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    description='OpenCV USB camera node for TurtleBot (RPi, no RVIZ).',
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'opencv_camera_node = opencv_camera_node.opencv_camera_node:main',
        ],
    },
)
