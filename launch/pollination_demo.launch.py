#!/usr/bin/env python3
#
# Author: Chuizheng Kong
# Date: 05/17/2024

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    nvidia_vision_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('vision_pkg'), 
                         'launch/isaac_ros_endoscope.launch.py')
        ) 
    )

    microscope_pub_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('vision_pkg'), 
                         'launch/isaac_ros_microscope.launch.py')
        )
    )

    take_picture_action_server_node = Node(
        package='gen3_7dof',
        executable='take_pictures_action_server',
        name='take_pictures_action_server'
    )

    yolo_display_node = Node(
        package='proj_farmhand',
        executable='yolo_display',
        name='yolo_display'
    )

    yolo_pursuit_node = Node(
        package='proj_farmhand',
        executable='yolo_pursuit_action_server',
        name='yolo_pursuit_action_server'
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(nvidia_vision_pipeline)
    # ld.add_action(microscope_pub_node)
    ld.add_action(take_picture_action_server_node)
    ld.add_action(yolo_display_node)
    ld.add_action(yolo_pursuit_node)
    return ld