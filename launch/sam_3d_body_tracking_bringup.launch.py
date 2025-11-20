
# Copyright 2024 National Research Council STIIMA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    pkg_dir = get_package_share_directory('skeleton_tracking')
    # vision_system_pkg_dir = get_package_share_directory('vision_system')

    vision_config_path_cmd = DeclareLaunchArgument(
        'vision_config_path',
        default_value=pkg_dir + '/config/sam_3d_body_processing_config.yaml',
        description='Full path to the config file of the vision system')

    vision_system_node_cmd_rs1 = Node(
        package='vision_system',
        executable='vision_system_node',
        name='camera_node',
        output='screen',
        parameters=[
            LaunchConfiguration('vision_config_path')
        ])

    rs1_group = GroupAction(
        actions=[PushRosNamespace('rs1'),
                # vision_config_path_cmd,
                vision_system_node_cmd_rs1])
                 
    return LaunchDescription([
        vision_config_path_cmd,
        rs1_group,
    ])

