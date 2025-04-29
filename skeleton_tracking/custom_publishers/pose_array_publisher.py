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

from typing import Dict
import numpy as np
from rclpy.publisher import Publisher
from skeleton_tracking.custom_publishers.custom_publisher_base import (
    CustomPublisherBase,
)
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header


def build_skeleton_as_pose_array_msg(keypoints: Dict[int, np.array], header):
    skeleton_msg = PoseArray()
    skeleton_msg.header = header
    for keypoint in keypoints.values():
        pose = Pose()
        pose.position.x = keypoint[0]
        pose.position.y = keypoint[1]
        pose.position.z = keypoint[2]
        skeleton_msg.poses.append(pose)
    return skeleton_msg


class PoseArrayPublisher(CustomPublisherBase):

    def get_msg_type(self):
        return PoseArray

    def get_topic(self) -> str:
        return 'poses'

    # @abstractmethod
    # def get_qos_profile(self) -> Union[QoSProfile, int]:
    #     pass

    def publish(
        self,
        publisher: Publisher,
        keypoints: Dict[int, np.array],
        header: Header,
    ):
        skeleton_msg = build_skeleton_as_pose_array_msg(keypoints, header)
        publisher.publish(skeleton_msg)

    def get_message(self, keypoints: Dict[int, np.array], header: Header):
        return build_skeleton_as_pose_array_msg(keypoints, header)
