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

from geometry_msgs.msg import Pose, PoseArray
import numpy as np
from rclpy.publisher import Publisher
from skeleton_tracking.custom_publishers.custom_publisher_base import CustomPublisherBase
from std_msgs.msg import Header


def compute_centroid(keypoints):
    return np.mean(list(keypoints.values()), axis=0)


def build_centroid_msg(centroid_xyz: np.array, header):
    centroid_msg = PoseArray()
    centroid_msg.header = header

    centroid_pose = Pose()
    centroid_pose.position.x = centroid_xyz[0]
    centroid_pose.position.y = centroid_xyz[1]
    centroid_pose.position.z = centroid_xyz[2]

    centroid_msg.poses.append(centroid_pose)
    return centroid_msg


class CentroidPublisher(CustomPublisherBase):

    def get_msg_type(self):
        return PoseArray

    def get_topic(self) -> str:
        return 'centroid'

    # @abstractmethod
    # def get_qos_profile(self) -> Union[QoSProfile, int]:
    #     pass

    def publish(self, publisher: Publisher,
                keypoints: Dict[int, np.array],
                header: Header):
        # compute centroid of keypoints
        centroid_xyz = compute_centroid(keypoints)
        centroid_msg = build_centroid_msg(centroid_xyz, header)

        publisher.publish(centroid_msg)

    def get_message(self,
                    keypoints: Dict[int, np.array],
                    header: Header):
        centroid_xyz = compute_centroid(keypoints)
        return build_centroid_msg(centroid_xyz, header)
