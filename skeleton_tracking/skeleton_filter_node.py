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

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from skeleton_tracking.skeleton_traking_utils import (
    build_skeleton_topology_msg,
    build_marker_array_msg,
    get_num_keypoints,
)
from std_msgs.msg import Header
from skeleton_tracking.kalman_filter import KalmanFilter
from skeleton_tracking.custom_publishers.custom_publisher_base import (
    CustomPublisherBase,
)

import importlib

N_KEYPOINTS = get_num_keypoints()


class SkeletonFilterNode(Node):
    def __init__(self):
        super().__init__('skeleton_filter_node')

        # Parameters
        self.declare_parameter('custom_publishers', [''])
        self.declare_parameter('keypoint_timeout', 0.1)

        self.declare_parameter(
            'q_noise', [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        )
        self.declare_parameter('r_noise', [0.05, 0.05, 0.1])
        self.declare_parameter('dt', 1.0 / 30.0)

        custom_publishers = self.get_parameter('custom_publishers').value
        self.keypoint_timeout = self.get_parameter('keypoint_timeout').value

        self.custom_publishers = {}
        self.load_custom_publishers_modules(custom_publishers)

        # Subscriber
        self.subscriber_keypoints = self.create_subscription(
            MarkerArray, 'skeleton_markers', self.callback_keypoint, 10
        )

        # # Init filters
        q_noise = self.get_parameter('q_noise').value
        r_noise = self.get_parameter('r_noise').value
        dt = self.get_parameter('dt').value
        self.keypoints_filters = [
            KalmanFilter(q_noise, r_noise, dt) for _ in range(N_KEYPOINTS)
        ]
        self.time_keypoint = np.zeros((N_KEYPOINTS,))

        # Publisher
        self.skeleton_marker_publisher = self.create_publisher(
            MarkerArray, 'skeleton_markers_filtered', 10
        )
        self.skeleton_topology_publisher = self.create_publisher(
            Marker, 'skeleton_filtered', 10
        )

        # TF broadcaster
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # State variables
        self.list_of_indexes_pres = []

        self.get_logger().info('SkeletonFilterNode initialized.')

    def load_custom_publishers_modules(self, custom_publishers):
        for publisher in custom_publishers:
            self.declare_parameter(f'{publisher}.module', '')

            publisher_full_module_name = (
                self.get_parameter(f'{publisher}.module')
                .get_parameter_value()
                .string_value
            )

            full_module_name = publisher_full_module_name.rsplit('.', 1)[0]
            class_name = publisher_full_module_name.rsplit('.', 1)[-1]

            self.get_logger().info(
                f'Loading custom publisher: {full_module_name}.{class_name}'
            )
            try:
                # Import the module dynamically
                custom_pub_module = importlib.import_module(full_module_name)
                custom_pub_class = getattr(custom_pub_module, class_name)

                # Check if the loaded class is a subclass of CustomPublisherBase
                if not issubclass(custom_pub_class, CustomPublisherBase):
                    self.get_logger().warning(
                        f'The custom publisher class {custom_pub_class.__name__} is not a subclass of CustomPublisherBase.'
                    )
                    continue

                publisher_instance = custom_pub_class()
                self.get_logger().info(f'Custom publisher {class_name} loaded.')
                publisher = self.create_publisher(
                    publisher_instance.get_msg_type(),
                    publisher_instance.get_topic(),
                    10,
                )
                self.custom_publishers[publisher] = publisher_instance

            except Exception as e:
                self.get_logger().warning(
                    f'Error loading custom publisher: {publisher}, it is not loaded.'
                )
                continue

    def callback_keypoint(self, keypoints_msg: MarkerArray):
        if not keypoints_msg.markers:
            return
        current_time_msg = self.get_clock().now().to_msg()
        current_time = current_time_msg.sec + current_time_msg.nanosec * 1e-9

        frame_id = keypoints_msg.markers[0].header.frame_id

        keypoints = {}
        for keypoint in keypoints_msg.markers:
            id_kp = keypoint.id
            keypoint_position = np.array(
                [
                    keypoint.pose.position.x,
                    keypoint.pose.position.y,
                    keypoint.pose.position.z,
                ]
            )
            # Kalman filter logic
            if (
                current_time - self.time_keypoint[id_kp] > self.keypoint_timeout
            ):  # keypoint lifetime is passed
                self.keypoints_filters[id_kp].initialize(keypoint_position)
            else:
                self.keypoints_filters[id_kp].update(keypoint_position, id_kp)

            keypoint_filtered_position = self.keypoints_filters[
                id_kp
            ].get_output_estimate()
            keypoints[id_kp] = keypoint_filtered_position
            self.time_keypoint[id_kp] = current_time

        # Open loop update for undetected keypoints
        for id_kp in set(range(N_KEYPOINTS)) - set(keypoints.keys()):
            if current_time - self.time_keypoint[id_kp] < self.keypoint_timeout:
                keypoint_nominal_position = self.keypoints_filters[
                    id_kp
                ].open_loop_update()
                keypoints[id_kp] = keypoint_nominal_position

                self.get_logger().info(
                    f'Keypoint: {id_kp} updated in open loop'
                )

        # Publish info
        header = Header(frame_id=frame_id, stamp=current_time_msg)

        skeleton = build_skeleton_topology_msg(keypoints, header)
        self.skeleton_topology_publisher.publish(skeleton)

        marker_array = build_marker_array_msg(keypoints, header)
        self.skeleton_marker_publisher.publish(marker_array)

        for publisher, publisher_instance in self.custom_publishers.items():
            publisher_instance.publish(publisher, keypoints, header)


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
