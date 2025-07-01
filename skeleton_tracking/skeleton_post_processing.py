
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


from math import floor

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import mediapipe as mp

import numpy as np
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from vision_system.post_processing import PostProcessing
from vision_system.vision_system_utils import deproject_pixel_to_point
from visualization_msgs.msg import Marker, MarkerArray
from skeleton_tracking.skeletonization_algorithm.skeletonization_algorithm import (
    load_skeleton_algorithm
)
from skeleton_tracking.skeleton_interfaces import Skeleton3D, Keypoint3D, KeypointID
from typing import List, Dict, Any, Tuple

EPS = 0.1


def de_normalize_keypoint(keypoint, width, height):
    return floor(keypoint.x * width), floor(keypoint.y * height)


# def initialize_pose_detector(min_detection_confidence=0.8,
#                              min_tracking_confidence=0.5):
#     pose_detector = pose_detector = mp.solutions.pose.Pose(
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence
#     )
#     return pose_detector


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

SKELETON_ALGORITHM_MODULE_MAP = {
    'mediapipe': 'skeleton_tracking.skeletonization_algorithm.mediapipe_skeletonization.MediaPipeSkeletonization',
    'yolo' : 'skeleton_tracking.skeletonization_algorithm.yolo_skeletonization.YoloSkeletonization',
}

class SkeletonDetection(PostProcessing):

    def __init__(self):
        self.internal_node = rclpy.create_node('skeleton_detection_node')
        self.skeleton_marker_publisher = self.internal_node.create_publisher(
            MarkerArray, 'skeleton_markers', 10)
        self.skeleton_topology_publisher = self.internal_node.create_publisher(
            Marker, 'skeleton', 10)
        self.debug_publisher = self.internal_node.create_publisher(
            Image, 'debug_skeleton_detection_image', 10)
        
        self.internal_node.declare_parameter('skeleton_algorithm', 'mediapipe')
        # self.declare_parameter('skeleton_algorithm_params', {})
        self.internal_node.declare_parameter('skeleton_algorithm_module', '')
        self.internal_node.declare_parameter('skeleton_algorithm_class', '')

        skeleton_algorithm_name = self.get_parameter('skeleton_algorithm').value
        # skeleton_algorithm_params = self.get_parameter('skeleton_algorithm_params').value

        if skeleton_algorithm_name not in SKELETON_ALGORITHM_MODULE_MAP:
            self.get_logger().info('Skeleton algorithm not recognized, trying to load custom module...')
            skeleton_algorithm_module = self.get_parameter('skeleton_algorithm_module').value
            skeleton_algorithm_class = self.get_parameter('skeleton_algorithm_class').value
            if not self.skeleton_algorithm_module or not self.skeleton_algorithm_class:
                self.get_logger().error("Custom algorithm specified, but module or class is missing!")
                raise ValueError("Custom algorithm module or class not specified.")
            skeleton_algorithm = load_skeleton_algorithm(skeleton_algorithm_module, 
                                                         skeleton_algorithm_class)
        else:
            skeleton_algorithm = SKELETON_ALGORITHM_MODULE_MAP[skeleton_algorithm_name]
            self.internal_node.get_logger().info(f'Using skeleton algorithm: {self.skeleton_algorithm}')
        skelton_algorithm_params_name = skeleton_algorithm.get_parameters_names()
        sekelton_algorithm_params = {}
        for (param_name, param_default) in skelton_algorithm_params_name:
            self.internal_node.declare_parameter(skeleton_algorithm_name + '.' + param_name, 
                                                 param_default)
            sekelton_algorithm_params[param_name] = self.internal_node.get_parameter(
                skeleton_algorithm_name + '.' + param_name).value
        self.skeleton_algorithm = skeleton_algorithm(sekelton_algorithm_params)

        self.internal_node.declare_parameter('skeleton_detection_node.debug_topic', True)
        self.internal_node.declare_parameter('skeleton_detection_node.roi_half_size', 0)

        self.debug = self.internal_node.get_parameter('skeleton_detection_node.debug_topic').value
        self.min_detection_confidence = self.internal_node.get_parameter(
            'skeleton_detection_node.min_detection_confidence').value
        self.min_tracking_confidence = self.internal_node.get_parameter(
            'skeleton_detection_node.min_tracking_confidence').value
        self.roi_half_size = self.internal_node.get_parameter(
            'skeleton_detection_node.roi_half_size').value

        if self.roi_half_size < 0:
            self.internal_node.get_logger().warning('ROI half size must be greater than 0')
            self.roi_half_size = 0

        self.camera_info = None
        # self.pose_detector = initialize_pose_detector(self.min_detection_confidence,
        #                                               self.min_tracking_confidence)

        self.cv_bridge = CvBridge()

    def initialize(self, camera_info: CameraInfo):
        self.camera_info = camera_info
        self.internal_node.get_logger().info('Skeleton detection running ...')
    
    def build_skeleton_topology_msg(self, 
                                    topology: List[Tuple[KeypointID, KeypointID]], 
                                    skeleton_3d: Skeleton3D):
        skeleton = Marker()
        skeleton.header.frame_id = self.camera_info.header.frame_id
        skeleton.header.stamp = self.internal_node.get_clock().now().to_msg()
        skeleton.ns = 'skeleton_segments'
        skeleton.type = Marker.LINE_LIST
        skeleton.action = Marker.ADD
        skeleton.scale.x = 0.03
        skeleton.color.r = 1.0
        skeleton.color.g = 0.0
        skeleton.color.b = 0.0
        skeleton.color.a = 1.0
        keypoint_dict = {kp.id: kp for kp in skeleton_3d.keypoints}

        for segment in topology:
            idx1, idx2 = segment
            if idx1 in keypoint_dict.keys() and idx2 in keypoint_dict.keys():
                kp1 = keypoint_dict[idx1]
                kp2 = keypoint_dict[idx2]
                x1, y1, z1 = kp1.x, kp1.y, kp1.z
                x2, y2, z2 = kp2.x, kp2.y, kp2.z
                skeleton.points.append(Point(x=float(x1), y=float(y1), z=float(z1)))
                skeleton.points.append(Point(x=float(x2), y=float(y2), z=float(z2)))
        return skeleton

    def create_marker_msg(self, marker_id, x, y, z):
        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.internal_node.get_clock().now().to_msg()

        marker.ns = 'skeleton'
        marker.id = marker_id
        marker.type = Marker.SPHERE

        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)

        return marker

    def process_frames(self, color_frame, distance_frame):
        skeletons = self.skeleton_algorithm.extract_skeletons(color_frame)
        if not skeletons:
            return
        
        skeletons_3d = []
        for skeleton in skeletons:
            marker_array = MarkerArray()
            skeleton_3d = Skeleton3D(id = skeleton.id, frame_id = self.camera_info.header.frame_id, keypoints = [])
            for keypoint in skeleton.keypoints:
                if not (keypoint.x < 1 and keypoint.y < 1 and keypoint.x > 0 and keypoint.y > 0):
                    continue
                x, y = de_normalize_keypoint(keypoint,
                                            self.camera_info.width,
                                            self.camera_info.height)

                y_min = clamp(y - self.roi_half_size, 0, self.camera_info.height - 1)
                # +1 to include the last pixel
                y_max = clamp(y + 1 + self.roi_half_size, 0, self.camera_info.height - 1)
                x_min = clamp(x - self.roi_half_size, 0, self.camera_info.width - 1)
                # +1 to include the last pixel
                x_max = clamp(x + 1 + self.roi_half_size, 0, self.camera_info.width - 1)

                roi_distance = distance_frame[y_min:y_max, x_min:x_max]

                # Filters out invalid values and calculates the mean ignoring zero and NaN
                valid_values = roi_distance[(roi_distance > 0) & np.isfinite(roi_distance)]

                if valid_values.size > 0:
                    average_depth_pixels = np.mean(valid_values)
                else:
                    continue

                x_m, y_m, z_m = deproject_pixel_to_point(
                    (x, y), average_depth_pixels, self.camera_info)

                keypoint_3d = Keypoint3D(id = keypoint.id,
                                         x = x_m,
                                         y = y_m,
                                         z = z_m)
                skeleton_3d.append_keypoint(keypoint_3d)
                # indexes.append(keypoint_id)
                # keypoints_3d.append((x_m, y_m, z_m))

            if self.is_not_person(skeleton_3d):
                continue
            skeletons_3d.append(skeleton_3d)
            marker = self.create_marker_msg(keypoint.id, x_m, y_m, z_m)
            marker_array.markers.append(marker)
            self.skeleton_marker_publisher.publish(marker_array)
            self.skeleton_topology_publisher.publish(
                self.build_skeleton_topology_msg(self.skeleton_algorithm.topology, skeleton_3d))

        if self.debug:
            self.debug_publisher.publish(
                self.cv_bridge.cv2_to_imgmsg(
                    self.self.skeleton_algorithm.get_color_frame_with_detection(color_frame)))

    # def is_not_person(self, indexes_present, landmark_list):
    #     # Check left shoulder to left hip distance
    #     mp_pose = mp.solutions.pose
    #     if (mp_pose.PoseLandmark.LEFT_SHOULDER.value in indexes_present and
    #             mp_pose.PoseLandmark.LEFT_HIP.value in indexes_present):

    #         idx_left_shoulder = indexes_present.index(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    #         idx_left_hip = indexes_present.index(mp_pose.PoseLandmark.LEFT_HIP.value)
    #         left_bust = np.linalg.norm(np.array(landmark_list[idx_left_shoulder]) -
    #                                    np.array(landmark_list[idx_left_hip]))

    #         if left_bust < 0.25 or left_bust > 1.2:  # Converted to meters
    #             return True

    #     # Check right shoulder to right hip distance
    #     if (mp_pose.PoseLandmark.RIGHT_SHOULDER.value in indexes_present and
    #             mp_pose.PoseLandmark.RIGHT_HIP.value in indexes_present):

    #         idx_right_shoulder = indexes_present.index(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    #         idx_right_hip = indexes_present.index(mp_pose.PoseLandmark.RIGHT_HIP.value)
    #         right_bust = np.linalg.norm(np.array(landmark_list[idx_right_shoulder]) -
    #                                     np.array(landmark_list[idx_right_hip]))

    #         if right_bust < 0.25 or right_bust > 1.2:  # Converted to meters
    #             return True

    #     # Check if there are fewer than two keypoints detected
    #     if len(indexes_present) < 2:
    #         self.internal_node.get_logger().info(
    #             'Keypoints detected for the human are less than 2.')
    #         return True

    #     return False

    def process_frame(self, color_frame):
        pass
