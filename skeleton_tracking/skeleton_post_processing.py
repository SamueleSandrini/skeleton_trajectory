
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

EPS = 0.1


def de_normalize_keypoint(keypoint, width, height):
    return floor(keypoint.x * width), floor(keypoint.y * height)


def initialize_pose_detector(min_detection_confidence=0.8,
                             min_tracking_confidence=0.5):
    pose_detector = pose_detector = mp.solutions.pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    return pose_detector


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


class SkeletonDetection(PostProcessing):

    def __init__(self):
        self.internal_node = rclpy.create_node('skeleton_detection_node')
        self.skeleton_marker_publisher = self.internal_node.create_publisher(
            MarkerArray, 'skeleton_markers', 10)
        self.skeleton_topology_publisher = self.internal_node.create_publisher(
            Marker, 'skeleton', 10)
        self.debug_publisher = self.internal_node.create_publisher(
            Image, 'debug_skeleton_detection_image', 10)

        self.internal_node.declare_parameter('skeleton_detection_node.debug_topic', True)
        self.internal_node.declare_parameter(
            'skeleton_detection_node.min_detection_confidence', 0.8)
        self.internal_node.declare_parameter(
            'skeleton_detection_node.min_tracking_confidence', 0.5)
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
        self.pose_detector = initialize_pose_detector(self.min_detection_confidence,
                                                      self.min_tracking_confidence)

        self.cv_bridge = CvBridge()

    def initialize(self, camera_info: CameraInfo):
        self.camera_info = camera_info
        self.internal_node.get_logger().info('Skeleton detection running ...')

    def build_skeleton_topology_msg(self, indexes, keypoints):
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

        for segment in mp.solutions.pose.POSE_CONNECTIONS:
            idx1, idx2 = segment
            if idx1 in indexes and idx2 in indexes:
                x1, y1, z1 = keypoints[indexes.index(idx1)]
                x2, y2, z2 = keypoints[indexes.index(idx2)]
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
        result = self.pose_detector.process(color_frame)
        if result.pose_landmarks is None:
            return

        keypoints = result.pose_landmarks.landmark

        indexes = []
        keypoints_3d = []

        marker_array = MarkerArray()

        for keypoint_id, keypoint in enumerate(keypoints):
            if not (keypoint.visibility > 0.8 and keypoint.x <
                    1 and keypoint.y < 1 and keypoint.x > 0 and keypoint.y > 0):
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

            indexes.append(keypoint_id)
            keypoints_3d.append((x_m, y_m, z_m))

            marker = self.create_marker_msg(keypoint_id, x_m, y_m, z_m)
            marker_array.markers.append(marker)

        if self.is_not_person(indexes, keypoints_3d):
            self.pose_detector = initialize_pose_detector(self.min_detection_confidence,
                                                          self.min_tracking_confidence)
        else:
            self.skeleton_marker_publisher.publish(marker_array)
            self.skeleton_topology_publisher.publish(
                self.build_skeleton_topology_msg(indexes, keypoints_3d))

        if self.debug:
            color_frame_with_detection = color_frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(color_frame_with_detection,
                                                      result.pose_landmarks,
                                                      mp.solutions.pose.POSE_CONNECTIONS)

            self.debug_publisher.publish(self.cv_bridge.cv2_to_imgmsg(color_frame_with_detection))

    def is_not_person(self, indexes_present, landmark_list):
        # Check left shoulder to left hip distance
        mp_pose = mp.solutions.pose
        if (mp_pose.PoseLandmark.LEFT_SHOULDER.value in indexes_present and
                mp_pose.PoseLandmark.LEFT_HIP.value in indexes_present):

            idx_left_shoulder = indexes_present.index(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            idx_left_hip = indexes_present.index(mp_pose.PoseLandmark.LEFT_HIP.value)
            left_bust = np.linalg.norm(np.array(landmark_list[idx_left_shoulder]) -
                                       np.array(landmark_list[idx_left_hip]))

            if left_bust < 0.25 or left_bust > 1.2:  # Converted to meters
                return True

        # Check right shoulder to right hip distance
        if (mp_pose.PoseLandmark.RIGHT_SHOULDER.value in indexes_present and
                mp_pose.PoseLandmark.RIGHT_HIP.value in indexes_present):

            idx_right_shoulder = indexes_present.index(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            idx_right_hip = indexes_present.index(mp_pose.PoseLandmark.RIGHT_HIP.value)
            right_bust = np.linalg.norm(np.array(landmark_list[idx_right_shoulder]) -
                                        np.array(landmark_list[idx_right_hip]))

            if right_bust < 0.25 or right_bust > 1.2:  # Converted to meters
                return True

        # Check if there are fewer than two keypoints detected
        if len(indexes_present) < 2:
            self.internal_node.get_logger().info(
                'Keypoints detected for the human are less than 2.')
            return True

        return False

    def process_frame(self, color_frame):
        pass
