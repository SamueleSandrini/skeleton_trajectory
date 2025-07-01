from skeleton_tracking.skeletonization_algorithm.skeletonization_algorithm import (
    BaseSkeletonizationAlgorithm)
from skeleton_tracking.skeleton_interfaces import (
  Skeleton2D, Keypoint2D, KeypointID, Skeleton3D)
from typing import List, Tuple, Type, Any
import mediapipe as mp
import numpy as np
from enum import IntEnum


MediaPipeKeypointID : IntEnum  = mp.solutions.pose.PoseLandmark


class MediaPipeSkeletonization(BaseSkeletonizationAlgorithm):
    
    def __init__(self):
        pass
        # , params: dict):
        # min_detection_confidence = params.get("min_detection_confidence", 0.5)
        # min_tracking_confidence = params.get("min_tracking_confidence", 0.5)

        # self.pose = mp.solutions.pose.Pose(
        #     min_detection_confidence = min_detection_confidence,
        #     min_tracking_confidence = min_tracking_confidence
        # )
        # self.results = None

    def initialize(self, params: dict):
        min_detection_confidence = params.get("min_detection_confidence", 0.5)
        min_tracking_confidence = params.get("min_tracking_confidence", 0.5)

        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.results = None
    @staticmethod
    def get_parameters_names() -> List[Tuple[str, Any]]:
        return [("min_detection_confidence", 0.5), 
                ("min_tracking_confidence", 0.5)]

    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        skeletons = []
        self.results = self.pose.process(rgb_image)

        if self.results.pose_landmarks:
            skeleton_id = 1
            keypoints = []

            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                keypoints.append(Keypoint2D(
                    id = MediaPipeKeypointID(idx),
                    x = landmark.x,
                    y = landmark.y,
                    metadata = {"visibility": landmark.visibility},
                ))

            skeletons.append(Skeleton2D(id = skeleton_id, 
                                        keypoints = keypoints))

        return skeletons

    @property
    def topology(self) -> List[Tuple[KeypointID, KeypointID]]:
        return list(mp.solutions.pose.POSE_CONNECTIONS)
        # return [(MediaPipeKeypointID(a), MediaPipeKeypointID(b)) 
        #         for a, b in mp.solutions.pose.POSE_CONNECTIONS]

    @property
    def keypoint_enum(self) -> Type[KeypointID]:
        return MediaPipeKeypointID

    def get_color_frame_with_detection(self, 
        rgb_image: np.ndarray,
    ) -> np.ndarray:
        color_frame_with_detection = rgb_image.copy()
        mp.solutions.drawing_utils.draw_landmarks(color_frame_with_detection,
                                                  self.results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS)
        return color_frame_with_detection


    def is_not_person(self, skeleton: Skeleton3D) -> bool:
        # Check left shoulder to left hip distance
        mp_pose = mp.solutions.pose
        landmark_dict = {kp.id: [kp.x, kp.y, kp.z] for kp in skeleton.keypoints}
        indexes_present = landmark_dict.keys()
        if (mp_pose.PoseLandmark.LEFT_SHOULDER.value in indexes_present and
                mp_pose.PoseLandmark.LEFT_HIP.value in indexes_present):

            # idx_left_shoulder = indexes_present[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            # idx_left_hip = indexes_present[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_bust = np.linalg.norm(np.array(landmark_dict[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) -
                                       np.array(landmark_dict[mp_pose.PoseLandmark.LEFT_HIP.value]))

            if left_bust < 0.25 or left_bust > 1.2:  # Converted to meters
                return True

        # Check right shoulder to right hip distance
        if (mp_pose.PoseLandmark.RIGHT_SHOULDER.value in indexes_present and
                mp_pose.PoseLandmark.RIGHT_HIP.value in indexes_present):

            # idx_right_shoulder = indexes_present[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            # idx_right_hip = indexes_present[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_bust = np.linalg.norm(np.array(landmark_dict[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) -
                                        np.array(landmark_dict[mp_pose.PoseLandmark.RIGHT_HIP.value]))

            if right_bust < 0.25 or right_bust > 1.2:  # Converted to meters
                return True

        # Check if there are fewer than two keypoints detected
        if len(indexes_present) < 2:
            # self.internal_node.get_logger().info(
            #     'Keypoints detected for the human are less than 2.')
            return True

        return False