from skeleton_tracking.skeletonization_algorithm.skeletonization_algorithm import (
    BaseSkeletonizationAlgorithm, Skeleton2D, Keypoint2D, KeypointID
)
from typing import List, Tuple, Type
import mediapipe as mp
import numpy as np
from enum import IntEnum


MediaPipeKeypointID : IntEnum  = mp.solutions.pose.PoseLandmark


class MediaPipeSkeletonization(BaseSkeletonizationAlgorithm):
    
    def __init__(self, params: dict):
        min_detection_confidence = params.get("min_detection_confidence", 0.5)
        min_tracking_confidence = params.get("min_tracking_confidence", 0.5)

        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

    @staticmethod
    def get_parameters_names() -> List[str]:
        return ["min_detection_confidence", 
                "min_tracking_confidence"]

    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        skeletons = []
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            skeleton_id = 1
            keypoints = []

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
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
