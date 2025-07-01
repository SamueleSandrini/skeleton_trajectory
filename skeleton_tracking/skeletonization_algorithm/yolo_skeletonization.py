from skeleton_tracking.skeletonization_algorithm.skeletonization_algorithm import (
    BaseSkeletonizationAlgorithm)
from skeleton_tracking.skeleton_interfaces import (
    Skeleton2D, Skeleton3D, Keypoint2D, KeypointID)
from typing import List, Tuple, Type, Dict, Any
import numpy as np
from ultralytics import YOLO

class CocoKeypointID(KeypointID):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

# Coco pose topology
COCO_POSE_CONNECTIONS: List[Tuple[CocoKeypointID, CocoKeypointID]] = [
    (CocoKeypointID.LEFT_EYE, CocoKeypointID.RIGHT_EYE),
    (CocoKeypointID.LEFT_EYE, CocoKeypointID.LEFT_EAR),
    (CocoKeypointID.RIGHT_EYE, CocoKeypointID.RIGHT_EAR),
    (CocoKeypointID.LEFT_SHOULDER, CocoKeypointID.RIGHT_SHOULDER),
    (CocoKeypointID.LEFT_SHOULDER, CocoKeypointID.LEFT_ELBOW),
    (CocoKeypointID.LEFT_ELBOW, CocoKeypointID.LEFT_WRIST),
    (CocoKeypointID.RIGHT_SHOULDER, CocoKeypointID.RIGHT_ELBOW),
    (CocoKeypointID.RIGHT_ELBOW, CocoKeypointID.RIGHT_WRIST),
    (CocoKeypointID.LEFT_SHOULDER, CocoKeypointID.LEFT_HIP),
    (CocoKeypointID.RIGHT_SHOULDER, CocoKeypointID.RIGHT_HIP),
    (CocoKeypointID.LEFT_HIP, CocoKeypointID.RIGHT_HIP),
    (CocoKeypointID.LEFT_HIP, CocoKeypointID.LEFT_KNEE),
    (CocoKeypointID.LEFT_KNEE, CocoKeypointID.LEFT_ANKLE),
    (CocoKeypointID.RIGHT_HIP, CocoKeypointID.RIGHT_KNEE),
    (CocoKeypointID.RIGHT_KNEE, CocoKeypointID.RIGHT_ANKLE),
]

class YoloPoseSkeletonization(BaseSkeletonizationAlgorithm):
    def __init__(self, params: Dict[str, Any]):
        model_path = params.get("model_path", 
                                "yolov8n-pose.pt")
        self.device = params.get("device", "cpu")
        
        self.model = YOLO(model_path, )
        self.results = None

    @staticmethod
    def get_parameters_names() -> List[Tuple[str, Any]]:
        return [("model_path", "yolov8n-pose.pt"), 
                ("device", "cpu")]

    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        self.results = self.model(rgb_image)

        # print(type(results))
        skeletons = []
        # print(f'Lunghezza result {len(results)}')
        for idx, result in enumerate(self.results):
            # print(type(result))
            # dfsafa
            if result.keypoints is not None:
                keypoints_array = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                # print(f"Keypoints Array: {keypoints_array}")
                # print(confidences)
                for skeleton_id, (xy_s, conf_s) in enumerate(zip(keypoints_array, confidences)):
                    keypoints = []
                    for kp_idx, (xy, conf) in enumerate(zip(xy_s, conf_s)):
                        # print(f"XY: {xy}")
                        # print(type(xy))
                        # print(conf)
                        # print(type(conf))
                        keypoints.append(Keypoint2D(
                            id=CocoKeypointID(kp_idx),
                            x=xy[0],
                            y=xy[1],
                            metadata={"confidence": float(conf)}
                        ))
                    # print(f"XY: {xy}")
                    # print(type(xy))
                    # print(conf)
                    # print(type(conf))
                    # keypoints.append(Keypoint2D(
                    #     id=CocoKeypointID(kp_idx),
                    #     x=xy[0],
                    #     y=xy[1],
                    #     metadata={"confidence": float(conf)}
                    # ))

                    skeletons.append(Skeleton2D(id=skeleton_id, keypoints=keypoints))
        return skeletons

    @property
    def topology(self) -> List[Tuple[KeypointID, KeypointID]]:
        return COCO_POSE_CONNECTIONS

    @property
    def keypoint_enum(self) -> Type[KeypointID]:
        return CocoKeypointID

    def is_not_person(self, skeleton: Skeleton3D) -> bool:
        return True

    def get_color_frame_with_detection(self, 
        rgb_image: np.ndarray
    ) -> np.ndarray:
        return self.results[0].plot().copy()
