from skeleton_tracking.skeletonization_algorithm import (
    BaseSkeletonizationAlgorithm, Skeleton2D, Keypoint2D, KeypointID
)
from typing import List, Tuple, Type, Dict, Any
from enum import IntEnum
import numpy as np
from ultralytics import YOLO

# Definizione dell'enumerazione dei keypoint basata su COCO
class YoloPoseKeypointID(KeypointID):
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

# Definizione della topologia dei keypoint (connessioni tra i punti)
YOLO_POSE_CONNECTIONS: List[Tuple[YoloPoseKeypointID, YoloPoseKeypointID]] = [
    (YoloPoseKeypointID.LEFT_EYE, YoloPoseKeypointID.RIGHT_EYE),
    (YoloPoseKeypointID.LEFT_EYE, YoloPoseKeypointID.LEFT_EAR),
    (YoloPoseKeypointID.RIGHT_EYE, YoloPoseKeypointID.RIGHT_EAR),
    (YoloPoseKeypointID.LEFT_SHOULDER, YoloPoseKeypointID.RIGHT_SHOULDER),
    (YoloPoseKeypointID.LEFT_SHOULDER, YoloPoseKeypointID.LEFT_ELBOW),
    (YoloPoseKeypointID.LEFT_ELBOW, YoloPoseKeypointID.LEFT_WRIST),
    (YoloPoseKeypointID.RIGHT_SHOULDER, YoloPoseKeypointID.RIGHT_ELBOW),
    (YoloPoseKeypointID.RIGHT_ELBOW, YoloPoseKeypointID.RIGHT_WRIST),
    (YoloPoseKeypointID.LEFT_SHOULDER, YoloPoseKeypointID.LEFT_HIP),
    (YoloPoseKeypointID.RIGHT_SHOULDER, YoloPoseKeypointID.RIGHT_HIP),
    (YoloPoseKeypointID.LEFT_HIP, YoloPoseKeypointID.RIGHT_HIP),
    (YoloPoseKeypointID.LEFT_HIP, YoloPoseKeypointID.LEFT_KNEE),
    (YoloPoseKeypointID.LEFT_KNEE, YoloPoseKeypointID.LEFT_ANKLE),
    (YoloPoseKeypointID.RIGHT_HIP, YoloPoseKeypointID.RIGHT_KNEE),
    (YoloPoseKeypointID.RIGHT_KNEE, YoloPoseKeypointID.RIGHT_ANKLE),
]

class YoloPoseSkeletonization(BaseSkeletonizationAlgorithm):
    def __init__(self, params: Dict[str, Any]):
        model_path = params.get("model_path", "yolov8n-pose.pt")
        self.model = YOLO(model_path)

    @staticmethod
    def get_parameters_names() -> List[str]:
        return ["model_path"]

    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        results = self.model(rgb_image)
        skeletons = []

        for idx, result in enumerate(results):
            keypoints = []
            if result.keypoints is not None:
                keypoints_array = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()

                for kp_idx, (xy, conf) in enumerate(zip(keypoints_array, confidences)):
                    keypoints.append(Keypoint2D(
                        id=YoloPoseKeypointID(kp_idx),
                        x=xy[0],
                        y=xy[1],
                        metadata={"confidence": float(conf)}
                    ))

                skeletons.append(Skeleton2D(id=idx, keypoints=keypoints))

        return skeletons

    @property
    def topology(self) -> List[Tuple[KeypointID, KeypointID]]:
        return YOLO_POSE_CONNECTIONS

    @property
    def keypoint_enum(self) -> Type[KeypointID]:
        return YoloPoseKeypointID
