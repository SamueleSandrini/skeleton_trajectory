from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any


class KeypointID(IntEnum):
    """
    Base class for keypoint IDs. Concrete skeleton models should define their own.
    """
    pass


@dataclass
class Keypoint2D:
    id: IntEnum         # Enum representing the keypoint (e.g., HEAD, SHOULDER)
    x: float            # X coordinate
    y: float            # Y coordinate
    # score: float        # Detection confidence
    metadata: Dict[str, Any] = field(default_factory=dict)  # Custom field

@dataclass
class Skeleton2D:
    id: int                         # Skeleton ID (e.g., person ID)
    keypoints: List[Keypoint2D]     # List of 2D keypoints


@dataclass
class Keypoint3D:
    id: KeypointID  # or will be better to put skeleton2d
    x: float
    y: float
    z: float

@dataclass
class Skeleton3D:
    id: KeypointID
    frame_id: str                         # e.g., "camera_link" or "map"
    keypoints: List[Keypoint3D]

    def append_keypoint(self, keypoint: Keypoint3D):
        """
        Append a keypoint to the skeleton.
        """
        self.keypoints.append(keypoint)