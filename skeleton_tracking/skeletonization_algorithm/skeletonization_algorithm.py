from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Type, Dict, Any
from enum import IntEnum
import numpy as np


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


class BaseSkeletonizationAlgorithm(ABC):

    @abstractmethod
    def __init__(self, params: dict):
        pass

    @staticmethod
    @abstractmethod
    def get_parameters_names() -> List[str]:
        pass


    @abstractmethod
    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        pass

    @property
    @abstractmethod
    def topology(self) -> List[Tuple[KeypointID, KeypointID]]:
        pass

    @property
    @abstractmethod
    def keypoint_enum(self) -> Type[KeypointID]:
        pass
