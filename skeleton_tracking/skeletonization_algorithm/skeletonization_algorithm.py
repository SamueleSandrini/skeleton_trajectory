from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Any
import numpy as np
import importlib

from skeleton_tracking.skeleton_interfaces import Skeleton2D, KeypointID, Skeleton3d

class BaseSkeletonizationAlgorithm(ABC):

    @abstractmethod
    def __init__(self, params: dict):
        pass

    @staticmethod
    @abstractmethod
    def get_parameters_names() -> List[Tuple[str, Any]]:
        pass


    @abstractmethod
    def extract_skeletons(self, rgb_image: np.ndarray) -> List[Skeleton2D]:
        pass

    @abstractmethod
    def is_not_person(self, skeleton: Skeleton3d) -> bool:
        pass

    @property
    @abstractmethod
    def topology(self) -> List[Tuple[KeypointID, KeypointID]]:
        pass

    @property
    @abstractmethod
    def keypoint_enum(self) -> Type[KeypointID]:
        pass

# def load_skeleton_algorithm(module_name: str, class_name: str, params: dict) -> BaseSkeletonizationAlgorithm:
#     """
#     Dynamically loads and instantiates a skeletonization algorithm.

#     :param module_name: Full dotted path of the module (e.g., 'skeleton_tracking.mediapipe_impl')
#     :param class_name: Class name to load (e.g., 'MediaPipeSkeletonization')
#     :param params: Parameters dictionary to initialize the class
#     :return: An instance of BaseSkeletonizationAlgorithm
#     """
#     module = importlib.import_module(module_name)
#     cls = getattr(module, class_name)
#     if not issubclass(cls, BaseSkeletonizationAlgorithm):
#         raise TypeError(f"{class_name} is not a subclass of BaseSkeletonizationAlgorithm")

#     return cls(params)

def load_skeleton_algorithm(module_name: str, class_name: str) -> type:
    """
    Dynamically loads and returns the skeletonization algorithm class.

    :param module_name: Full dotted path of the module (e.g., 'skeleton_tracking.mediapipe_impl')
    :param class_name: Class name to load (e.g., 'MediaPipeSkeletonization')
    :return: The class object of the skeletonization algorithm
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Verifica se la classe è una sottoclasse di BaseSkeletonizationAlgorithm
    if not issubclass(cls, BaseSkeletonizationAlgorithm):
        raise TypeError(f"{class_name} is not a subclass of BaseSkeletonizationAlgorithm")

    return cls
