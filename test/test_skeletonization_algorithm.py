import pytest
# import numpy as np
import cv2
import os

from skeleton_tracking.skeletonization_algorithm.mediapipe_skeletonization import MediaPipeSkeletonization
from skeleton_tracking.skeletonization_algorithm.yolo_skeletonization import YoloPoseSkeletonization
from skeleton_tracking.skeleton_interfaces import Skeleton2D, Keypoint2D

current_dir = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_PATH = os.path.join(current_dir, 'test_image.jpg')  


@pytest.fixture(scope="module")
def test_image():
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")
    img = cv2.imread(TEST_IMAGE_PATH)
    assert img is not None, "cv2.imread failed to load image"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def test_mediapipe_extract_skeletons(test_image):
    skeletonization_alg = MediaPipeSkeletonization()
    skeletonization_alg.initialize(params={"min_detection_confidence": 0.3})
    skeletons = skeletonization_alg.extract_skeletons(test_image)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 1, "No skeletons detected"
    for skel in skeletons:
        assert isinstance(skel, Skeleton2D)
        for kp in skel.keypoints:
            assert isinstance(kp, Keypoint2D)
            assert 0 <= kp.x <= 1
            assert 0 <= kp.y <= 1
            assert "visibility" in kp.metadata
            assert 0 <= kp.metadata["visibility"] <= 1


def test_mediapipe_topology_and_enum():
    skeletonization_alg = MediaPipeSkeletonization()
    skeletonization_alg.initialize(params={})
    topology = skeletonization_alg.topology
    # enum_class = algo.keypoint_enum

    assert isinstance(topology, list)
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in topology)
    # assert "NOSE" in enum_class.__members__


def test_yolo_extract_skeletons(test_image):
    skeletonization_alg = YoloPoseSkeletonization(params={
        "model_path": "yolov8n-pose.pt",  # Ensure the model is available
        "device": "cpu"
    })
    skeletons = skeletonization_alg.extract_skeletons(test_image)

    assert isinstance(skeletons, list)
    assert len(skeletons) == 1, "No skeletons detected"
    for skel in skeletons:
        assert isinstance(skel, Skeleton2D)
        for kp in skel.keypoints:
            assert isinstance(kp, Keypoint2D)
            assert "confidence" in kp.metadata

def test_yolo_extract_multipe_skeletons():
    skeletonization_alg = YoloPoseSkeletonization(params={
        "model_path": "yolov8n-pose.pt",  # Ensure the model is available
        "device": "cpu"
    })
    skeletons = skeletonization_alg.extract_skeletons("https://ultralytics.com/images/bus.jpg")
    
    assert isinstance(skeletons, list)
    assert len(skeletons) == 4, "No skeletons detected"
    for skel in skeletons:
        assert isinstance(skel, Skeleton2D)
        for kp in skel.keypoints:
            assert isinstance(kp, Keypoint2D)
            assert "confidence" in kp.metadata
            # print(f"Skeleton ID: {kp.id}, Keypoints: x,y,z: {[(kp.x, kp.y, kp.metadata.get('confidence', 'N/A')) for kp in skel.keypoints]}")

def test_yolo_topology_and_enum():
    algo = YoloPoseSkeletonization({})
    topology = algo.topology
    enum_class = algo.keypoint_enum

    assert isinstance(topology, list)
    assert all(isinstance(a.value, int) and isinstance(b.value, int) for a, b in topology)
    assert "NOSE" in enum_class.__members__


def test_parameter_names_consistency():
    mp_params = MediaPipeSkeletonization.get_parameters_names()
    yolo_params = YoloPoseSkeletonization.get_parameters_names()

    assert "min_detection_confidence" in mp_params
    assert "min_tracking_confidence" in mp_params
    assert "model_path" in yolo_params
    assert "device" in yolo_params
