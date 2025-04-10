import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import time

N_KEYPOINTS = 33
FRAME_ID = "base_link"

def get_nominal_skeleton():
    # Define nominal 3D positions for each keypoint to form a humanoid mannequin
    # All z values are set to 0 to place the skeleton on the XY plane

    skeleton = {
        0: np.array([0.0, 1.7, 0.0]),   # head
        1: np.array([0.0, 1.5, 0.0]),   # neck
        2: np.array([-0.2, 1.5, 0.0]),  # left shoulder
        3: np.array([-0.5, 1.3, 0.0]),  # left elbow
        4: np.array([-0.7, 1.1, 0.0]),  # left hand
        5: np.array([0.2, 1.5, 0.0]),   # right shoulder
        6: np.array([0.5, 1.3, 0.0]),   # right elbow
        7: np.array([0.7, 1.1, 0.0]),   # right hand
        8: np.array([0.0, 1.2, 0.0]),   # upper torso
        9: np.array([0.0, 1.0, 0.0]),   # lower torso
        10: np.array([-0.2, 0.8, 0.0]), # left hip
        11: np.array([-0.2, 0.4, 0.0]), # left knee
        12: np.array([-0.2, 0.0, 0.0]), # left foot
        13: np.array([0.2, 0.8, 0.0]),  # right hip
        14: np.array([0.2, 0.4, 0.0]),  # right knee
        15: np.array([0.2, 0.0, 0.0]),  # right foot
    }

    # For the remaining keypoints (to reach 33), assign a default position on the torso
    for i in range(16, 33):
        skeleton[i] = np.array([0.0, 1.0, 0.0])

    return skeleton

def create_noisy_marker_array(n_keypoints=33, noise_std=0.03):
    markers = []
    timestamp = rclpy.time.Time().to_msg()
    frame_id = 'base_link'

    nominal_positions = get_nominal_skeleton()

    for i in range(n_keypoints):
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Add Gaussian noise around the nominal position
        nominal = nominal_positions[i]
        noisy_position = nominal + np.random.normal(0, noise_std, size=3)
        marker.pose.position.x = noisy_position[0]
        marker.pose.position.y = noisy_position[1]
        marker.pose.position.z = noisy_position[2]

        markers.append(marker)

    marker_array = MarkerArray(markers=markers)
    return marker_array


@pytest.mark.rostest
def test_filter_removes_noise(monkeypatch):
    rclpy.init()
    node = rclpy.create_node("test_node")

    pub = node.create_publisher(MarkerArray, "/skeleton_markers", 10)
    sub = node.create_subscription(MarkerArray, "/skeleton_markers_filtered", lambda msg: callback(msg, node), 10)

    received_messages = []

    def callback(msg, node):
        received_messages.append(msg)
        if len(received_messages) >= 10:
            node.get_logger().info("10 messages received, shutting down.")
            rclpy.shutdown()

    # Pubblica messaggi rumorosi
    start = time.time()
    while rclpy.ok() and (time.time() - start) < 25.0:
        msg = create_noisy_marker_array()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)

    # rclpy.spin(node)  # attende che il callback riceva abbastanza dati

    # # Analisi: per ogni keypoint, calcola la varianza della posizione filtrata
    # all_positions = {i: [] for i in range(N_KEYPOINTS)}
    # for msg in received_messages:
    #     for marker in msg.markers:
    #         all_positions[marker.id].append([
    #             marker.pose.position.x,
    #             marker.pose.position.y,
    #             marker.pose.position.z
    #         ])

    # # Calcola varianza dei keypoint filtrati
    # for kp_id, positions in all_positions.items():
    #     if len(positions) > 1:
    #         positions_np = np.array(positions)
    #         var = np.var(positions_np, axis=0)
    #         print(f"Keypoint {kp_id} filtered variance: {var}")
    #         assert np.all(var < 0.05), f"Filtered variance too high for keypoint {kp_id}: {var}"

    # node.destroy_node()
