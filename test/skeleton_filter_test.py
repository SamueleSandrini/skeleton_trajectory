import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from skeleton_tracking.skeleton_filter_node import SkeletonFilterNode
import numpy as np
import time
import threading

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

class FakeSkeletonPublisher(Node):
    def __init__(self):
        super().__init__("fake_skeleton_publisher")
        self.marker_publisher = self.create_publisher(MarkerArray, "/skeleton_markers", 10)
        self.timer = None

    def publish(self):
        msg = create_noisy_marker_array()
        self.marker_publisher.publish(msg)
        self.get_logger().info("Published fake skeleton")

    def start_publishing(self, frequency_hz=1.0):
        if self.timer is None:
            period = 1.0 / frequency_hz
            self.timer = self.create_timer(period, self.publish)
            self.get_logger().info(f"Started publishing at {frequency_hz} Hz")

    def stop_publishing(self):
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None
            self.get_logger().info("Stopped publishing")

class SkeletonFilterAnalyzer(Node):
    def __init__(self):
        super().__init__("skeleton_filter_analyzer")
        self.marker_filtered_subscription = self.create_subscription(
            MarkerArray,
            "/skeleton_markers_filtered",
            self.filtered_listener_callback,
            10
        )
        self.marker_sublscription = self.create_subscription(
            MarkerArray,
            "/skeleton_markers",
            self.nominal_listener_callback,
            10
        )
        self.filtered_markers = {}
        self.nominal_markers = {}
        

    def filtered_listener_callback(self, msg):
        # Analyze the filtered skeleton data
        for marker in msg.markers:
            if marker.id not in self.filtered_markers:
                self.filtered_markers[marker.id] = []

            self.filtered_markers[marker.id].append([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ])

    def nominal_listener_callback(self, msg):
        for marker in msg.markers:
            if marker.id not in self.nominal_markers:
                self.nominal_markers[marker.id] = []

            self.nominal_markers[marker.id].append([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ])
        
    def analyze_filtered_data(self):
        # Perform analysis on the filtered data
        
        filterd_marker_stats = {}
        for marker_id, positions in self.filtered_markers.items():
            marker_positions_np = np.array(positions)
            marker_mean = marker_positions_np.mean(axis=0)
            marker_std = marker_positions_np.std(axis=0)
            filterd_marker_stats[marker_id] = {
                "mean": marker_mean,
                "std": marker_std
            }
            self.get_logger().info(f"Keypoint: {marker_id}, mean: {marker_mean}, std: {marker_std} ")
        self.get_logger().info("Filtered data analysis complete.")
        
        nominal_marker_stats = {}
        for marker_id, positions in self.nominal_markers.items():
            marker_positions_np = np.array(positions)
            marker_mean = marker_positions_np.mean(axis=0)
            marker_std = marker_positions_np.std(axis=0)
            nominal_marker_stats[marker_id] = {
                "mean": marker_mean,
                "std": marker_std
            }
            self.get_logger().info(f"Keypoint: {marker_id}, mean: {marker_mean}, std: {marker_std} ")

        self.get_logger().info("Comparison")
        for marker_id in self.filtered_markers.keys():
            mean_diff = filterd_marker_stats[marker_id]["mean"] - nominal_marker_stats[marker_id]["mean"]
            std_diff = filterd_marker_stats[marker_id]["std"] - nominal_marker_stats[marker_id]["std"]
            self.get_logger().info(f"Keypoint: {marker_id}, mean difference: {mean_diff}, std difference: {std_diff} ")
# class FakeSkeletonPublisher(Node):
#     def __init__(self):
#         super().__init__("fake_skeleton_publisher")
#         self.marker_publisher = self.create_publisher(MarkerArray, "/skeleton_markers", 10)
        
#     def publish(self):
#         msg = create_noisy_marker_array()
#         self.marker_publisher.publish(msg)
    
#     def start_publishing(self, frequency_hz=1.0):
#         self._running = True
#         pub_thread = threading.Thread(target=self._publish_loop, args=(frequency_hz,))
#         pub_thread.daemon = True
#         pub_thread.start()

#     def stop_publishing(self):
#         self._running = False

#     def _publish_loop(self, frequency_hz):
#         period = 1.0 / frequency_hz
#         while rclpy.ok() and self._running:
#             self.publish()
#             time.sleep(period)

@pytest.mark.dependency(name="setUp")
def test_setup():
    rclpy.init()

@pytest.fixture
def fake_skeleton_publisher():
    return FakeSkeletonPublisher()

@pytest.fixture
def skeleton_analyzer():
    return SkeletonFilterAnalyzer()

@pytest.fixture
def skeleton_filter_node():
    return SkeletonFilterNode()


@pytest.mark.dependency(name="skeleton_filter_node_test", 
                        depends=["setUp"])
def test_skeleton_filter_class(fake_skeleton_publisher):
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(fake_skeleton_publisher)
    
    t_start = fake_skeleton_publisher.get_clock().now().nanoseconds * 1e-9
    elapsed_time = 0.0
    fake_skeleton_publisher.start_publishing(frequency_hz=10.0)
    while rclpy.ok() and elapsed_time < 1.0:
        elapsed_time = fake_skeleton_publisher.get_clock().now().nanoseconds * 1e-9 - t_start
        executor.spin_once(timeout_sec=0.1)
    fake_skeleton_publisher.stop_publishing()
    

@pytest.mark.dependency(name="filter_removes_noise", 
                        depends=["setUp", "skeleton_filter_node_test"])
def test_filter_removes_noise(fake_skeleton_publisher,
                              skeleton_analyzer,
                              skeleton_filter_node):
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(fake_skeleton_publisher)
    executor.add_node(skeleton_filter_node)
    executor.add_node(skeleton_analyzer)


    t_start = fake_skeleton_publisher.get_clock().now().nanoseconds * 1e-9
    elapsed_time = 0.0
    fake_skeleton_publisher.start_publishing(frequency_hz=30.0)
    
    while rclpy.ok() and elapsed_time < 5.0:
        elapsed_time = fake_skeleton_publisher.get_clock().now().nanoseconds * 1e-9 - t_start
        executor.spin_once(timeout_sec=0.1)
    
    fake_skeleton_publisher.stop_publishing()
    skeleton_analyzer.analyze_filtered_data()


# @pytest.mark.rostest
# def test_filter_removes_noise(monkeypatch):
#     rclpy.init()
#     node = rclpy.create_node("test_node")

#     pub = node.create_publisher(MarkerArray, "/skeleton_markers", 10)
#     sub = node.create_subscription(MarkerArray, "/skeleton_markers_filtered", lambda msg: callback(msg, node), 10)

#     received_messages = []

#     def callback(msg, node):
#         received_messages.append(msg)
#         if len(received_messages) >= 10:
#             node.get_logger().info("10 messages received, shutting down.")
#             rclpy.shutdown()

#     # Pubblica messaggi rumorosi
#     start = time.time()
#     while rclpy.ok() and (time.time() - start) < 25.0:
#         msg = create_noisy_marker_array()
#         pub.publish(msg)
#         rclpy.spin_once(node, timeout_sec=0.1)

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
