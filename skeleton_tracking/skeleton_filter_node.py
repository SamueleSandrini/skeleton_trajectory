
# Copyright 2024 National Research Council STIIMA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, TwistStamped, Transform, Vector3, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
# from KalmanFilter import KalmanFilter
import importlib
from skeleton_tracking.skeleton_traking_utils import build_skeleton_topology_msg, build_marker_array_msg
from std_msgs.msg import Header
from skeleton_tracking.kalman_filter import KalmanFilter
N_KEYPOINTS = 33


# def load_custom_publishers():
        #     sampler_module_name = self.get_parameter('sampling_stategy_module').get_parameter_value().string_value
        # module_name = sampler_module_name.rsplit('.', 1)[0] 
        # self.get_logger().info(f"Loading sampling strategy module: {module_name}")
        # class_name = sampler_module_name.rsplit('.', 1)[-1]
        # self.get_logger().info(f"Loading sampling strategy class: {class_name}")
  # try:
  #     # Import the module dynamically
  #     publisher_module = importlib.import_module(module_name)
  #     # Retrieve the class 
  #     publisher_class = getattr(publisher_module, class_name)
      
  #     # Check if the loaded class is a subclass of CustomPublisherBaseClass
  #     if not issubclass(publisher_class, CustomPublisherBaseClass):
  #         raise TypeError(f"The sampler class {publisher_class.__name__} is not a subclass of CustomPublisherBaseClass.")
      
  #     sampler_instance = sampler_class()

  #     parameter_names = sampler_instance.get_parameters_names()
  #     sampler_params = {}
  #     for (param_name, param_default) in parameter_names:
  #         self.declare_parameter(param_name, param_default)
  #         sampler_params[param_name] = self.get_parameter(param_name).get_parameter_value().double_value

  #     # Return the instantiated sampler with the retrieved parameters
  #     return sampler_instance, sampler_params

class SkeletonFilterNode(Node):
    def __init__(self):
        super().__init__('skeleton_filter_node')

        # Parameters
        self.declare_parameter('custom_publishers', [])
        self.declare_parameter('keypoint_timeout', 0.1)

        self.declare_parameter('q_noise', [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        self.declare_parameter('r_noise', [0.05, 0.05, 0.1])
        self.declare_parameter('dt', 1.0 / 30.0)

        custom_publishers = self.get_parameter('custom_publishers').value
        self.keypoint_timeout = self.get_parameter('keypoint_timeout').value

        # for publisher in custom_publishers:
        #   self.declare_parameter(f'{publisher}.package')
        #   self.declare_parameter(f'{publisher}.module')
        #   self.declare_parameter(f'{publisher}.class')
        #   package_name = self.get_parameter(f'{publisher}.package').get_parameter_value().string_value
        #   module_name = self.get_parameter(f'{publisher}.module').get_parameter_value().string_value
        #   class_name = self.get_parameter(f'{publisher}.class').get_parameter_value().string_value

        # Subscriber
        self.subscriber_keypoints = self.create_subscription(MarkerArray, 'skeleton_markers', self.callback_keypoint, 10)


        # # Init filters
        q_noise = self.get_parameter('q_noise').value
        r_noise = self.get_parameter('r_noise').value
        dt = self.get_parameter('dt').value
        self.keypoints_filters = [KalmanFilter(q_noise,
                                               r_noise,
                                               dt) for _ in range(N_KEYPOINTS)]
        self.time_keypoint = np.zeros((N_KEYPOINTS,))
        
        # Publisher
        self.skeleton_marker_publisher = self.create_publisher(MarkerArray, 'skeleton_markers_filtered', 10)
        self.skeleton_topology_publisher = self.create_publisher(Marker, 'skeleton_filtered', 10)

        # self.pub_marker_filtered = self.create_publisher(Marker, f'{self.camera_ns}/keypoints_filtered', 10)
        # self.pub_marker_variance = self.create_publisher(Marker, f'{self.camera_ns}/marker_variance', 10)
        # self.pub_skeleton_filtered = self.create_publisher(Marker, f'{self.camera_ns}/skeleton_filtered', 10)
        # self.pub_skeleton_filtered_array = self.create_publisher(PoseArray, f'{self.camera_ns}/poses', 10)
        # self.pub_centroid_array = self.create_publisher(PoseArray, f'{self.camera_ns}/centroids', 10)
        # self.pub_keypoint_velocity = self.create_publisher(TwistStamped, f'{self.camera_ns}/keypoint_velocity', 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # State variables
        self.list_of_indexes_pres = []

        self.get_logger().info('SkeletonFilterNode initialized.')

    def callback_keypoint(self, keypoints_msg: MarkerArray):
        if not keypoints_msg.markers:
            return
        current_time_msg = self.get_clock().now().to_msg()
        current_time = current_time_msg.sec + current_time_msg.nanosec * 1e-9

        frame_id = keypoints_msg.markers[0].header.frame_id

        keypoint_list = []
        list_of_indexes_pres = []

        for keypoint in keypoints_msg.markers:
          id_kp = keypoint.id
          keypoint_position = np.array([keypoint.pose.position.x, 
                                        keypoint.pose.position.y, 
                                        keypoint.pose.position.z])
          # Kalman filter logic
          if current_time - self.time_keypoint[id_kp] > self.keypoint_timeout: # keypoint lifetime is passed
            self.keypoints_filters[id_kp].initialize(keypoint_position)
          else:
            self.keypoints_filters[id_kp].update(keypoint_position, id_kp)

          keypoint_filtered_position = self.keypoints_filters[id_kp].get_output_estimate()
        
          keypoint_list.append(Pose(position=Point(x = keypoint_filtered_position[0], 
                                                   y = keypoint_filtered_position[1], 
                                                   z = keypoint_filtered_position[2])))
          list_of_indexes_pres.append(id_kp)
          self.time_keypoint[id_kp] = current_time

        # Open loop update for undetected keypoints
        for id_kp in set(range(N_KEYPOINTS)) - set(list_of_indexes_pres):
          if current_time - self.time_keypoint[id_kp] < self.keypoint_timeout:
            keypoint_nominal_position = self.keypoints_filters[id_kp].open_loop_update()
            keypoint_list.append(Pose(position=Point(x=keypoint_nominal_position[0], 
                                                     y=keypoint_nominal_position[1], 
                                                     z=keypoint_nominal_position[2])))
            list_of_indexes_pres.append(id_kp)
            self.get_logger().info(f"Keypoint: {id_kp} updated in open loop")

        # Publish info
        header = Header(frame_id=frame_id, stamp=current_time_msg)
        
        skeleton = build_skeleton_topology_msg(keypoint_list, 
                                               list_of_indexes_pres,
                                               header)
        self.skeleton_topology_publisher.publish(skeleton)
        
        marker_array = build_marker_array_msg(keypoint_list, 
                                              list_of_indexes_pres, 
                                              header)
        self.skeleton_marker_publisher.publish(marker_array)

                                                                 
        # skeleton = Marker()
        # skeleton.header.stamp = current_time
        # skeleton.header.frame_id = frame_id
        # skeleton.ns = "SegmentsFiltered"
        # skeleton.type = Marker.LINE_LIST
        # skeleton.scale.x = 0.03
        # skeleton.color.r = 0.0
        # skeleton.color.g = 1.0
        # skeleton.color.b = 0.0
        # skeleton.color.a = 0.7

        # self.publish_keypoint(id_kp, y_filtered, frame_id)
        # self.publish_velocity(id_kp)
        # self.publish_variance(id_kp, y_filtered, frame_id)

        # self.publish_pose_array(frame_id)
        # self.publish_centroid(frame_id)
        # self.publish_skeleton_lines(skeleton)
        # self.pub_skeleton_filtered.publish(skeleton)

    # def publish_keypoint(self, idx, position, frame_id):
    #     marker = Marker()
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.header.frame_id = frame_id
    #     marker.ns = "KeypointsFiltered"
    #     marker.id = idx
    #     marker.type = Marker.SPHERE
    #     marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
    #     marker.pose.orientation.w = 1.0
    #     marker.scale.x = marker.scale.y = marker.scale.z = 0.04
    #     marker.color.g = 1.0
    #     marker.color.a = 1.0
    #     self.pub_marker_filtered.publish(marker)

    # def publish_velocity(self, idx):
    #     vel = self.keypoints_filters[idx].getCartesianVelocity()
    #     msg = TwistStamped()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = f'KeypointFrame{idx}'
    #     msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vel
    #     self.pub_keypoint_velocity.publish(msg)

    #     # Broadcast TF
    #     self.broadcast_tf(np.eye(3), vel, 'camera_frame', msg.header.frame_id)

    # def publish_variance(self, idx, position, frame_id):
    #     dev = self.keypoints_filters[idx].getPosDevSt()
    #     marker = Marker()
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.header.frame_id = frame_id
    #     marker.ns = "KeypointsFilteredVariance"
    #     marker.id = idx
    #     marker.type = Marker.SPHERE
    #     marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
    #     marker.pose.orientation = Quaternion(x=dev[0], y=dev[1], z=dev[2], w=1.0)
    #     self.pub_marker_variance.publish(marker)

    # def publish_pose_array(self, frame_id):
    #     array_msg = PoseArray()
    #     array_msg.header.stamp = self.get_clock().now().to_msg()
    #     array_msg.header.frame_id = frame_id
    #     array_msg.poses = self.list_keypoints
    #     self.pub_skeleton_filtered_array.publish(array_msg)

    # def publish_centroid(self, frame_id):
    #     if not self.list_keypoints:
    #         return
    #     xs = [pose.position.x for pose in self.list_keypoints]
    #     ys = [pose.position.y for pose in self.list_keypoints]
    #     zs = [pose.position.z for pose in self.list_keypoints]
    #     centroid = Pose(position=Point(x=np.mean(xs), y=np.mean(ys), z=np.mean(zs)))
    #     centroid_array = PoseArray()
    #     centroid_array.header.stamp = self.get_clock().now().to_msg()
    #     centroid_array.header.frame_id = frame_id
    #     centroid_array.poses.append(centroid)
    #     self.pub_centroid_array.publish(centroid_array)

    # def publish_skeleton_lines(self, skeleton):
    #     for start_idx, end_idx in self.mp_pose.POSE_CONNECTIONS:
    #         if start_idx in self.list_of_indexes_pres and end_idx in self.list_of_indexes_pres:
    #             start_pose = self.list_keypoints[self.list_of_indexes_pres.index(start_idx)]
    #             end_pose = self.list_keypoints[self.list_of_indexes_pres.index(end_idx)]
    #             skeleton.points.append(start_pose.position)
    #             skeleton.points.append(end_pose.position)

    # def broadcast_tf(self, matR, tran, origin_frame, child_frame):
    #     quat = R.from_matrix(matR).as_quat()
    #     transform = TransformStamped()
    #     transform.header.stamp = self.get_clock().now().to_msg()
    #     transform.header.frame_id = origin_frame
    #     transform.child_frame_id = child_frame
    #     transform.transform.translation = Vector3(x=tran[0], y=tran[1], z=tran[2])
    #     transform.transform.rotation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    #     self.tf_broadcaster.sendTransform(transform)

def main(args=None):
  rclpy.init(args=args)
  node = SkeletonFilterNode()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()