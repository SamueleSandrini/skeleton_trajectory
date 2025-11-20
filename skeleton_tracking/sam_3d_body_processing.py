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

from cv_bridge import CvBridge
from geometry_msgs.msg import Point

import cv2
import numpy as np
import rclpy
import torch

from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray

from vision_system.post_processing import PostProcessing

from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
from tools.build_detector import HumanDetector


class Sam3DBodyDetection(PostProcessing):

    def __init__(self):
        # Internal node, same pattern as SkeletonDetection
        self.internal_node = rclpy.create_node('sam_3d_body_detection_node')

        # Publisher for 3D joint markers
        self.joints_marker_publisher = self.internal_node.create_publisher(
            MarkerArray, 'sam_3d_body_joints', 10
        )

        # Publisher for 3D mesh (TRIANGLE_LIST marker)
        self.mesh_publisher = self.internal_node.create_publisher(
            Marker, 'sam_3d_body_mesh', 10
        )

        # Publisher for debug image
        self.debug_publisher = self.internal_node.create_publisher(
            Image, 'debug_sam_3d_body_image', 10
        )

        # Parameters
        self.internal_node.declare_parameter('sam_3d_body_node.debug_topic', True)
        self.internal_node.declare_parameter(
            'sam_3d_body_node.model_name', 'facebook/sam-3d-body-dinov3'
        )
        self.internal_node.declare_parameter(
            'sam_3d_body_node.inference_type', 'full'  # full | body | hand
        )
        self.internal_node.declare_parameter(
            'sam_3d_body_node.use_cuda', True
        )
        self.internal_node.declare_parameter(
            'sam_3d_body_node.use_detector', False
        )
        self.internal_node.declare_parameter(
            'sam_3d_body_node.det_bbox_thr', 0.5
        )
        self.internal_node.declare_parameter(
            'sam_3d_body_node.det_nms_thr', 0.3
        )


        self.debug = self.internal_node.get_parameter(
            'sam_3d_body_node.debug_topic').value
        self.model_name = self.internal_node.get_parameter(
            'sam_3d_body_node.model_name').value
        self.inference_type = self.internal_node.get_parameter(
            'sam_3d_body_node.inference_type').value
        self.use_cuda = self.internal_node.get_parameter(
            'sam_3d_body_node.use_cuda').value
        self.use_detector = self.internal_node.get_parameter(
            'sam_3d_body_node.use_detector').value
        self.det_bbox_thr = float(self.internal_node.get_parameter(
            'sam_3d_body_node.det_bbox_thr').value)
        self.det_nms_thr = float(self.internal_node.get_parameter(
            'sam_3d_body_node.det_nms_thr').value)

        # debug here
        self.internal_node.get_logger().info(f'Debug mode: {self.debug}')

        self.camera_info: CameraInfo = None
        self.cam_int = None  # Camera intrinsics as torch.Tensor

        self.cv_bridge = CvBridge()

        # Load model
        self.internal_node.get_logger().info(
            f'Loading SAM-3D-Body model: {self.model_name} ...'
        )
        self.internal_node.get_logger().info(f'pre load model_name: {self.model_name}')

        model, model_cfg = load_sam_3d_body_hf(self.model_name)
        self.internal_node.get_logger().info(f'post load model_name: {self.model_name}')

        # SAM3DBodyEstimator, based on the implementation you provided
        if self.use_cuda and torch.cuda.is_available():
            model = model.to('cuda')
        else:
            self.internal_node.get_logger().warn(
                'CUDA not available or disabled. '
                'SAM3DBodyEstimator, as currently implemented, always moves '
                'the batch to "cuda". You may need to modify the estimator '
                'to support pure CPU execution (replace "cuda" with the '
                'estimator device).'
            )
        self.internal_node.get_logger().info(f'pre estimator: {self.model_name}')
        
        human_detector = None
        if self.use_detector:
            det_device = 'cuda' if (self.use_cuda and torch.cuda.is_available()) else 'cpu'
            try:
                self.internal_node.get_logger().info(
                    f'Loading ViTDet HumanDetector on device: {det_device}'
                )
                human_detector = HumanDetector(
                    name="vitdet",
                    device=det_device,
                    # se usi il parametro "path" per i pesi locali:
                    # path="/path/dove/hai/salvato/model_final_f05665.pkl"
                )
                self.internal_node.get_logger().info('ViTDet human detector loaded.')
            except Exception as e:
                self.internal_node.get_logger().error(
                    f'Failed to initialize ViTDet human detector: {e}. '
                    'Proceeding without detector.'
                )
                human_detector = None

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=None,
            fov_estimator=None,
        )

        self.internal_node.get_logger().info('SAM-3D-Body model loaded.')

    def initialize(self, camera_info: CameraInfo):
        """Initialize with camera intrinsics (same pattern as SkeletonDetection)."""
        self.camera_info = camera_info
        self.cam_int = self._build_cam_int(camera_info)
        self.internal_node.get_logger().info('SAM-3D-Body detection running ...')

    def _build_cam_int(self, camera_info: CameraInfo):
        """
        Build a 3x3 camera intrinsic matrix from CameraInfo.

        SAM3DBodyEstimator expects a torch.Tensor that is then moved
        to the batch device. Here we create [1, 3, 3] so it matches
        the expected shape.
        """
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        cam_int_np = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32
        )

        device = self.estimator.device
        cam_int = torch.tensor(cam_int_np, device=device).unsqueeze(0)  # [1, 3, 3]
        return cam_int

    def _create_joint_marker(self, marker_id, x, y, z, person_id=0):
        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.internal_node.get_clock().now().to_msg()

        marker.ns = f'sam3d_joints_person_{person_id}'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)

        return marker

    def _create_mesh_marker(self, vertices, person_id=0):
        """
        Create a TRIANGLE_LIST marker using vertices (V, 3) and
        self.estimator.faces.

        NOTE: this can be heavy (lots of triangles). Depending on
        your use case, you may want to publish this less frequently
        or downsample the mesh.
        """
        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.internal_node.get_clock().now().to_msg()

        marker.ns = 'sam3d_mesh'
        marker.id = person_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # For TRIANGLE_LIST, scale is typically 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5  # semi-transparent

        faces = self.estimator.faces  # (F, 3)

        for f in faces:
            v1 = vertices[f[0]]
            v2 = vertices[f[1]]
            v3 = vertices[f[2]]

            p1 = Point(x=float(v1[0]), y=float(v1[1]), z=float(v1[2]))
            p2 = Point(x=float(v2[0]), y=float(v2[1]), z=float(v2[2]))
            p3 = Point(x=float(v3[0]), y=float(v3[1]), z=float(v3[2]))

            marker.points.extend([p1, p2, p3])

        return marker

    def process_frames(self, color_frame, distance_frame):
        """
        Interface compatible with SkeletonDetection.

        Args:
            color_frame: np.ndarray, assumed to be BGR (as in OpenCV).
                         If your pipeline already provides RGB, remove the cvtColor call.
            distance_frame: np.ndarray with depth; not used here, as SAM-3D-Body is monocular.
        """
        if self.camera_info is None:
            self.internal_node.get_logger().warn(
                'CameraInfo not initialized. Call initialize() before process_frames.'
            )
            return

        if color_frame is None:
            return

        # SAM3DBodyEstimator expects RGB when an np.ndarray is passed.
        # If color_frame is already RGB, comment out the line below.
        # img_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        img_rgb = color_frame
        try:
            outputs = self.estimator.process_one_image(
                            img_rgb,
                            cam_int=self.cam_int,
                            inference_type=self.inference_type,
                            bbox_thr=self.det_bbox_thr,
                            nms_thr=self.det_nms_thr,
                        )
        except Exception as e:
            self.internal_node.get_logger().error(
                f'SAM-3D-Body inference failed: {e}'
            )
            return

        if outputs is None or len(outputs) == 0:
            # No humans detected
            return

        joints_marker_array = MarkerArray()

        # Debug image
        debug_img = color_frame.copy()

        for person_id, person_out in enumerate(outputs):
            # Bounding box (x1, y1, x2, y2) in pixels
            bbox = person_out["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            if self.debug:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 3D joints:
            # Prefer "pred_joint_coords" if available, otherwise fall back to "pred_keypoints_3d"
            if person_out.get("pred_joint_coords") is not None:
                joints_3d = person_out["pred_joint_coords"]
            else:
                joints_3d = person_out["pred_keypoints_3d"]

            # Create a marker for each joint
            for j_id, joint in enumerate(joints_3d):
                x, y, z = joint
                marker = self._create_joint_marker(
                    marker_id=person_id * 1000 + j_id,
                    x=x, y=y, z=z,
                    person_id=person_id
                )
                joints_marker_array.markers.append(marker)

            # Full 3D mesh
            vertices = person_out["pred_vertices"]
            mesh_marker = self._create_mesh_marker(vertices, person_id=person_id)
            self.mesh_publisher.publish(mesh_marker)

        # Publish 3D joint markers
        self.joints_marker_publisher.publish(joints_marker_array)

        # Publish debug image with bounding boxes
        if self.debug:
            self.debug_publisher.publish(
                self.cv_bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            )

    def process_frame(self, color_frame):
        """
        If your framework also calls a single-frame version (without depth),
        just delegate to process_frames and ignore the distance frame.
        """
        self.process_frames(color_frame, None)
