
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

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import mediapipe as mp


def build_skeleton_topology_msg(keypoints, header):
  skeleton = Marker()
  skeleton.header = header
  skeleton.ns = 'skeleton_segments'
  skeleton.type = Marker.LINE_LIST
  skeleton.action = Marker.ADD
  skeleton.scale.x = 0.03
  skeleton.color.r = 1.0
  skeleton.color.g = 0.0
  skeleton.color.b = 0.0
  skeleton.color.a = 1.0

  indexes = keypoints.keys()
  for segment in mp.solutions.pose.POSE_CONNECTIONS:
    idx1, idx2 = segment
    if idx1 in indexes and idx2 in indexes:
      keypoint_1 = keypoints[idx1]
      keypoint_2 = keypoints[idx2]
      x1, y1, z1 = keypoint_1
      x2, y2, z2 = keypoint_2
      skeleton.points.append(Point(x = float(x1), y = float(y1), z = float(z1)))
      skeleton.points.append(Point(x = float(x2), y = float(y2), z = float(z2)))
  return skeleton

def create_marker_msg(id, x, y, z, header):
  marker = Marker()
  marker.header = header

  marker.ns = 'skeleton'
  marker.id = id
  marker.type = Marker.SPHERE
  
  marker.action = Marker.ADD
  marker.scale.x = 0.03
  marker.scale.y = 0.03
  marker.scale.z = 0.03

  marker.color.r = 1.0
  marker.color.g = 0.0
  marker.color.b = 0.0
  marker.color.a = 1.0

  marker.pose.position.x = float(x)
  marker.pose.position.y = float(y)
  marker.pose.position.z = float(z)
  
  return marker

def build_marker_array_msg(keypoints, header):
  marker_array = MarkerArray()
  for index, keypoint in keypoints.items():
    marker = create_marker_msg(index, 
                               keypoint[0], 
                               keypoint[1], 
                               keypoint[2], 
                               header)
    marker_array.markers.append(marker)
  return marker_array
