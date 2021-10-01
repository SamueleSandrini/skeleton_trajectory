#! /usr/bin/env python

import time
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int64
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pyrealsense2 as rs2
from math import floor

# Realsense Topic
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'

#Publisher node skeleton
SKELETON_PUB = '/skeleton'
SKELETON_MARKER_PUB = '/skeleton_marker'
SKELETON_PUB_ARRAY = '/skeleton_marker_PoseArray'
#Publisher for frequency measurement
FREQ_MEAS_PUB = 'iteration'

SKELETON_FILTERED_ARRAY = '/poses'
SKELETON_FILTERED = '/skeleton_filtered'
KEYPOINTS_FILTERED = '/keypoints_filtered'

CAMERA_FRAME = "camera_color_optical_frame"

# Costant loginfo
PARAMETERS_LOG = 'Camera Parameters acquired \n  Parameters:{}'

class RealSense():
    """
    RealSense class for Subscribe interesting topic.
    """

    def __init__(self):
        """
        Class builder
        @param -
        @return RealSense RealSense object
        """
        self.bridge = CvBridge()
        self.colorFrame = None
        self.depthFrame = None

        # Mediapipe Utils and pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose=self.mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5)

        # Frequency measurement
        self.pubIterations = rospy.Publisher(FREQ_MEAS_PUB, Int64, queue_size=10)
        self.iterations=0
        # Publischer skeleton KeyPoints and connections
        self.pubMarker = rospy.Publisher(SKELETON_MARKER_PUB, MarkerArray, queue_size = 100)
        self.pubSkeleton = rospy.Publisher(SKELETON_PUB, Marker, queue_size = 100)
        #It can be usefull for compatibility

        # Gestione camera pyrealsense2
        self.intrinsics = None
        self.cameraInfoReceived = False

        #Publisher usefull for delate
        self.pubSkeletonFilteredArray = rospy.Publisher(SKELETON_FILTERED_ARRAY,PoseArray, queue_size = 100)
        self.pubSkeletonFiltered = rospy.Publisher(SKELETON_FILTERED, Marker, queue_size = 100)
        self.pubMarkerFiltered = rospy.Publisher(KEYPOINTS_FILTERED, Marker, queue_size = 100)



    def callback(self,frameRgb,frameDepth):
        """
        Callback method to retrieve the content of the topic and convert it in cv2 format. Identify human KeyPoints.
        @param frameRgb : camera msg rgb
        @param frameDepth : camera msg depth
        """
        # Convertion from ros msg image to cv2 image
        self.colorFrame = self.bridge.imgmsg_to_cv2(frameRgb, desired_encoding="passthrough")
        self.depthFrame = self.bridge.imgmsg_to_cv2(frameDepth, desired_encoding="passthrough")
        self.frameDistance = self.bridge.imgmsg_to_cv2(frameDepth, desired_encoding="32FC1")

        """
        cv_image_array = np.array(self.frameDistance, dtype = np.dtype('f8'))
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)

        depth_rgb = cv2.merge([cv_image_norm,cv_image_norm,cv_image_norm])
        """
        #cv2.cvtColor(cv_image_norm,cv2.COLOR_GRAY2RGB)



        # KeyPoints Human identification
        results = self.pose.process(self.colorFrame)

        """
        skeleton = PoseArray()
        skeleton.header.stamp = rospy.Time.now()
        skeleton.header.frame_id = self.frame_id
        """

        skeleton = Marker()
        skeleton.type = Marker.LINE_LIST
        skeleton.ns = "Segments"
        skeleton.header.frame_id = self.frame_id
        skeleton.header.stamp = rospy.Time.now()
        skeleton.action = Marker.ADD
        skeleton.scale.x=0.03
        skeleton.color.r = 1.0
        skeleton.color.g = 0.0
        skeleton.color.b = 0.0
        skeleton.color.a =1.0

        #skeletonArray = PoseArray()
        #skeletonArray.header.stamp = rospy.Time.now()
        #skeletonArray.header.frame_id = self.frame_id

        markerArray=MarkerArray()

        #Check if some pose is detected
        if results.pose_landmarks is not None:
            #Iterate all landmarks
            indexesPres=[]
            landmarkList=[]

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                #Check if visibility is enought and if x and y is inside the area (mediapipe estimetes also outside es. 1.1 )
                if landmark.visibility > 0.8 and landmark.x<1 and landmark.y < 1 and landmark.x>0 and landmark.y>0:
                    # De-normalization
                    x = floor(self.intrinsics.width * landmark.x)    # Column
                    y = floor(self.intrinsics.height * landmark.y)  # Row

                    depthPixel = self.frameDistance[y,x]

                    if (depthPixel<0.1):
                        print("Keypoint number: ", idx, "has zero depth.")
                        continue

                    # Deprojection : Image frame -> Camera frame (camera_color_optical_frame)
                    deprojection=rs2.rs2_deproject_pixel_to_point(self.intrinsics,[x,y], depthPixel)

                    # Add index marker on list of presences
                    indexesPres.append(idx)
                    landmarkList.append(deprojection)
                    #Populate message

                    # Marker msg definition
                    marker = Marker()
                    marker.type = Marker.SPHERE
                    marker.ns = "Skeleton"
                    marker.header.frame_id = self.frame_id

                    marker.action = Marker.ADD
                    marker.scale.x = 0.03
                    marker.scale.y = 0.03
                    marker.scale.z = 0.03

                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0

                    marker.color.a =1.0
                    marker.id = idx
                    #print(marker.id)
                    marker.header.stamp = rospy.Time.now()
                    marker.pose = Pose(Point(deprojection[0]/1000.0,deprojection[1]/1000.0,deprojection[2]/1000.0),Quaternion(0,0,0,1))

                    markerArray.markers.append(marker)
                    #print(markerArray)
            #print(landmarkList)
            #print(indexesPres)

            #print(markerArray)
            #Check based on bust if it is a person
            if self.isNotPerson(indexesPres,landmarkList):
                #print("Non è una persona")
                self.mp_pose = mp.solutions.pose    #Reintialize pose detection to avoid tracking effect
            else:
                #print("E' una persona")
                # Publish all keypoints
                self.pubMarker.publish(markerArray)


                n=0
                for connection in self.mp_pose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if (start_idx in indexesPres) and (end_idx in indexesPres):
                         index_start=indexesPres.index(start_idx)              # 1 2 10 21 : i don't have to go in 21th element of landmark List but 4th
                         index_end=indexesPres.index(end_idx)
                         skeleton.points.append(Point(landmarkList[index_start][0]/1000,landmarkList[index_start][1]/1000,landmarkList[index_start][2]/1000))
                         skeleton.points.append(Point(landmarkList[index_end][0]/1000,landmarkList[index_end][1]/1000,landmarkList[index_end][2]/1000))
                         n+=1
                self.pubSkeleton.publish(skeleton)

                # Draw skeleton on image  ---  Note: (It use reference to the variable so if the var points at another image all of these are changed)
                self.mp_drawing.draw_landmarks(self.colorFrame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                #self.mp_drawing.draw_landmarks(depth_rgb, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)


        else:
            #Create pose array of all filtered KeyPoints empy
            skeletonArrayFiltered = PoseArray()
            skeletonArrayFiltered.header.stamp = rospy.Time.now()
            skeletonArrayFiltered.header.frame_id = CAMERA_FRAME
            self.pubSkeletonFilteredArray.publish(skeletonArrayFiltered)
            # Message costructor for all segments Filtering
            self.skeleton = Marker()
            self.skeleton.type = Marker.LINE_LIST
            self.skeleton.ns = "SegmentsFiltered"
            self.skeleton.header.frame_id = CAMERA_FRAME
            self.skeleton.action = Marker.DELETE
            self.skeleton.id=100
            self.pubSkeletonFiltered.publish(self.skeleton)
            #Delete all KeyPoints
            for k in range(0,33):
                self.marker = Marker()
                self.marker.type = Marker.SPHERE
                self.marker.ns = "KeypointsFiltered"
                self.marker.header.frame_id = CAMERA_FRAME
                self.marker.header.stamp = rospy.Time.now()
                self.marker.id=k
                self.marker.action = Marker.DELETE
                self.pubMarkerFiltered.publish(self.marker)

        """
        img = cv2.cvtColor(self.colorFrame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Rgb',img)

        cv2.imshow("Depth RGB",depth_rgb)
        # Check if a kay is pressed
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            self.stop()                 # Unregister all subscriber
        """
        # Publisher for measure frequency
        self.pubIterations.publish(self.iterations)
        self.iterations=self.iterations+1


    def callbackOnlyRgb(self,frameRgb):
        self.colorFrame = self.bridge.imgmsg_to_cv2(frameRgb, desired_encoding="passthrough")

    def cameraInfoCallback(self,cameraInfo):
        """
        Callback for get Intrinsic Parameter of Camera and create intrinsics object (pyrealsense2 library)
        """
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]

        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]
        self.cameraInfoReceived = True

        #Reference frame
        self.frame_id=cameraInfo.header.frame_id
        print(self.frame_id)

    def waitCameraInfo(self):
        while not self.cameraInfoReceived:
            pass
        self.sub_info.unregister()
        rospy.loginfo(PARAMETERS_LOG.format(self.intrinsics))

    def acquire(self):
        """
        Method for acquiring in syncronization way rgb and depth frame
        """
        print("Dentro Acquire")
        self.subcriberColorFrame = message_filters.Subscriber(COLOR_FRAME_TOPIC, Image)
        self.subcriberDepthFrame = message_filters.Subscriber(DEPTH_ALIGNED_TOPIC, Image)
        # Subscriber Synchronization
        subSync = message_filters.TimeSynchronizer([self.subcriberColorFrame, self.subcriberDepthFrame], queue_size=10)
        #Call callback sincronized
        subSync.registerCallback(self.callback)

        rospy.spin()


    def acquireOnlyRgb(self):
        """
        Method for acquiring in syncronization way rgb
        """
        self.subcriberColor= rospy.Subscriber(COLOR_FRAME_TOPIC, Image, self.callbackOnlyRgb, queue_size=1)

    def showImage(self,nameWindowRgb,nameWindowDepth):
        """
        Method for showing the image
        """
        #Rgb -> Bgr convertion for cv2 imshow
        imgImshow = cv2.cvtColor(realsense.colorFrame, cv2.COLOR_RGB2BGR)
        cv2.imshow(nameWindowRgb, imgImshow)
        cv2.imshow(nameWindowDepth,self.depthFrame)
    def getCameraParam(self):
        self.sub_info = rospy.Subscriber(CAMERA_INFO_TOPIC,CameraInfo,self.cameraInfoCallback)
    def stop(self):
        '''Method to disconnect the subscribers from kinect2_bridge topics, to release
            some memory and avoid filling up the queue.'''
        self.subcriberColorFrame.unregister()
        self.subcriberDepthFrame.unregister()

    def isNotPerson(self,indexesPres,landmarkList):
        """
        Method for check if the measured link can be of a person.
        """
        if ((self.mp_pose.PoseLandmark.LEFT_SHOULDER.value in indexesPres) and (self.mp_pose.PoseLandmark.LEFT_HIP.value in indexesPres)):
            idx_lShoulder=indexesPres.index(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            idx_lHip=indexesPres.index(self.mp_pose.PoseLandmark.LEFT_HIP.value)
            leftBust=np.linalg.norm(np.array(landmarkList[idx_lShoulder])-np.array(landmarkList[idx_lHip]))
            #print("Dimensione Busto Left: {}".format(leftBust))
            if leftBust<250 or leftBust>1200:
                return True
        if ((self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value in indexesPres) and (self.mp_pose.PoseLandmark.RIGHT_HIP.value in indexesPres)):
            idx_rShoulder=indexesPres.index(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            idx_rHip=indexesPres.index(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
            rigthBust=np.linalg.norm(np.array(landmarkList[idx_rShoulder])-np.array(landmarkList[idx_rHip]))
            #print("Dimensione Busto Rigth: {}".format(rigthBust))
            if rigthBust<250 or rigthBust>1200:
                return True
        if len(indexesPres)<2:
            print("Keypoints less then 2")
            return True
        """
        print(indexesPres)
        print (self.mp_pose.PoseLandmark.RIGHT_HIP.value in indexesPres)
        print(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        print (self.mp_pose.PoseLandmark.RIGHT_KNEE.value in indexesPres)
        print(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        print(((self.mp_pose.PoseLandmark.RIGHT_KNEE.value in indexesPres) and (self.mp_pose.PoseLandmark.RIGHT_HIP.value in indexesPres)))
        print (self.mp_pose.PoseLandmark.LEFT_HIP.value in indexesPres)
        print(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        print (self.mp_pose.PoseLandmark.LEFT_KNEE.value in indexesPres)
        print(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        """
        #print("Numero di Keypoints:",len(indexesPres))
        """
        if ((self.mp_pose.PoseLandmark.RIGHT_KNEE.value in indexesPres) and (self.mp_pose.PoseLandmark.RIGHT_HIP.value in indexesPres)):
            print("Dentro Femore Right")
            idx_rKnee=indexesPres.index(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
            idx_rHip=indexesPres.index(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
            rightFemur=np.linalg.norm(np.array(landmarkList[idx_rKnee])-np.array(landmarkList[idx_rHip]))
            #print("Keypoint 26 Ginocchio: [{},{},{}]\n Keypoint 24 Anca: [{},{},{}]".format(landmarkList[idx_rKnee][0],landmarkList[idx_rKnee][1],landmarkList[idx_rKnee][2],landmarkList[idx_rHip][0],landmarkList[idx_rHip][1],landmarkList[idx_rHip][2]))
            print("Dimensione Femore Right: {}".format(rightFemur))
            if rightFemur<250 or rightFemur>500:
                return True
        if ((self.mp_pose.PoseLandmark.LEFT_KNEE.value in indexesPres) and (self.mp_pose.PoseLandmark.LEFT_HIP.value in indexesPres)):
            print("Dentro Femore Left")
            idx_lKnee=indexesPres.index(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
            idx_lHip=indexesPres.index(self.mp_pose.PoseLandmark.LEFT_HIP.value)
            leftFemur=np.linalg.norm(np.array(landmarkList[idx_lKnee])-np.array(landmarkList[idx_lHip]))
            #print("Keypoint 26 Ginocchio: [{},{},{}]\n Keypoint 24 Anca: [{},{},{}]".format(landmarkList[idx_rKnee][0],landmarkList[idx_rKnee][1],landmarkList[idx_rKnee][2],landmarkList[idx_rHip][0],landmarkList[idx_rHip][1],landmarkList[idx_rHip][2]))
            print("Dimensione Femore Left: {}".format(leftFemur))
            if leftFemur<250 or leftFemur>500:
                return True
        """
        return False
