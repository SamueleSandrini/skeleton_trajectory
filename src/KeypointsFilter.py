#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, TwistStamped, Transform, Vector3, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R
from KalmanFilter import KalmanFilter
import time
import mediapipe as mp

CAMERA_FRAME="camera_color_optical_frame"

SKELETON_MARKER_SUB = '/skeleton_marker'
KEYPOINTS_FILTERED = '/keypoints_filtered'
SKELETON_FILTERED = '/skeleton_filtered'
SKELETON_FILTERED_ARRAY = '/poses'
KEYPOINT_VELOCITY = '/keypoint_velocity'
VARIANCE_MARKER = '/marker_variance'

N_KEYPOINTS = 33

class KeypointsFilter():
    def __init__(self):
        """
        Class builder
        """
        self.isKeypointTracked = np.zeros((N_KEYPOINTS,),dtype=bool)
        self.keypointsFilters = np.empty((N_KEYPOINTS,),dtype=object)
        self.timeKeypoint = np.zeros((N_KEYPOINTS,))

        #Create array of KalmanFilter object one for each keypoint
        for k in range(0,N_KEYPOINTS):
            self.keypointsFilters[k]=KalmanFilter()

        #Subscriber
        self.subKeypoints = None

        #Message general definition
        self.marker = Marker()
        self.marker.type = Marker.SPHERE
        self.marker.ns = "KeypointsFiltered"
        self.marker.header.frame_id = CAMERA_FRAME
        self.marker.header.stamp = rospy.Time.now()
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.04
        self.marker.scale.y = 0.04
        self.marker.scale.z = 0.04

        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0

        self.marker.color.a =1.0

        #Marker Variance
        self.markerVariance = Marker()
        self.markerVariance.type = Marker.SPHERE
        self.markerVariance.ns = "KeypointsFilteredVariance"
        self.markerVariance.header.frame_id = CAMERA_FRAME
        self.markerVariance.header.stamp = rospy.Time.now()
        self.markerVariance.action = Marker.ADD
        self.pubMarkerVariance = rospy.Publisher(VARIANCE_MARKER, Marker, queue_size = 100)

        #Publisher keypoint filtered
        self.pubMarkerFiltered = rospy.Publisher(KEYPOINTS_FILTERED, Marker, queue_size = 100)

        """
        Da qui in poi
        """
        self.listKeypoints = []
        self.listOfIndexesPres = []


        self.pubSkeletonFiltered = rospy.Publisher(SKELETON_FILTERED, Marker, queue_size = 100)
        self.pubSkeletonFilteredArray = rospy.Publisher(SKELETON_FILTERED_ARRAY,PoseArray, queue_size = 100)

        #Publisher Keypoint velocity
        self.pubKeypointVelocity = rospy.Publisher(KEYPOINT_VELOCITY, TwistStamped, queue_size = 100)


        # Mediapipe Utils and pose
        self.mp_pose = mp.solutions.pose

    def callbackKeypoint(self, keypoints):
        """
        Callback method to retrieve keypoints and apply Kalman Filter
        @param: keypoint: ROS Marker message
        """
        self.skeleton = Marker()
        self.skeleton.type = Marker.LINE_LIST
        self.skeleton.ns = "SegmentsFiltered"
        self.skeleton.header.frame_id = CAMERA_FRAME

        self.skeleton.id=100
        self.skeleton.action = Marker.ADD
        self.skeleton.scale.x=0.03
        self.skeleton.color.r = 0.0
        self.skeleton.color.g = 2.0
        self.skeleton.color.b = 0.0
        self.skeleton.color.a =0.7

        for keypoint in keypoints.markers:

            idKeypoint=keypoint.id

            #Measurements
            y=np.ones((3,))*[keypoint.pose.position.x,keypoint.pose.position.y,keypoint.pose.position.z]
            #Time of this keypoints
            self.timeKeypoint[idKeypoint]=time.time()

            #Check if time is too old
            for id,singleTime in enumerate(self.timeKeypoint):
                #print(singleTime)
                if singleTime!=0.0:
                    deltaT=time.time()-singleTime
                else:
                    deltaT=0
                #print("Id Keypoint: {}, tempo:{}".format(idKeypoint,deltaT))
                if deltaT>1.0/10.0:
                    if self.isKeypointTracked[id]:
                        self.isKeypointTracked[id]=False
                        print("Keypoint number: {} too old, time:{}".format(id,deltaT))
                        print("DeltaT: ",deltaT)
                    else:
                        pass
                        #print("Keypoint number: {} già troppo vecchio:".format(id))

            #Apply Kalaman Filter if tracked or initialize it if first time
            if not self.isKeypointTracked[idKeypoint]:
                self.isKeypointTracked[idKeypoint]=self.keypointsFilters[idKeypoint].initialize(y)
                yFiltered = self.keypointsFilters[idKeypoint].getYAfterInitialize()                 #Actually it is not filtered but in this way only one Pose msg.
                print("Keypoint number: {} is re-initialized".format(idKeypoint))
                #self.isKeypointTracked[idKeypoint]=True                                 #After initialization it will be tacked
            else:
                yFiltered = self.keypointsFilters[idKeypoint].update(y,idKeypoint)
                #print("Update keypoint: {}".format(idKeypoint))

            #Show velocity vector
            self.showReferenceFrame(np.eye(3),yFiltered,CAMERA_FRAME,'KeypointFrame'+str(idKeypoint))
            self.pubKeypointVelocity.publish(self.createTwistMessage(self.keypointsFilters[idKeypoint].getCartesianVelocity(),idKeypoint))

            #Calcolo varianza
            print("Varianza posizione:",self.keypointsFilters[idKeypoint].getPosDevSt())
            self.markerVariance = Marker()
            self.markerVariance.header.stamp = rospy.Time.now()
            self.markerVariance.id = idKeypoint
            self.markerVariance.pose = Pose(Point(yFiltered[0],yFiltered[1],yFiltered[2]),Quaternion(self.keypointsFilters[idKeypoint].getPosDevSt()[0],self.keypointsFilters[idKeypoint].getPosDevSt()[1],self.keypointsFilters[idKeypoint].getPosDevSt()[2],1))
            self.pubMarkerVariance.publish(self.markerVariance)

            #Quando ok sia calcolo velocity che varianza da mettere anche nel for di quelli in ope-loop

            #Populate message (this is message of keypoints that are seen by mediapipe and not tacked in open loop)
            self.marker.header.stamp = rospy.Time.now()
            self.marker.id = idKeypoint
            self.marker.pose = Pose(Point(yFiltered[0],yFiltered[1],yFiltered[2]),Quaternion(0,0,0,1))

            self.pubMarkerFiltered.publish(self.marker)
            """
            Da qui in poi
            """

            self.listOfIndexesPres.append(idKeypoint)
            self.listKeypoints.append(self.marker.pose)

        self.skeleton.header.stamp = rospy.Time.now()

        #Iterate all non detected keypoints and update it in open loop if the time passed is not too much (approximate 3 samples)
        for idKeypointNotDetected in np.setdiff1d(np.array(range(0,N_KEYPOINTS)),self.listOfIndexesPres):
            if self.isKeypointTracked[idKeypointNotDetected]:
                yModel = self.keypointsFilters[idKeypointNotDetected].updateOpenLoop()  #It returns
                #Populate message
                self.marker.header.stamp = rospy.Time.now()
                self.marker.id = idKeypointNotDetected
                self.marker.pose = Pose(Point(yModel[0],yModel[1],yModel[2]),Quaternion(0,0,0,1))
                self.pubMarkerFiltered.publish(self.marker)
                self.listOfIndexesPres.append(idKeypointNotDetected)
                self.listKeypoints.append( Pose(Point(yModel[0],yModel[1],yModel[2]),Quaternion(0,0,0,1)))
                print("Keypoint number: {} is in open Loop".format(idKeypointNotDetected))
                #Calcolo varianza
                print("Varianza posizione:",self.keypointsFilters[idKeypoint].getPosDevSt())
                self.markerVariance = Marker()
                self.markerVariance.header.stamp = rospy.Time.now()
                self.markerVariance.id = idKeypointNotDetected
                self.markerVariance.pose = Pose(Point(yModel[0],yModel[1],yModel[2]),Quaternion(self.keypointsFilters[idKeypointNotDetected].getPosDevSt()[0],self.keypointsFilters[idKeypointNotDetected].getPosDevSt()[1],self.keypointsFilters[idKeypointNotDetected].getPosDevSt()[2],1))
                self.pubMarkerVariance.publish(self.markerVariance)
        #Create pose array of all filtered KeyPoints
        skeletonArrayFiltered = PoseArray()
        skeletonArrayFiltered.header.stamp = rospy.Time.now()
        skeletonArrayFiltered.header.frame_id = CAMERA_FRAME

        #print(self.listKeypoints)
        skeletonArrayFiltered.poses = self.listKeypoints
        self.pubSkeletonFilteredArray.publish(skeletonArrayFiltered)
        #Iterate connections between keypoints: skeleton information (plot)
        n=0

        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if (start_idx in self.listOfIndexesPres) and (end_idx in self.listOfIndexesPres):
                 #self.skeleton.id = n
                 index_start=self.listOfIndexesPres.index(start_idx)              # 1 2 10 21 : i don't have to go in 21th element of landmark List but 4th
                 index_end=self.listOfIndexesPres.index(end_idx)
                 self.skeleton.points.append(self.listKeypoints[index_start].position)
                 self.skeleton.points.append(self.listKeypoints[index_end].position)
                 n+=1
        self.pubSkeletonFiltered.publish(self.skeleton)

        #Reset presences when one skeleton is completed
        self.listOfIndexesPres = []
        self.listKeypoints = []


    def subAndFilter(self):
        self.subscriberKeypoint=rospy.Subscriber(SKELETON_MARKER_SUB,MarkerArray,self.callbackKeypoint)
        rospy.spin()
    def createTwistMessage(self,cartesianVelocityVector,idKeyp):

        msg=TwistStamped()


        msg.header.frame_id='KeypointFrame'+str(idKeyp)
        msg.header.stamp = rospy.Time.now()

        msg.twist.linear.x = cartesianVelocityVector[0]
        msg.twist.linear.y = cartesianVelocityVector[1]
        msg.twist.linear.z = cartesianVelocityVector[2]
        msg.twist.angular.x = 0
        msg.twist.angular.y = 0
        msg.twist.angular.z = 0
        return msg
    def showReferenceFrame(self,matR,tran,originFrame,destinationFrame):
        """
        Method for show (publish) a reference frame
        @param matR: rotation matrix
        @param tran: translation Vector
        @param originFrame: base frame
        @param destinationFrame: child frame
        """
        t0=Transform()
        t0TS=TransformStamped()
        r = R.from_matrix(matR)
        quatR=r.as_quat()
        t0.rotation=Quaternion(quatR[0],quatR[1],quatR[2],quatR[3])
        t0.translation=Vector3(tran[0],tran[1],tran[2])
        t0TS.header.frame_id=originFrame
        t0TS.header.stamp= rospy.Time.now()
        t0TS.child_frame_id=destinationFrame
        t0TS.transform=t0
        br = tf2_ros.TransformBroadcaster()
        br.sendTransform(t0TS)
